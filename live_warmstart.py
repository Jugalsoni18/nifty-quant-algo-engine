"""
live_warmstart.py
=====================
Warm-starts the live bot's buffer AND the V12Model's internal HTF/pivot
state with recent history, so the bot can make properly-calibrated
predictions from its FIRST live bar instead of needing ~20 trading days
of live-accumulated ticks before the long-window rank features
(price_position_rank, vix_ma20, adx_1m_rank) become accurate.

CRITICAL — READ BEFORE CHANGING WARMSTART_MODE
---------------------------------------------
An earlier revision of this file claimed the training data used a
PINNED-strike reconstruction, and treated the rolling-ATM behaviour as
a "flip artifact" that had been "fixed in the training data". That claim
is FALSE for the file training actually consumed, and it produced a
warm-start that fed the live bot a series the model had never seen.

Measured directly on data/NIFTY_1min_rich_iv_filled.csv (offset-0 leg) by
call-put parity: options price off the FUTURE, so C(K) - P(K) = F - K with
F = S + basis, i.e. basis = K - S + (C - P). Forcing K = round(S/50)*50
(rather than a fixed weekly pin) is the only hypothesis that keeps the
implied basis in a physically plausible weekly-futures band; the pinned
hypothesis requires the "basis" to drift to +288 pts (1.6% of index) by
week's end. The CSV is therefore PER-BAR DYNAMIC, and the 4.97-5.29%
"flip rate" is just spot crossing a 25-pt half-step every 20-60 bars --
the rolling process itself, which is what the model learned.

WARMSTART_MODE = "dynamic" (default) is therefore the correct setting:
it uses the raw rolling-ATM fetch, which is what matches training.
Setting it to "pinned" restores the old behaviour for comparison only --
it does NOT match the model and will skew every feature computed from
call_close/put_close.

TWO THINGS THIS WARM-STARTS, NOT JUST ONE:
  1. The buffer dict (call/put OHLCV, IV, OI, spot, india_vix) -- via
     warm_start_buffer() from buffer_warmstart.py.
  2. model.htf and model.pivot -- V12Model's OWN internal rolling state,
     separate from the buffer. Replaying only the buffer and skipping
     this would leave the model's HTF/pivot state cold even though the
     buffer looks warm -- a real, separate gap from the buffer-size fix.

AUTOMATIC: computes its own date range (today back N calendar days,
covering the target trading-day window with margin), and requires no
manual date/expiry input -- called once at bot startup before the feed
connects.
"""

import time as time_mod
from datetime import date, timedelta, time as dtime

import numpy as np
import pandas as pd
import requests

from buffer_warmstart import (
    BUFFER_SIZE, FULL_WINDOW, build_extended_buffer, warm_start_buffer,
    load_historical_bars,
)

BASE_URL = "https://api.dhan.co/v2"
LAST_PIN_STRIKES = {}
STRIKE_OFFSETS = list(range(-10, 11))  # ATM-10..ATM+10, matching training columns
CHUNK_DAYS = 14
SLEEP_SEC = 0.6
TARGET_TRADING_DAYS = 22   # slightly above FULL_WINDOW/375=20, for margin
LOOKBACK_CALENDAR_DAYS = 32  # comfortably covers TARGET_TRADING_DAYS incl. weekends/holidays

# Cache range fetches so the rule-only pin reconstruction reuses the dynamic
# warm-start ATM and ATM+1 CALL requests instead of downloading them again.
_FETCH_RANGE_CACHE = {}


def _headers(client_id, access_token):
    return {"Content-Type": "application/json", "Accept": "application/json",
            "access-token": access_token, "client-id": client_id}

def _offset_label(off):
    return "ATM" if off == 0 else f"ATM{'+' if off > 0 else ''}{off}"

def _date_chunks(start, end, days):
    cur = start
    while cur <= end:
        yield cur, min(cur + timedelta(days=days - 1), end)
        cur += timedelta(days=days)

def _fetch_relative_strike(client_id, access_token, strike_label, drv_type, frm, to, log=None):
    api_to = to + timedelta(days=1)
    payload = {
        "exchangeSegment": "NSE_FNO", "interval": "1", "securityId": "13",
        "instrument": "OPTIDX", "expiryFlag": "WEEK", "expiryCode": 1,
        "strike": strike_label, "drvOptionType": drv_type,
        "requiredData": ["open", "high", "low", "close", "iv", "volume", "oi", "spot", "strike"],
        "fromDate": frm.strftime("%Y-%m-%d"), "toDate": api_to.strftime("%Y-%m-%d"),
    }
    for attempt in range(1, 4):
        try:
            r = requests.post(f"{BASE_URL}/charts/rollingoption", json=payload,
                              headers=_headers(client_id, access_token), timeout=60)
            if r.status_code == 429:
                time_mod.sleep(20); continue
            if r.status_code != 200:
                if log: log.warning(f"live_warmstart: HTTP {r.status_code} [{drv_type} {strike_label}]: {r.text[:200]}")
                return pd.DataFrame()
            js = r.json()
            key = "ce" if drv_type == "CALL" else "pe"
            node = js.get("data", {}).get(key)
            if not node or not node.get("timestamp"):
                return pd.DataFrame()
            n = len(node["timestamp"])
            df = pd.DataFrame({
                "timestamp": node.get("timestamp", [None]*n), "open": node.get("open", [None]*n),
                "high": node.get("high", [None]*n), "low": node.get("low", [None]*n),
                "close": node.get("close", [None]*n), "volume": node.get("volume", [0]*n),
                "iv": node.get("iv", [None]*n), "oi": node.get("oi", [0]*n),
                "actual_strike": node.get("strike", [None]*n), "spot": node.get("spot", [None]*n),
            })
            df["datetime"] = (pd.to_datetime(df["timestamp"], unit="s", utc=True)
                               .dt.tz_convert("Asia/Kolkata").dt.tz_localize(None))
            return df
        except Exception as e:
            if log: log.warning(f"live_warmstart: attempt {attempt}/3 error: {e}")
            time_mod.sleep(3 * attempt)
    return pd.DataFrame()

def _fetch_full_range(client_id, access_token, strike_label, drv_type, start, end, log=None):
    cache_key = (strike_label, drv_type, start, end)
    cached = _FETCH_RANGE_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()

    frames = []
    for frm, to in _date_chunks(start, end, CHUNK_DAYS):
        df = _fetch_relative_strike(
            client_id, access_token, strike_label, drv_type, frm, to, log
        )
        time_mod.sleep(SLEEP_SEC)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["timestamp"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    _FETCH_RANGE_CACHE[cache_key] = out.copy()
    return out


def fetch_historical_vix(start_date, end_date, log=None):
    """Fetch daily India VIX closes from NSE historical VIX API.

    NSE's newer historical route is attempted first:
        /api/historicalOR/vixhistory?from=DD-MM-YYYY&to=DD-MM-YYYY
    The older /api/historical/vixhistory route is retained as a fallback.

    The result is a DataFrame with columns [date, india_vix], or None if all
    attempts fail. The caller can then use its existing Zerodha/Dhan VIX
    backfill rather than crashing or silently treating a partial series as
    complete.

    Training used one VIX value per calendar day, forward-filled over that
    day's minute bars, so this intentionally returns daily closes rather than
    intraday VIX candles.
    """
    try:
        from requests.adapters import HTTPAdapter

        session = requests.Session()
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/",
            "Origin": "https://www.nseindia.com",
            "X-Requested-With": "XMLHttpRequest",
        })

        # The main bot may globally install an HTTPAdapter that itself retries
        # NSE 5xx responses. Disable adapter-level retries for this session's
        # API route so the explicit retry loop below controls the total attempt
        # count and diagnostics (otherwise retries become nested/multiplied).
        session.mount(
            "https://www.nseindia.com/api/",
            HTTPAdapter(max_retries=0),
        )

        # Prime the same session to obtain NSE cookies before querying history.
        try:
            prime = session.get("https://www.nseindia.com/", timeout=10)
            if log and prime.status_code >= 400:
                log.warning(
                    f"fetch_historical_vix: NSE session-prime HTTP "
                    f"{prime.status_code}; will still try historical endpoint"
                )
        except requests.RequestException as exc:
            if log:
                log.warning(
                    f"fetch_historical_vix: NSE session-prime failed ({exc}); "
                    "will still try historical endpoint"
                )

        # Route order: current documented/used route first, then the legacy
        # route as compatibility fallback if NSE rejects or breaks the first.
        urls = (
            "https://www.nseindia.com/api/historicalOR/vixhistory",
            "https://www.nseindia.com/api/historical/vixhistory",
        )
        records = []

        for frm, to in _date_chunks(start_date, end_date, 14):
            params = {
                "from": frm.strftime("%d-%m-%Y"),
                "to": to.strftime("%d-%m-%Y"),
            }
            chunk_ok = False
            last_error = "no response"

            # Try the preferred route first, and legacy route only if all
            # explicit attempts against the preferred route fail.
            for url in urls:
                for attempt in range(1, 4):
                    try:
                        response = session.get(url, params=params, timeout=15)
                        if response.status_code != 200:
                            last_error = f"HTTP {response.status_code}"
                            if log:
                                log.warning(
                                    f"fetch_historical_vix: {last_error} "
                                    f"for {params['from']} to {params['to']} "
                                    f"endpoint={url} body={response.text[:250]!r} "
                                    f"(attempt {attempt}/3)"
                                )
                        else:
                            try:
                                payload = response.json()
                            except ValueError as exc:
                                payload = None
                                last_error = f"HTTP 200 but invalid JSON ({exc})"

                            data = payload.get("data") if isinstance(payload, dict) else None
                            if isinstance(data, list):
                                records.extend(data)
                                chunk_ok = True
                                if log:
                                    log.info(
                                        f"fetch_historical_vix: {len(data)} rows for "
                                        f"{params['from']} to {params['to']} via {url}"
                                    )
                                break

                            if payload is not None:
                                last_error = "HTTP 200 but response has no list-valued data field"
                            if log:
                                body_preview = repr(response.text[:250])
                                log.warning(
                                    f"fetch_historical_vix: {last_error} for "
                                    f"{params['from']} to {params['to']} "
                                    f"endpoint={url} body={body_preview} "
                                    f"(attempt {attempt}/3)"
                                )
                    except requests.RequestException as exc:
                        last_error = f"request error: {exc}"
                        if log:
                            log.warning(
                                f"fetch_historical_vix: {last_error} for "
                                f"{params['from']} to {params['to']} "
                                f"endpoint={url} (attempt {attempt}/3)"
                            )

                    if attempt < 3:
                        time_mod.sleep(2 ** (attempt - 1))

                if chunk_ok:
                    break
                if log:
                    log.warning(
                        f"fetch_historical_vix: endpoint exhausted for "
                        f"{params['from']} to {params['to']} endpoint={url}; "
                        f"last_error={last_error}."
                    )

            if not chunk_ok:
                if log:
                    log.warning(
                        f"fetch_historical_vix: all endpoints/retries exhausted for "
                        f"{params['from']} to {params['to']} -- falling back to "
                        "constant-fill / broker backfill"
                    )
                return None

        if not records:
            if log:
                log.warning(
                    "fetch_historical_vix: endpoint returned no historical rows -- "
                    "falling back to constant-fill / broker backfill"
                )
            return None

        raw = pd.DataFrame(records)
        required = {"EOD_TIMESTAMP", "EOD_CLOSE_INDEX_VAL"}
        if not required.issubset(raw.columns):
            if log:
                log.warning(
                    "fetch_historical_vix: response schema missing required fields; "
                    f"columns={list(raw.columns)!r} -- falling back to broker backfill"
                )
            return None

        dates = pd.to_datetime(raw["EOD_TIMESTAMP"], errors="coerce", dayfirst=True)
        values = pd.to_numeric(raw["EOD_CLOSE_INDEX_VAL"], errors="coerce")
        vix_df = pd.DataFrame({
            "date": dates.dt.normalize(),
            "india_vix": values,
        })
        vix_df = (
            vix_df.dropna(subset=["date", "india_vix"])
                  .drop_duplicates(subset=["date"], keep="last")
                  .sort_values("date")
                  .reset_index(drop=True)
        )
        if vix_df.empty:
            if log:
                log.warning(
                    "fetch_historical_vix: no valid date/VIX pairs after parsing -- "
                    "falling back to broker backfill"
                )
            return None

        # Guard against an anti-bot or malformed payload being mistaken for VIX.
        valid = vix_df["india_vix"].between(3.0, 100.0)
        if not valid.all():
            if log:
                log.warning(
                    "fetch_historical_vix: discarding out-of-range VIX rows "
                    f"({int((~valid).sum())} invalid rows)"
                )
            vix_df = vix_df.loc[valid].reset_index(drop=True)
        if vix_df.empty:
            return None

        if log:
            log.info(
                f"fetch_historical_vix: got {len(vix_df)} daily VIX values "
                f"({vix_df['date'].min().date()} to {vix_df['date'].max().date()}); "
                f"range={vix_df['india_vix'].min():.2f}..{vix_df['india_vix'].max():.2f}"
            )
        return vix_df

    except Exception as exc:
        if log:
            log.warning(
                f"fetch_historical_vix: unexpected failure ({exc!r}) -- "
                "falling back to constant-fill / broker backfill",
                exc_info=True,
            )
        return None


def fetch_pinned_recent_data(client_id, access_token, log=None):
    """
    Fetches ATM-10..ATM+10 for the recent LOOKBACK_CALENDAR_DAYS window and
    reconstructs a PINNED-per-week series -- same methodology as
    build_pinned_strike_data.py, applied to a rolling recent window instead
    of a fixed historical range.
    """
    end = date.today()
    start = end - timedelta(days=LOOKBACK_CALENDAR_DAYS)
    if log: log.info(f"live_warmstart: fetching ATM-10..ATM+10, {start} to {end} "
                     f"({len(STRIKE_OFFSETS)*2} series)...")

    all_strikes = {}
    for off in STRIKE_OFFSETS:
        label = _offset_label(off)
        call_df = _fetch_full_range(client_id, access_token, label, "CALL", start, end, log)
        put_df = _fetch_full_range(client_id, access_token, label, "PUT", start, end, log)
        all_strikes[off] = {"CALL": call_df, "PUT": put_df}

    atm_call = all_strikes[0]["CALL"].copy()
    if atm_call.empty:
        if log: log.error("live_warmstart: no ATM CALL data fetched -- cannot warm-start")
        return None

    atm_call["_week"] = pd.to_datetime(atm_call["datetime"].dt.date).dt.to_period("W-TUE")
    pin_strikes = atm_call.groupby("_week").first()["actual_strike"].to_dict()
    LAST_PIN_STRIKES.update(pin_strikes)

    def build_pinned(drv_type):
        rows = []
        for off in STRIKE_OFFSETS:
            df = all_strikes[off][drv_type]
            if df.empty:
                continue
            df = df.copy()
            df["_week"] = pd.to_datetime(df["datetime"].dt.date).dt.to_period("W-TUE")
            df["_pin_strike"] = df["_week"].map(pin_strikes)
            rows.append(df[df["actual_strike"] == df["_pin_strike"]])
        if not rows:
            return pd.DataFrame()
        return pd.concat(rows, ignore_index=True).drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    pinned_call = build_pinned("CALL")
    pinned_put = build_pinned("PUT")
    if pinned_call.empty or pinned_put.empty:
        if log: log.error("live_warmstart: pinned reconstruction produced no data")
        return None

    merged = pd.merge(
        pinned_call[["datetime", "close", "high", "low", "open", "spot", "iv", "oi", "volume"]].rename(
            columns=lambda c: f"call_{c}" if c != "datetime" else c),
        pinned_put[["datetime", "close", "high", "low", "open", "iv", "oi", "volume"]].rename(
            columns=lambda c: f"put_{c}" if c != "datetime" else c),
        on="datetime", how="inner",
    )
    merged = merged.rename(columns={"call_spot": "spot_close"})

    vix_df = fetch_historical_vix(start, end, log)
    if vix_df is not None:
        merged["_date"] = pd.to_datetime(merged["datetime"].dt.date)
        merged = merged.merge(vix_df, left_on="_date", right_on="date", how="left")
        merged["india_vix"] = merged["india_vix"].ffill().bfill()
        merged = merged.drop(columns=["_date", "date"])
        if log: log.info(f"live_warmstart: real historical VIX merged in "
                         f"(range {merged['india_vix'].min():.2f}-{merged['india_vix'].max():.2f})")
    else:
        if log: log.warning("live_warmstart: VIX fetch failed -- india_vix will be constant-filled "
                            "at whatever INDIA_VIX was at startup for the ENTIRE warm-started "
                            "history (same degraded behavior as before this fix)")
    if log: log.info(f"live_warmstart: reconstructed {len(merged)} pinned rows "
                     f"({merged['datetime'].min()} to {merged['datetime'].max()})")
    return merged


# ─────────────────────────────────────────────────────────────────────────
# DYNAMIC-ATM WARM-START -- the correct one for this model.
# ─────────────────────────────────────────────────────────────────────────
# Module-level switch so the pinned path stays available for rollback /
# side-by-side comparison without editing code.
WARMSTART_MODE = "dynamic"     # "dynamic" | "pinned"


def fetch_dynamic_recent_data(client_id, access_token, log=None):
    """
    Fetches the DYNAMIC-ATM series for the recent LOOKBACK_CALENDAR_DAYS
    window: the option at round(spot/50)*50, RE-CHOSEN EVERY BAR.

    WHY THIS REPLACES fetch_pinned_recent_data()
    --------------------------------------------
    This module's docstring (and build_pinned_strike_data.py /
    niftybacktest4.py) assumed the training CSV had been rebuilt as a
    PINNED series and that the rolling-ATM behaviour was a "flip artifact"
    that had been "fixed" in the training data. That assumption is FALSE
    for the file training actually consumed.

    Measured directly on data/NIFTY_1min_rich_iv_filled.csv, offset-0 leg,
    via call-put parity. Options price off the FUTURE, not spot, so
    C(K) - P(K) = F - K with F = S + basis, giving basis = K - S + (C - P).
    Assuming K = round(S/50)*50 keeps the implied basis inside a plausible
    weekly-futures band (roughly +/-60 pts). Assuming K is pinned to the
    week's first ATM forces the implied "basis" to drift with spot, reaching
    +288 pts (1.6% of index) by week's end -- impossible for a weekly
    contract. So the CSV is per-bar DYNAMIC: exactly what the rolling
    endpoint returns.

    The 4.97-5.29% "flip rate" is not an artifact. Spot crosses a 25-pt
    half-step once per 20-60 bars, which IS 1.7-5% of 1-minute bars. The
    flips are the rolling process itself, and the model was trained on them.

    WHAT THE OLD FUNCTION ACTUALLY PRODUCED -- and why it needed fixing:
    it kept, from each offset block, only rows where
    actual_strike == that week's first ATM. For offset +5 that means only
    the bars where spot happened to sit ~250 pts below the pin. Every offset
    therefore contributed a DIFFERENT, disjoint slice of time, and the
    concat + drop_duplicates(subset=["datetime"]) that followed then kept an
    essentially arbitrary one of the 21 blocks per timestamp. The resulting
    call_close was a mosaic of different strikes at different times, so
    every feature computed from it (price_position_rank, rvol, atr, the rank
    windows) described a series that never existed -- while looking like
    real data rather than an obvious gap.

    Dhan's /charts/rollingoption already rolls the strike bar-by-bar and
    reports the rolled strike in `actual_strike`, so the ATM block IS the
    dynamic ATM series. No filter is needed -- or wanted.
    """
    end = date.today()
    start = end - timedelta(days=LOOKBACK_CALENDAR_DAYS)
    if log: log.info(f"live_warmstart[DYNAMIC]: fetching rolling ATM "
                     f"{start} to {end} ...")

    atm_call = _fetch_full_range(client_id, access_token, "ATM", "CALL", start, end, log)
    atm_put  = _fetch_full_range(client_id, access_token, "ATM", "PUT",  start, end, log)
    if atm_call.empty or atm_put.empty:
        if log: log.error("live_warmstart[DYNAMIC]: no ATM CALL/PUT data -- "
                          "cannot warm-start")
        return None

    # ── Self-check: is the endpoint really rolling per bar? ─────────────
    # Worth the two extra calls: this is the exact assumption the previous
    # warm-start got wrong, and getting it wrong is completely silent.
    try:
        nb = _fetch_full_range(client_id, access_token, "ATM+1", "CALL", start, end, log)
        if not nb.empty:
            m = pd.merge(
                atm_call[["datetime", "actual_strike", "spot"]],
                nb[["datetime", "actual_strike"]].rename(
                    columns={"actual_strike": "strike_p1"}),
                on="datetime", how="inner").dropna()
            if len(m):
                exp = (m["spot"] / 50.0).round() * 50.0
                d0 = (m["actual_strike"] - exp).abs()
                d1 = (m["strike_p1"] - (exp + 50)).abs()
                if log: log.info(
                    f"live_warmstart[DYNAMIC] roll-check on {len(m):,} bars: "
                    f"ATM == round(S/50)*50 on {(d0 < 1).mean()*100:.1f}%, "
                    f"ATM+1 == that+50 on {(d1 < 1).mean()*100:.1f}%  |  "
                    f"distinct ATM strikes in window = {m['actual_strike'].nunique()} "
                    f"(1 => endpoint is PINNED, many => rolling)")
    except Exception as e:
        if log: log.warning(f"live_warmstart[DYNAMIC] roll-check skipped: {e}")

    # Keep LAST_PIN_STRIKES populated/meaningful for any other consumer.
    # NOTE: under dynamic mode this is no longer a "pin" -- it is simply the
    # week's FIRST dynamic ATM, i.e. what the pin would have been.
    ac = atm_call.copy()
    ac["_week"] = pd.to_datetime(ac["datetime"].dt.date).dt.to_period("W-TUE")
    LAST_PIN_STRIKES.update(ac.groupby("_week").first()["actual_strike"].to_dict())

    merged = pd.merge(
        atm_call[["datetime", "close", "high", "low", "open", "spot", "iv", "oi", "volume"]]
            .rename(columns=lambda c: f"call_{c}" if c != "datetime" else c),
        atm_put[["datetime", "close", "high", "low", "open", "iv", "oi", "volume"]]
            .rename(columns=lambda c: f"put_{c}" if c != "datetime" else c),
        on="datetime", how="inner",
    )
    merged = merged.rename(columns={"call_spot": "spot_close"})
    merged = (merged.drop_duplicates(subset=["datetime"])
                    .sort_values("datetime").reset_index(drop=True))

    vix_df = fetch_historical_vix(start, end, log)
    if vix_df is not None:
        merged["_date"] = pd.to_datetime(merged["datetime"].dt.date)
        merged = merged.merge(vix_df, left_on="_date", right_on="date", how="left")
        merged["india_vix"] = merged["india_vix"].ffill().bfill()
        merged = merged.drop(columns=["_date", "date"])
        if log: log.info(f"live_warmstart[DYNAMIC]: real historical VIX merged in "
                         f"(range {merged['india_vix'].min():.2f}-{merged['india_vix'].max():.2f})")
    else:
        if log: log.warning("live_warmstart[DYNAMIC]: VIX fetch failed -- india_vix "
                            "will be constant-filled for the ENTIRE warm-started history")

    if log: log.info(f"live_warmstart[DYNAMIC]: reconstructed {len(merged):,} DYNAMIC-ATM "
                     f"rows ({merged['datetime'].min()} to {merged['datetime'].max()})")
    return merged


def fetch_pinned_call_history(client_id, access_token, log=None):
    """Return a separate weekly-pinned CALL close series for rule indicators.

    This does NOT change the dynamic-ATM history returned by
    fetch_dynamic_recent_data() and must not be fed into the model buffer.

    The pin for each W-TUE period is the first dynamic-ATM CALL strike in
    that period, matching the existing pinned-history reconstruction logic.
    Returns columns: datetime, close, actual_strike.
    """
    end = date.today()
    start = end - timedelta(days=LOOKBACK_CALENDAR_DAYS)

    if log:
        log.info(
            f"RULE-PIN warmstart: fetching CALL histories {start} to {end} "
            f"across {len(STRIKE_OFFSETS)} relative strikes"
        )

    all_calls = {}
    for off in STRIKE_OFFSETS:
        label = _offset_label(off)
        df = _fetch_full_range(
            client_id, access_token, label, "CALL", start, end, log
        )
        if not df.empty:
            all_calls[off] = df

    atm_call = all_calls.get(0)
    if atm_call is None or atm_call.empty:
        if log:
            log.error("RULE-PIN warmstart failed: no ATM CALL history available")
        return pd.DataFrame(columns=["datetime", "close", "actual_strike"])

    atm_call = atm_call.copy()
    atm_call["_week"] = (
        pd.to_datetime(atm_call["datetime"].dt.date).dt.to_period("W-TUE")
    )
    pin_strikes = atm_call.groupby("_week").first()["actual_strike"].to_dict()
    LAST_PIN_STRIKES.update(pin_strikes)

    pinned_frames = []
    for _off, source in all_calls.items():
        x = source.copy()
        x["_week"] = pd.to_datetime(x["datetime"].dt.date).dt.to_period("W-TUE")
        x["_pin_strike"] = x["_week"].map(pin_strikes)
        actual = pd.to_numeric(x["actual_strike"], errors="coerce")
        pinned = pd.to_numeric(x["_pin_strike"], errors="coerce")
        mask = actual.notna() & pinned.notna() & (actual == pinned)
        if mask.any():
            pinned_frames.append(
                x.loc[mask, ["datetime", "close", "actual_strike"]]
            )

    if not pinned_frames:
        if log:
            log.error("RULE-PIN warmstart failed: no pinned CALL rows found")
        return pd.DataFrame(columns=["datetime", "close", "actual_strike"])

    out = pd.concat(pinned_frames, ignore_index=True)
    out["datetime"] = pd.to_datetime(out["datetime"])
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["datetime", "close"])
    out = out.loc[out["close"] > 0]
    out = (
        out.drop_duplicates(subset=["datetime"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    if log:
        log.info(
            f"RULE-PIN warmstart: reconstructed {len(out):,} CALL rows "
            f"({out['datetime'].min()} to {out['datetime'].max()})"
        )
    return out


def _adx_scalar(arr, p=14):
    """Local copy of v12_l5_inference.py's _adx() formula -- deliberately NOT
    imported from there, so this file keeps working standalone in an l3-only
    folder (where v12_l5_inference.py isn't installed at all, per the
    two-separate-folders deployment). Keep in sync manually if that formula
    ever changes."""
    if len(arr) < p + 1:
        return 20.0
    a = arr.astype(float)
    pdm = np.maximum(np.diff(a), 0); ndm = np.maximum(-np.diff(a), 0)
    pdm = np.where(pdm > ndm, pdm, 0.0); ndm = np.where(ndm > pdm, ndm, 0.0)
    atr_ = np.abs(np.diff(a))
    def _sm(x):
        w = x[-(p+1):]; alpha = 2.0/(p+1); v = float(w[0])
        for xx in w[1:]:
            v = alpha*float(xx) + (1-alpha)*v
        return v
    pdi = 100 * _sm(pdm) / (_sm(atr_) + 1e-9)
    ndi = 100 * _sm(ndm) / (_sm(atr_) + 1e-9)
    return float(100 * abs(pdi - ndi) / (pdi + ndi + 1e-9))


def compute_adx1m_series(buf, window=30):
    """Per-bar ADX_1m for the whole buffer, using the SAME simplified formula
    both v12_inference.py (l3) and v12_l5_inference.py's _build_features()
    use -- needed so either model's rvol_hist.update_extra() (adx_1m_rank /
    price_position_rank self-rank history) can be backfilled bar-by-bar,
    exactly like rvol/vix already are via compute_rvol_series(). _adx_scalar
    only ever looks at the trailing ~15 diffs, so a small sliding window is
    equivalent to feeding the full growing history and is far cheaper
    (O(N*window), not O(N^2))."""
    close = np.array(buf["call_close"], dtype=float)
    n = len(close)
    out = np.full(n, 20.0)
    for i in range(n):
        lo = max(0, i - window + 1)
        out[i] = _adx_scalar(close[lo:i+1], 14)
    return out


def replay_model_state(buf, model, rvol_arr, vix_arr, log=None, label="model",
                       adx1m_arr=None, price_arr=None, stochk_arr=None):
    """
    Replays a warm-started buffer into ONE model's htf/pivot/rvol_hist.
    Factored out of warmstart_live_bot() so multiple models (e.g. v12_l3's
    live model AND a separate v12_l5 model running in parallel) can each
    get their OWN independent warm rolling state from a SINGLE Dhan/VIX
    fetch, instead of re-fetching once per model.

    adx1m_arr/price_arr: meaningful for ANY model whose rvol_hist exposes
    update_extra() -- as of the l3 feature-pipeline fix pass, BOTH v12_l3's
    and v12_l5's RvolHistoryV12 track adx_1m/price self-rank history (see
    v12_inference.py / v12_l5_inference.py). The hasattr() guard below means
    this stays a safe no-op for any OTHER model that doesn't expose these
    methods, so this file needs no per-model branching.
    """
    n_replayed = 0
    n_rvol_replayed = 0
    n_extra_replayed = 0
    for i in range(len(buf["call_close"])):
        try:
            bc = buf["call_close"][i]; bh = buf["call_high"][i]
            bl = buf["call_low"][i]; bo = buf["call_open"][i]
            bv = buf["call_volume"][i]; sp = buf["spot_close"][i]
            bt = buf["datetime"][i]
            if not (isinstance(bc, (int, float)) and bc > 0):
                continue
            model.htf.update(bt, o=bo, h=bh, l=bl, c=bc, vol=bv, sp=sp)
            model.pivot.update(bt.date(), spot_high=sp, spot_low=sp, spot_close=sp)
            n_replayed += 1
        except Exception as e:
            if log: log.warning(f"live_warmstart: [{label}] htf/pivot replay skipped bar {i}: {e}")
        try:
            if hasattr(model, "rvol_hist") and model.rvol_hist is not None:
                model.rvol_hist.update(float(rvol_arr[i]), float(vix_arr[i]))
                n_rvol_replayed += 1
        except Exception as e:
            if log: log.warning(f"live_warmstart: [{label}] rvol_hist replay skipped bar {i}: {e}")
        try:
            if (hasattr(model, "rvol_hist") and model.rvol_hist is not None
                    and hasattr(model.rvol_hist, "update_extra")
                    and adx1m_arr is not None and price_arr is not None):
                model.rvol_hist.update_extra(float(adx1m_arr[i]), float(price_arr[i]))
                n_extra_replayed += 1
        except Exception as e:
            if log: log.warning(f"live_warmstart: [{label}] adx1m/price rank replay skipped bar {i}: {e}")
        try:
            if (hasattr(model, "rvol_hist") and model.rvol_hist is not None
                    and hasattr(model.rvol_hist, "stoch_update") and stochk_arr is not None):
                model.rvol_hist.stoch_update(float(stochk_arr[i]))
        except Exception as e:
            if log: log.warning(f"live_warmstart: [{label}] stoch replay skipped bar {i}: {e}")
    if log: log.info(f"live_warmstart: [{label}] replayed {n_replayed} bars into htf/pivot, "
                     f"{n_rvol_replayed} bars into rvol_hist, "
                     f"{n_extra_replayed} bars into adx1m/price rank history")
    return n_replayed, n_rvol_replayed


def compute_rvol_series(buf):
    """Matches training's exact rvol formula (nifty_v12_l5_train.py L176-179):
        tv = call_volume.clip(lower=0) + put_volume.clip(lower=0)
        rvol = (tv / (tv.rolling(20).mean() + 1e-9)).clip(upper=10)
    Returns (rvol_arr, vix_arr), same length as the buffer."""
    cv = np.clip(np.array(buf["call_volume"], dtype=float), 0, None)
    pv = np.clip(np.array(buf["put_volume"], dtype=float), 0, None)
    tv = cv + pv
    tv20 = pd.Series(tv).rolling(20, min_periods=1).mean().to_numpy()
    rvol_arr = np.clip(tv / (tv20 + 1e-9), None, 10.0)
    vix_arr = np.array(buf["india_vix"], dtype=float)
    return rvol_arr, vix_arr


def compute_stochk_series(buf, k=14):
    """Per-bar raw %K for the whole buffer, feeding either model's
    rvol_hist.stoch_update() so its double-smoothed stoch_k/stoch_d (SMA-3 of
    raw %K, then SMA-3 of that) has a real (if short, 3-bar) history from bar 1
    instead of stoch_d==stoch_k on the first few live bars."""
    close = np.array(buf["call_close"], dtype=float)
    high  = np.array(buf["call_high"],  dtype=float)
    low   = np.array(buf["call_low"],   dtype=float)
    hh = pd.Series(high).rolling(k, min_periods=1).max().to_numpy()
    ll = pd.Series(low).rolling(k, min_periods=1).min().to_numpy()
    return (close - ll) / (hh - ll + 1e-9) * 100


def warmstart_live_bot(buffer_dict_factory, model, dhan_client_id, dhan_access_token,
                       india_vix_default=16.0, log=None, extra_models=None):
    """
    Full warm-start: fetches pinned recent data, builds an extended buffer,
    replays it into the buffer AND into model.htf/model.pivot/model.rvol_hist,
    so both the raw buffer and the model's own internal state are warm
    before the first live tick arrives.

    buffer_dict_factory: a zero-arg callable that returns a fresh buffer
    dict (pass build_extended_buffer from buffer_warmstart, or the bot's
    own equivalent once BUFFER_SIZE is fixed to match).

    extra_models: optional list of (label, model) tuples -- additional
    models to warm-start from the SAME fetched data in the SAME call, so
    e.g. a v12_l5 model running in parallel with the primary v12_l3 model
    doesn't need its own separate Dhan fetch. Each gets its own
    independent htf/pivot/rvol_hist replay.

    Returns the warm-started buffer dict, or None if warm-start failed
    (caller should fall back to an empty buffer + live-only accumulation
    rather than crash the bot).
    """
    if WARMSTART_MODE == "dynamic":
        data = fetch_dynamic_recent_data(dhan_client_id, dhan_access_token, log)
    else:
        if log: log.warning("live_warmstart: WARMSTART_MODE='pinned' -- using the "
                            "pin-filtered reconstruction, which does NOT match the "
                            "dynamic-ATM series the model was trained on")
        data = fetch_pinned_recent_data(dhan_client_id, dhan_access_token, log)
    if data is None or data.empty:
        if log: log.error("live_warmstart: FAILED -- bot will start with an empty buffer "
                          "and accumulate live-only, same as before this fix")
        return None

    buf = buffer_dict_factory()
    warm_start_buffer(buf, data, india_vix_default)

    # Precompute rvol for every buffered bar, EXACTLY matching training's
    # formula (nifty_v12_l5_train.py line 176-179).
    rvol_arr, vix_arr = compute_rvol_series(buf)

    # CORRECTION (was wrong above): adx_1m_rank/price_position_rank/adx_5m_
    # /15m_/30m_rank are NOT hardcoded to 0.5 in either model's real feature
    # pipeline -- confirmed for l5 directly against nifty_v12_l5_train.py,
    # and confirmed for l3 by cross-checking l3's own trained feature-name
    # list (extracted_features.json), which shows these as real, distinct,
    # non-constant trained features. Both v12_l5_inference.py and
    # v12_inference.py (l3) were fixed to compute them for real (rvol_hist
    # now tracks adx1m/price too; HTFStateV12 tracks a per-timeframe adx-rank
    # history). Precompute adx1m/price arrays so either model's rvol_hist can
    # be backfilled the same way rvol/vix already are -- otherwise a freshly
    # warm-started bot would still start the FIRST 100 live bars with a cold
    # adx1m/price rank (0.5) despite everything else being warm. The
    # 5m/15m/30m adx ranks need no separate array here: they backfill for
    # free through the SAME model.htf.update() call below, since that's
    # where both inference files' HTFStateV12 now hooks its rank history.
    # (Any OTHER model whose rvol_hist has no update_extra() gets a pure
    # no-op here via the hasattr() guard in replay_model_state -- unaffected.)
    adx1m_arr = compute_adx1m_series(buf)
    price_arr = np.array(buf["call_close"], dtype=float)

    # stoch_k/stoch_d: v12_l5_inference.py's _stoch() used to return (K, K) --
    # identical every bar. Fixed to a real double-smoothed stochastic (SMA-3 of
    # raw %K, then SMA-3 of that), which only needs a 3-bar history to be
    # accurate -- backfilled here so it's correct from live bar 1, not just
    # after 3 live bars.
    stochk_arr = compute_stochk_series(buf)

    # Replay model.htf / model.pivot / model.rvol_hist for every historical
    # bar, in order -- without this, the buffer is warm but the model's OWN
    # internal rolling state (separate from the buffer dict) stays cold.
    if log: log.info("live_warmstart: replaying model.htf/model.pivot/model.rvol_hist state...")
    n_replayed, n_rvol_replayed = replay_model_state(
        buf, model, rvol_arr, vix_arr, log, label="primary",
        adx1m_arr=adx1m_arr, price_arr=price_arr, stochk_arr=stochk_arr,
    )

    # Any additional models (e.g. a v12_l5 model running in parallel) get
    # their OWN independent replay from the SAME fetched data -- no second
    # Dhan/VIX fetch needed.
    for extra_label, extra_model in (extra_models or []):
        replay_model_state(
            buf, extra_model, rvol_arr, vix_arr, log, label=extra_label,
            adx1m_arr=adx1m_arr, price_arr=price_arr, stochk_arr=stochk_arr,
        )

    if log: log.info(f"live_warmstart: replayed {n_replayed} bars into model.htf/model.pivot, "
                     f"{n_rvol_replayed} bars into model.rvol_hist")
    if log: log.info(f"live_warmstart: COMPLETE -- buffer at {len(buf['call_close'])}/{BUFFER_SIZE}, "
                     f"model state warm (htf, pivot, AND rvol_hist) for "
                     f"{1 + len(extra_models or [])} model(s). "
                     f"Bot(s) can predict meaningfully from bar 1.")
    return buf