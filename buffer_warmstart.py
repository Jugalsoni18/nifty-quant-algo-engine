"""
buffer_warmstart.py
=======================
Fixes the buffer-gap bug quantified in nifty_v12_l5_buffer_gap_diagnostic.py:
  price_position_rank  correlation 0.54  (badly broken)
  vix_ma20              correlation 0.77  (meaningfully degraded)
  adx_1m_rank            correlation 0.85  (moderately degraded)
  rvol_pct_rank          correlation 0.99  (fine, no fix needed)

TWO SEPARATE PROBLEMS, THIS FILE ONLY SOLVES ONE:

  1. BUFFER SIZE + WINDOW MATH (fixed here). The live bot's buffer was
     only 250 bars, and its own rolling_rank() helper used a hardcoded
     window=100 -- neither even ATTEMPTED to match training's 7,500-bar
     (20-trading-day) window. This file extends the buffer and fixes
     the window math to match training exactly.

  2. DATA SOURCING (NOT solved here -- see below). Pre-filling the
     buffer with 20 real trading days of history requires the ROLLING
     ATM call/put series -- a different actual contract on different
     days, as spot moves and weekly expiries roll over. Most of those
     contracts are already expired by "today." Retail broker APIs
     (Dhan included) typically only serve currently-listed instruments,
     not archived data for expired contracts. NIFTY_1min_rich_iv_filled.csv
     was built by SOME pipeline that already solved this -- if you have
     ongoing access to that pipeline/vendor, it's the natural source for
     warm-start data too (see load_historical_bars() below, which
     accepts a DataFrame in that CSV's exact schema). If you don't have
     that access, this half of the problem needs its own investigation
     before the warm-start can run end-to-end -- this file will not
     silently paper over that gap.

USAGE:
    from buffer_warmstart import BUFFER_SIZE, warm_start_buffer, rolling_rank

    # at bot startup, BEFORE the feed connects:
    historical_df = <however you obtain ~21 trading days of history,
                      same columns as NIFTY_1min_rich_iv_filled.csv>
    warm_start_buffer(buffer, historical_df)   # buffer = the live bot's
                                                 # existing buffer dict of
                                                 # deques

    # then during live operation, use the corrected rolling_rank() below
    # in place of whatever version currently computes rvol_pct_rank/
    # adx_1m_rank/price_position_rank/vix_ma20 in the live bot.
"""

import numpy as np
import pandas as pd
from collections import deque

# ─────────────────────────────────────────────────────────────────────────
# FIX 1: buffer size, matching training's actual 20-trading-day window
# (20 * 375 = 7,500 one-minute bars per trading day), with a small
# safety margin. Was 250 -- 30x too short for the affected features.
# ─────────────────────────────────────────────────────────────────────────

FULL_WINDOW = 20 * 375   # matches nifty_v12_l5_train.py's _rvol_w exactly
BUFFER_SIZE = FULL_WINDOW + 500   # small safety margin above the window itself

BUFFER_KEYS = [
    "datetime", "spot_close",
    "call_close", "call_high", "call_low", "call_open",
    "put_close", "put_high", "put_low", "put_open",
    "call_volume", "put_volume",
    "call_oi", "put_oi", "call_iv", "put_iv", "india_vix",
    "otm_call_high", "otm_call_low", "otm_call_close",
    "otm_put_high",  "otm_put_low",  "otm_put_close",
]

def build_extended_buffer():
    """Drop-in replacement for the live bot's `buffer = {k: deque(...) ...}`
    line -- same keys, just BUFFER_SIZE now large enough for the long-
    window features instead of the previous 250."""
    return {k: deque(maxlen=BUFFER_SIZE) for k in BUFFER_KEYS}


# ─────────────────────────────────────────────────────────────────────────
# FIX 2: rolling_rank() with the CORRECT window, matching training exactly.
# The live bot's own version of this used a hardcoded window=100 -- not
# even attempting to match the 7,500-bar window training used. This
# replaces it.
# ─────────────────────────────────────────────────────────────────────────

def rolling_rank(arr, window=FULL_WINDOW, min_periods=100):
    """Rank the last valid observation against the trailing training window."""
    values = np.asarray(list(arr) if not isinstance(arr, np.ndarray) else arr, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if len(values) < min_periods:
        return 0.5
    w = values[-int(window):] if len(values) > int(window) else values
    return float(np.mean(w <= w[-1]))


def compute_vix_ma20(vix_arr, window=FULL_WINDOW):
    """Mean of the available trailing VIX window (full match once warmed)."""
    values = np.asarray(list(vix_arr) if not isinstance(vix_arr, np.ndarray) else vix_arr, dtype=float).reshape(-1)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) == 0:
        return 16.0
    w = values[-int(window):] if len(values) > int(window) else values
    return float(np.mean(w))


def compute_vix_above_rolling_median(vix_arr, current_vix, window=FULL_WINDOW):
    """Return 1 when current VIX is above the trailing available median."""
    values = np.asarray(list(vix_arr) if not isinstance(vix_arr, np.ndarray) else vix_arr, dtype=float).reshape(-1)
    values = values[np.isfinite(values) & (values > 0)]
    try:
        current = float(current_vix)
    except (TypeError, ValueError):
        return 0
    if len(values) == 0 or not np.isfinite(current) or current <= 0:
        return 0
    w = values[-int(window):] if len(values) > int(window) else values
    return int(current > float(np.median(w)))


# ─────────────────────────────────────────────────────────────────────────
# WARM-START LOADER -- accepts historical data in the SAME schema as
# NIFTY_1min_rich_iv_filled.csv (or any source that can supply it),
# populates the extended buffer BEFORE live ticks start arriving.
# ─────────────────────────────────────────────────────────────────────────

def load_historical_bars(historical_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize supported historical schemas into the buffer schema.

    This is a format/quality gate, not a data fetch. It requires real CALL,
    PUT and spot closes, removes duplicate/invalid timestamps and orders rows
    chronologically. Missing optional fields are handled with explicit
    defaults by warm_start_buffer().
    """
    if not isinstance(historical_df, pd.DataFrame):
        raise TypeError("load_historical_bars expects a pandas DataFrame")
    if historical_df.empty:
        raise ValueError("load_historical_bars received an empty DataFrame")

    df = historical_df.copy()
    rename_map = {
        "nifty_WEEK_ATM_CALL_1m__close": "call_close",
        "nifty_WEEK_ATM_CALL_1m__high": "call_high",
        "nifty_WEEK_ATM_CALL_1m__low": "call_low",
        "nifty_WEEK_ATM_CALL_1m__open": "call_open",
        "nifty_WEEK_ATM_CALL_1m__volume": "call_volume",
        "nifty_WEEK_ATM_CALL_1m__oi": "call_oi",
        "nifty_WEEK_ATM_CALL_1m__iv": "call_iv",
        "nifty_WEEK_ATM_PUT_1m__close": "put_close",
        "nifty_WEEK_ATM_PUT_1m__high": "put_high",
        "nifty_WEEK_ATM_PUT_1m__low": "put_low",
        "nifty_WEEK_ATM_PUT_1m__open": "put_open",
        "nifty_WEEK_ATM_PUT_1m__volume": "put_volume",
        "nifty_WEEK_ATM_PUT_1m__oi": "put_oi",
        "nifty_WEEK_ATM_PUT_1m__iv": "put_iv",
        "nifty_WEEK_ATM_CALL_1m__spot": "spot_close",
        "nifty_WEEK_ATM_PUT_1m__spot": "spot_close",
        "timestamp": "datetime",
    }
    for source, target in rename_map.items():
        if source in df.columns and target not in df.columns:
            df = df.rename(columns={source: target})

    required = ["datetime", "call_close", "put_close", "spot_close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"load_historical_bars: missing required columns {missing}; "
            "the source must provide real CALL, PUT and spot prices, not just index history."
        )

    # Handle the common case of a uniformly timezone-aware source, while
    # preserving already-naive Dhan/IST timestamps as local clock time.
    dt = pd.to_datetime(df["datetime"], errors="coerce")
    if isinstance(getattr(dt, "dtype", None), pd.DatetimeTZDtype):
        dt = dt.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df["datetime"] = dt
    df = df.dropna(subset=["datetime"]).copy()

    numeric_cols = [
        "call_close", "call_high", "call_low", "call_open", "call_volume",
        "call_oi", "call_iv", "put_close", "put_high", "put_low", "put_open",
        "put_volume", "put_oi", "put_iv", "spot_close", "india_vix",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # These are indispensable model inputs. Invalid rows must not inject zero,
    # NaN, or infinities into rolling features.
    for col in ("call_close", "put_close", "spot_close"):
        df = df[np.isfinite(df[col]) & (df[col] > 0)]

    df = (df.sort_values("datetime")
            .drop_duplicates(subset=["datetime"], keep="last")
            .reset_index(drop=True))
    if df.empty:
        raise ValueError(
            "load_historical_bars: no valid rows remain after timestamp/price validation"
        )
    return df


def _finite_float(value, default: float) -> float:
    """Return a finite float, using default for missing/invalid values."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def warm_start_buffer(
    buffer: dict,
    historical_df: pd.DataFrame,
    india_vix_default: float = 16.0,
) -> int:
    """Replace the target buffer with validated historical rows in time order.

    Returns the number of rows loaded. The target mapping must contain the
    standard BUFFER_KEYS, so no feature silently loses alignment.
    """
    df = load_historical_bars(historical_df)
    missing_keys = [key for key in BUFFER_KEYS if key not in buffer]
    if missing_keys:
        raise ValueError(
            f"warm_start_buffer: target buffer is missing keys {missing_keys}; "
            "use build_extended_buffer()"
        )

    df = df.tail(BUFFER_SIZE).reset_index(drop=True)
    # Warm-start is intended to be done before live feed subscription. Clear
    # all series first so stale rows cannot be mixed with older/newer history.
    for key in BUFFER_KEYS:
        clear = getattr(buffer[key], "clear", None)
        if not callable(clear):
            raise TypeError(
                f"warm_start_buffer: buffer[{key!r}] must be a deque/list with clear()"
            )
        clear()

    n_loaded = 0
    for row in df.to_dict("records"):
        call_close = _finite_float(row.get("call_close"), 0.0)
        put_close = _finite_float(row.get("put_close"), 0.0)
        spot_close = _finite_float(row.get("spot_close"), 0.0)
        # Should already be guaranteed by load_historical_bars, but retain a
        # defensive guard at the final write boundary.
        if min(call_close, put_close, spot_close) <= 0:
            continue

        call_high = _finite_float(row.get("call_high"), call_close)
        call_low = _finite_float(row.get("call_low"), call_close)
        call_open = _finite_float(row.get("call_open"), call_close)
        put_high = _finite_float(row.get("put_high"), put_close)
        put_low = _finite_float(row.get("put_low"), put_close)
        put_open = _finite_float(row.get("put_open"), put_close)

        values = {
            "datetime": row["datetime"],
            "spot_close": spot_close,
            "call_close": call_close,
            "call_high": call_high,
            "call_low": call_low,
            "call_open": call_open,
            "put_close": put_close,
            "put_high": put_high,
            "put_low": put_low,
            "put_open": put_open,
            "call_volume": max(0.0, _finite_float(row.get("call_volume"), 0.0)),
            "put_volume": max(0.0, _finite_float(row.get("put_volume"), 0.0)),
            "call_oi": max(0.0, _finite_float(row.get("call_oi"), 0.0)),
            "put_oi": max(0.0, _finite_float(row.get("put_oi"), 0.0)),
            "call_iv": _finite_float(row.get("call_iv"), 18.0),
            "put_iv": _finite_float(row.get("put_iv"), 19.0),
            "india_vix": _finite_float(row.get("india_vix"), india_vix_default),
            # Historical source generally cannot provide the current execution
            # OTM strike at each past bar. Keep the existing ATM proxy semantics.
            "otm_call_high": _finite_float(row.get("otm_call_high"), call_high),
            "otm_call_low": _finite_float(row.get("otm_call_low"), call_low),
            "otm_call_close": _finite_float(row.get("otm_call_close"), call_close),
            "otm_put_high": _finite_float(row.get("otm_put_high"), put_high),
            "otm_put_low": _finite_float(row.get("otm_put_low"), put_low),
            "otm_put_close": _finite_float(row.get("otm_put_close"), put_close),
        }
        for key in BUFFER_KEYS:
            buffer[key].append(values[key])
        n_loaded += 1

    print(
        f"warm_start_buffer: loaded {n_loaded:,} valid historical bars "
        f"(buffer now {len(buffer['call_close']):,}/{BUFFER_SIZE})"
    )
    if n_loaded < FULL_WINDOW:
        print(
            f"  WARNING: loaded {n_loaded:,} bars; long-window features need "
            f"{FULL_WINDOW:,} for a full training-window match."
        )
    return n_loaded


if __name__ == "__main__":
    # SMOKE TEST -- this file has no historical data source of its own
    # (that's the open question from the last message), so this proves
    # the module WORKS using synthetic data, not that it's wired to a
    # real feed yet. Run this directly to confirm the module imports and
    # behaves correctly before it gets integrated into an actual bot.
    print("=" * 70)
    print("  buffer_warmstart.py SMOKE TEST -- synthetic data, not real feed")
    print("=" * 70)

    print(f"\nFULL_WINDOW = {FULL_WINDOW:,} bars (20 trading days)")
    print(f"BUFFER_SIZE = {BUFFER_SIZE:,} bars")

    print("\nBuilding extended buffer...")
    buf = build_extended_buffer()
    print(f"  Created {len(buf)} deques, each maxlen={BUFFER_SIZE:,}")

    print("\nGenerating synthetic historical data (random walk, NOT real "
          "market data) to test the loader end-to-end...")
    n_synthetic = BUFFER_SIZE + 500  # MORE than BUFFER_SIZE, to actually
                                       # test the deque's truncation --
                                       # my first version of this test
                                       # only generated FULL_WINDOW+200
                                       # bars, which is LESS than
                                       # BUFFER_SIZE, so it never actually
                                       # exercised the capping behavior
                                       # the assertion below checks for
    rng = np.random.RandomState(42)
    dt_index = pd.date_range("2025-01-01 09:15", periods=n_synthetic, freq="1min")
    spot_walk = 23000 + np.cumsum(rng.normal(0, 2, n_synthetic))
    call_walk = np.clip(100 + np.cumsum(rng.normal(0, 1, n_synthetic)), 10, None)
    put_walk  = np.clip(100 + np.cumsum(rng.normal(0, 1, n_synthetic)), 10, None)

    synthetic_df = pd.DataFrame({
        "datetime": dt_index,
        "call_close": call_walk, "call_high": call_walk * 1.01, "call_low": call_walk * 0.99,
        "call_open": call_walk, "call_volume": rng.randint(1000, 50000, n_synthetic),
        "call_oi": rng.randint(1e6, 1e7, n_synthetic), "call_iv": rng.uniform(12, 25, n_synthetic),
        "put_close": put_walk, "put_high": put_walk * 1.01, "put_low": put_walk * 0.99,
        "put_open": put_walk, "put_volume": rng.randint(1000, 50000, n_synthetic),
        "put_oi": rng.randint(1e6, 1e7, n_synthetic), "put_iv": rng.uniform(12, 25, n_synthetic),
        "spot_close": spot_walk, "india_vix": rng.uniform(10, 20, n_synthetic),
    })
    print(f"  Generated {n_synthetic:,} synthetic bars "
          f"({n_synthetic - FULL_WINDOW:,} more than one full window)")

    print("\nRunning warm_start_buffer()...")
    warm_start_buffer(buf, synthetic_df)

    print("\nVerifying buffer was populated correctly...")
    assert len(buf["call_close"]) == BUFFER_SIZE, \
        f"Expected buffer capped at {BUFFER_SIZE}, got {len(buf['call_close'])}"
    print(f"  Buffer correctly capped at maxlen={BUFFER_SIZE:,} "
          f"(oldest {n_synthetic - BUFFER_SIZE:,} synthetic bars correctly dropped)")

    print("\nTesting rolling_rank() / compute_vix_ma20() on the warm-started buffer...")
    rvol_test_arr = np.array(buf["call_volume"], dtype=float)
    rank_val = rolling_rank(rvol_test_arr)
    vix_arr = np.array(buf["india_vix"], dtype=float)
    vix_ma = compute_vix_ma20(vix_arr)
    vix_med_flag = compute_vix_above_rolling_median(vix_arr, vix_arr[-1])
    print(f"  rolling_rank(call_volume) = {rank_val:.4f}  (should be in [0,1])")
    print(f"  compute_vix_ma20()        = {vix_ma:.2f}  (should be ~10-20, matching synthetic range)")
    print(f"  compute_vix_above_rolling_median() = {vix_med_flag}  (should be 0 or 1)")

    assert 0.0 <= rank_val <= 1.0, "rolling_rank output out of expected [0,1] range"
    assert 10.0 <= vix_ma <= 20.0, "vix_ma20 output outside synthetic data's actual range"

    print("\n" + "=" * 70)
    print("  SMOKE TEST PASSED -- module behaves correctly on synthetic data.")
    print("  This does NOT confirm real historical data is available or that")
    print("  this is wired into a live bot -- both are still open, separate")
    print("  from what this test covers.")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────
# REAL DHAN FETCH -- rewritten to match the user's own proven, already-
# working download_otm_full_chain.py exactly, rather than the dhanhq
# Python wrapper's expired_options_data() method (which failed -- wrong
# response-shape assumption). Confirmed details from that working script:
#   - Raw REST POST to https://api.dhan.co/v2/charts/rollingoption,
#     camelCase JSON fields, NOT the dhanhq wrapper's snake_case method.
#   - expiryFlag="WEEK", expiryCode=1 (not 0) -- confirmed working values.
#   - Response shape: {"data": {"ce": {"timestamp": [...], "open": [...],
#     ...}}} -- a dict of PARALLEL ARRAYS keyed by "ce"/"pe", not a flat
#     dict-of-scalars or list-of-row-dicts. This nesting is exactly what
#     broke the previous version's naive pd.DataFrame(call_resp["data"]).
#   - timestamp comes back as epoch SECONDS, needs UTC->IST conversion.
#   - toDate is NON-INCLUSIVE -- add 1 day to actually include the end date.
# ─────────────────────────────────────────────────────────────────────────

DHAN_BASE_URL = "https://api.dhan.co/v2"

def _dhan_headers(client_id: str, access_token: str) -> dict:
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "access-token": access_token,
        "client-id": client_id,
    }

def _fetch_rolling_atm(client_id: str, access_token: str, drv_type: str,
                        from_date, to_date, expiry_code: int = 1):
    """One CALL or PUT ATM fetch, exactly matching the confirmed-working
    request/response shape from download_otm_full_chain.py's fetch_rolling()."""
    import requests
    api_to = to_date + pd.Timedelta(days=1)  # toDate is non-inclusive
    payload = {
        "exchangeSegment": "NSE_FNO", "interval": "1", "securityId": "13",
        "instrument": "OPTIDX", "expiryFlag": "WEEK", "expiryCode": expiry_code,
        "strike": "ATM", "drvOptionType": drv_type,
        "requiredData": ["open", "high", "low", "close", "iv", "volume", "strike", "oi", "spot"],
        "fromDate": from_date.strftime("%Y-%m-%d"), "toDate": api_to.strftime("%Y-%m-%d"),
    }
    r = requests.post(f"{DHAN_BASE_URL}/charts/rollingoption", json=payload,
                       headers=_dhan_headers(client_id, access_token), timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Dhan API error {r.status_code} for {drv_type}: {r.text[:300]}")
    js = r.json()
    key = "ce" if drv_type == "CALL" else "pe"
    node = js.get("data", {}).get(key)
    if not node or not node.get("timestamp"):
        return pd.DataFrame()
    n = len(node["timestamp"])
    df = pd.DataFrame({
        "timestamp": node.get("timestamp", [None] * n),
        "open": node.get("open", [None] * n), "high": node.get("high", [None] * n),
        "low": node.get("low", [None] * n), "close": node.get("close", [None] * n),
        "volume": node.get("volume", [0] * n), "iv": node.get("iv", [None] * n),
        "oi": node.get("oi", [0] * n), "spot": node.get("spot", [None] * n),
    })
    df["datetime"] = (pd.to_datetime(df["timestamp"], unit="s", utc=True)
                       .dt.tz_convert("Asia/Kolkata").dt.tz_localize(None))
    return df


def fetch_dhan_warmstart_data(
    client_id: str,
    access_token: str,
    lookback_calendar_days: int = 30,
    expiry_code: int = 1,
    chunk_days: int = 7,
) -> pd.DataFrame:
    """Fetch rolling-ATM CALL/PUT data in small date chunks to reduce 504s.

    The Dhan endpoint's ``toDate`` is treated as non-inclusive by the proven
    request wrapper, so each helper call receives the final inclusive date.
    This function still depends on Dhan providing the requested old history.
    """
    if lookback_calendar_days < 1:
        raise ValueError("lookback_calendar_days must be >= 1")
    if chunk_days < 1:
        raise ValueError("chunk_days must be >= 1")

    from datetime import timedelta
    to_date = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize()
    from_date = to_date - pd.Timedelta(days=lookback_calendar_days)
    print(
        f"fetch_dhan_warmstart_data: requesting {from_date.date()} to "
        f"{to_date.date()} in <= {chunk_days}-day chunks "
        f"(expiryFlag='WEEK', expiryCode={expiry_code})..."
    )

    call_parts, put_parts = [], []
    chunk_start = from_date
    while chunk_start <= to_date:
        chunk_end = min(to_date, chunk_start + pd.Timedelta(days=chunk_days - 1))
        call_part = _fetch_rolling_atm(
            client_id, access_token, "CALL", chunk_start, chunk_end, expiry_code
        )
        put_part = _fetch_rolling_atm(
            client_id, access_token, "PUT", chunk_start, chunk_end, expiry_code
        )
        if not call_part.empty:
            call_parts.append(call_part)
        if not put_part.empty:
            put_parts.append(put_part)
        print(
            f"  {chunk_start.date()}..{chunk_end.date()}: "
            f"CALL={len(call_part)} PUT={len(put_part)}"
        )
        chunk_start = chunk_end + pd.Timedelta(days=1)

    call_df = pd.concat(call_parts, ignore_index=True) if call_parts else pd.DataFrame()
    put_df = pd.concat(put_parts, ignore_index=True) if put_parts else pd.DataFrame()
    if call_df.empty or put_df.empty:
        raise RuntimeError(
            "fetch_dhan_warmstart_data: got no usable rows for CALL or PUT; "
            "check Dhan permissions, expiryCode, and historical availability."
        )

    call_df = (call_df.drop_duplicates("datetime", keep="last")
                      .set_index("datetime").sort_index())
    put_df = (put_df.drop_duplicates("datetime", keep="last")
                     .set_index("datetime").sort_index())
    merged = pd.DataFrame({
        "call_close": call_df["close"], "call_high": call_df["high"],
        "call_low": call_df["low"], "call_open": call_df["open"],
        "call_volume": call_df["volume"], "call_oi": call_df["oi"],
        "call_iv": call_df["iv"], "spot_close": call_df["spot"],
        "put_close": put_df["close"], "put_high": put_df["high"],
        "put_low": put_df["low"], "put_open": put_df["open"],
        "put_volume": put_df["volume"], "put_oi": put_df["oi"],
        "put_iv": put_df["iv"],
    }).reset_index()
    merged = merged.sort_values("datetime").drop_duplicates("datetime", keep="last")
    return merged.reset_index(drop=True)

