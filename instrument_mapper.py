"""
instrument_mapper.py — Nifty50 Option Instrument Resolver
──────────────────────────────────────────────────────────
- InstrumentMapper.load() — no-arg, auto-detects nearest NIFTY expiry,
  caches metadata for today. Dhan feed instruments stay in livetrading.py.
- Dynamic OTM picker — fetches live chain, scores by OI/volume/distance/IV,
  filters by live capital from Zerodha margins API.


CHANGES APPLIED:
  - Removed dead/duplicate parser code
  - _parse_nse_symbol() supports both:
      1) Monthly text format   : NIFTY26MAY23700PE
      2) Dated numeric format  : NIFTY2651923500PE
  - Dhan scrip cache freshness moved to sidecar JSON metadata file
    instead of DataFrame.attrs
  - pick_best_otm_strike() now resolves full Dhan contract, not just ID
  - pick_best_otm_strike() catches RuntimeError too
  - Self-test updated for dict return + monthly symbol case


Dhan detailed CSV fix:
  - Detailed CSV uses DISPLAY_NAME as old SEM_CUSTOM_SYMBOL, not SEM_TRADING_SYMBOL
  - Detailed CSV removes SEM_TRADING_SYMBOL entirely
  - Contract resolution now matches on:
      UNDERLYING_SYMBOL / SYMBOL_NAME + INSTRUMENT + OPTION_TYPE + STRIKE_PRICE + SM_EXPIRY_DATE
  - DISPLAY_NAME is retained only for display/debugging
  - Added debug logging for live schema / filter counts

BUG FIX (get_otm_strikes):
  - CE filter changed from  strike > spot  →  strike > atm
  - PE filter changed from  strike < spot  →  strike < atm
  - Prevents ATM strike (distance_pts=0) from entering the OTM candidate
    list when spot drifts between two strike levels (e.g. spot=23472,
    ATM=23450 → 23450PE was passing  strike < spot  and winning the
    score ranking with +25pts for dist<=50, then being subscribed to the
    same Dhan ID already used for atm_put → feed ID collision → ZERO-BAR
    → false SL trigger at stale price).
"""


import json
import logging
import re
from datetime import date
from pathlib import Path


import pandas as pd
from kiteconnect import KiteConnect


log = logging.getLogger("instrument_mapper")


CACHE_FILE = Path(__file__).parent / "instruments_cache.json"


DHAN_SCRIP_CACHE = Path(__file__).parent / "dhan_scrip_cache.parquet"
DHAN_SCRIP_META = Path(__file__).parent / "dhan_scrip_cache.meta.json"
DHAN_SCRIP_URL = "https://images.dhan.co/api-data/api-scrip-master-detailed.csv"


STRIKE_STEP = 50
STRIKE_RANGE_PTS = 500
LOT_SIZE = 65
MAX_LOTS = 1
MIN_TRADE_CAPITAL = 5000


_MONTH_TO_STR = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}
_MONTH_ABBR_TO_NUM = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
    "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
    "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}



# =============================================================================
# SECTION A — Symbol Parser
# =============================================================================


def _parse_nse_symbol(sym: str) -> dict:
    """
    Parse Zerodha/NSE option symbols into canonical fields.

    Supports:
      1) Monthly symbols:
         NIFTY26MAY23700PE
         -> underlying=NIFTY, year=2026, month=5, day=None, strike=23700, option_type=PE

      2) Weekly / dated compact symbols:
         NIFTY2651923500PE
         -> underlying=NIFTY, year=2026, month=5, day=19, strike=23500, option_type=PE
    """
    sym = str(sym).strip().upper()

    m = re.fullmatch(
        r"([A-Z]+)(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d+)(CE|PE)",
        sym,
    )
    if m:
        name, yy, mon_txt, strike, otype = m.groups()
        return {
            "underlying": name,
            "year": 2000 + int(yy),
            "month": _MONTH_ABBR_TO_NUM[mon_txt],
            "day": None,
            "strike": float(strike),
            "option_type": otype,
            "expiry_kind": "MONTHLY_TEXT",
            "source_symbol": sym,
        }

    m = re.fullmatch(r"([A-Z]+)(\d{2})(.+)(CE|PE)", sym)
    if not m:
        raise ValueError(f"Unsupported option symbol format: {sym!r}")

    name, yy, middle, otype = m.groups()
    year = 2000 + int(yy)

    def _try(width: int):
        mon_s = middle[:width]
        dd_s = middle[width:width + 2]
        strike_s = middle[width + 2:]

        if not (mon_s.isdigit() and dd_s.isdigit() and strike_s.isdigit()):
            return None

        mon = int(mon_s)
        dd = int(dd_s)

        if not (1 <= mon <= 12 and 1 <= dd <= 31):
            return None

        return mon, dd, float(strike_s)

    r = _try(2)
    if r and 10 <= r[0] <= 12:
        mon, dd, strike = r
        return {
            "underlying": name,
            "year": year,
            "month": mon,
            "day": dd,
            "strike": strike,
            "option_type": otype,
            "expiry_kind": "DATED_NUMERIC",
            "source_symbol": sym,
        }

    r = _try(1)
    if r and 1 <= r[0] <= 9:
        mon, dd, strike = r
        return {
            "underlying": name,
            "year": year,
            "month": mon,
            "day": dd,
            "strike": strike,
            "option_type": otype,
            "expiry_kind": "DATED_NUMERIC",
            "source_symbol": sym,
        }

    raise ValueError(
        f"Cannot unambiguously parse {sym!r}. "
        f"Expected monthly form like 'NIFTY26MAY23700PE' or "
        f"dated form like 'NIFTY2651923500PE'."
    )


def _zerodha_to_dhan_symbol(tradingsymbol: str) -> str:
    """
    Debug/helper only. Converts a Zerodha symbol into a Dhan display-style
    symbol when possible. Do not use this as the source of truth for lookup.
    """
    p = _parse_nse_symbol(tradingsymbol)
    mon_str = _MONTH_TO_STR[p["month"]]
    return f'{p["underlying"]}-{mon_str}{p["year"]}-{int(p["strike"])}-{p["option_type"]}'



# =============================================================================
# SECTION B — InstrumentMapper
# =============================================================================


class InstrumentMapper:
    def __init__(self, kite: KiteConnect):
        self.kite = kite
        self._cache = {}
        self._dhan_scrip = None
        self._loaded = False

    def load(self):
        cached = self._load_cache()
        if cached:
            self._cache = cached
            log.info(
                f"InstrumentMapper: loaded from cache "
                f"expiry={self._cache.get('_expiry', 'unknown')}"
            )
        else:
            log.info("InstrumentMapper: downloading NFO instruments for Nifty50...")
            df = pd.DataFrame(self.kite.instruments("NFO"))
            df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
            df["strike"] = pd.to_numeric(df["strike"], errors="coerce")

            nf = df[
                (df["name"] == "NIFTY") &
                (df["instrument_type"].isin(["CE", "PE"])) &
                (df["segment"] == "NFO-OPT")
            ].copy()

            if nf.empty:
                raise RuntimeError(
                    "No NIFTY options found in NFO instruments dump. "
                    "Check kite.instruments('NFO') response."
                )

            skip_before = pd.Timestamp.today().normalize() + pd.Timedelta(days=2)
            upcoming = nf[nf["expiry"] >= skip_before]

            if upcoming.empty:
                raise RuntimeError("No upcoming NIFTY expiry found in instruments dump.")

            nearest_expiry = upcoming["expiry"].min().date()
            total_strikes = len(nf[nf["expiry"] == pd.Timestamp(nearest_expiry)])

            self._cache["_expiry"] = str(nearest_expiry)
            self._cache["_total_strikes"] = total_strikes
            self._cache["_loaded_date"] = str(date.today())

            self._save_cache()
            log.info(
                f"InstrumentMapper: nearest NIFTY expiry = {nearest_expiry} "
                f"({total_strikes} strikes for this expiry)"
            )

        self._load_dhan_scrip()
        self._loaded = True

    def _read_dhan_meta(self) -> dict:
        if not DHAN_SCRIP_META.exists():
            return {}
        try:
            return json.loads(DHAN_SCRIP_META.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_dhan_meta(self, payload: dict):
        DHAN_SCRIP_META.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_dhan_scrip(self):
        """
        Load Dhan detailed scrip master with enough metadata for robust lookup.
        Uses a sidecar metadata JSON for cache freshness instead of DataFrame.attrs.
        """
        today_str = str(date.today())

        if DHAN_SCRIP_CACHE.exists() and DHAN_SCRIP_META.exists():
            try:
                meta = self._read_dhan_meta()
                if meta.get("cache_date") == today_str:
                    df = pd.read_parquet(DHAN_SCRIP_CACHE)
                    self._dhan_scrip = df
                    log.info(
                        f"InstrumentMapper: Dhan scrip master loaded from cache "
                        f"({len(df):,} rows)"
                    )
                    return
            except Exception as e:
                log.warning(
                    f"InstrumentMapper: Dhan scrip cache read failed ({e}) — re-downloading"
                )

        log.info("InstrumentMapper: downloading Dhan detailed scrip master CSV...")
        try:
            raw = pd.read_csv(DHAN_SCRIP_URL, low_memory=False)

            log.info(f"InstrumentMapper: Dhan raw rows = {len(raw):,}")
            log.info(f"InstrumentMapper: Dhan columns = {list(raw.columns)}")
            try:
                log.info("InstrumentMapper: Dhan sample rows:\n%s", raw.head(5).to_string())
            except Exception:
                pass

            rename_map = {
                "INSTRUMENT": "SEM_INSTRUMENT_NAME",
                "SM_EXPIRY_DATE": "SEM_EXPIRY_DATE",
                "STRIKE_PRICE": "SEM_STRIKE_PRICE",
                "OPTION_TYPE": "SEM_OPTION_TYPE",
                "EXPIRY_FLAG": "SEM_EXPIRY_FLAG",
                "SYMBOL_NAME": "SM_SYMBOL_NAME",
                "DISPLAY_NAME": "DHAN_DISPLAY_NAME",
            }
            raw = raw.rename(columns=rename_map)

            if "SEM_SMST_SECURITY_ID" not in raw.columns:
                for c in ["SECURITY_ID", "SECURITYID", "SEC_ID"]:
                    if c in raw.columns:
                        raw = raw.rename(columns={c: "SEM_SMST_SECURITY_ID"})
                        break

            required_cols = [
                "SEM_SMST_SECURITY_ID",
                "SEM_INSTRUMENT_NAME",
                "SEM_EXPIRY_DATE",
            ]
            missing = [c for c in required_cols if c not in raw.columns]
            if missing:
                raise RuntimeError(
                    f"Dhan scrip master missing required columns after normalization: {missing}. "
                    f"Available columns: {list(raw.columns)}"
                )

            for col in [
                "SEM_INSTRUMENT_NAME", "SM_SYMBOL_NAME", "UNDERLYING_SYMBOL",
                "SEM_OPTION_TYPE", "SEM_EXPIRY_FLAG", "EXCH_ID", "SEGMENT", "DHAN_DISPLAY_NAME"
            ]:
                if col in raw.columns:
                    raw[col] = raw[col].astype(str).str.upper().str.strip()

            raw["SEM_SMST_SECURITY_ID"] = raw["SEM_SMST_SECURITY_ID"].astype(str).str.strip()

            if "SEM_EXPIRY_DATE" in raw.columns:
                raw["SEM_EXPIRY_DATE"] = pd.to_datetime(raw["SEM_EXPIRY_DATE"], errors="coerce")

            if "SEM_STRIKE_PRICE" in raw.columns:
                raw["SEM_STRIKE_PRICE"] = pd.to_numeric(raw["SEM_STRIKE_PRICE"], errors="coerce")

            if "SEM_INSTRUMENT_NAME" in raw.columns:
                log.info(
                    "InstrumentMapper: INSTRUMENT top values = %s",
                    raw["SEM_INSTRUMENT_NAME"].value_counts(dropna=False).head(15).to_dict()
                )
            if "SM_SYMBOL_NAME" in raw.columns:
                log.info(
                    "InstrumentMapper: SYMBOL_NAME top values = %s",
                    raw["SM_SYMBOL_NAME"].value_counts(dropna=False).head(20).to_dict()
                )
            if "UNDERLYING_SYMBOL" in raw.columns:
                log.info(
                    "InstrumentMapper: UNDERLYING_SYMBOL top values = %s",
                    raw["UNDERLYING_SYMBOL"].value_counts(dropna=False).head(20).to_dict()
                )

            df = raw.copy()

            before = len(df)
            if "SEGMENT" in df.columns:
                df = df[df["SEGMENT"].isin(["D", "DERIVATIVES", "DERIVATIVE"])]
                log.info(f"InstrumentMapper: after SEGMENT filter = {len(df):,}/{before:,}")
                before = len(df)

            if "EXCH_ID" in df.columns:
                df = df[df["EXCH_ID"] == "NSE"]
                log.info(f"InstrumentMapper: after EXCH_ID filter = {len(df):,}/{before:,}")
                before = len(df)

            if "SEM_INSTRUMENT_NAME" in df.columns:
                allowed_instr = {"OPTIDX", "OPTSTK", "OPTCUR", "OPTFUT", "OPT"}
                if df["SEM_INSTRUMENT_NAME"].isin(allowed_instr).any():
                    df = df[df["SEM_INSTRUMENT_NAME"].isin(allowed_instr)]
                    log.info(f"InstrumentMapper: after INSTRUMENT filter = {len(df):,}/{before:,}")
                    before = len(df)

            if "UNDERLYING_SYMBOL" in df.columns and df["UNDERLYING_SYMBOL"].eq("NIFTY").any():
                df = df[df["UNDERLYING_SYMBOL"] == "NIFTY"]
                log.info(f"InstrumentMapper: after UNDERLYING_SYMBOL filter = {len(df):,}/{before:,}")
                before = len(df)
            elif "SM_SYMBOL_NAME" in df.columns and df["SM_SYMBOL_NAME"].eq("NIFTY").any():
                df = df[df["SM_SYMBOL_NAME"] == "NIFTY"]
                log.info(f"InstrumentMapper: after SYMBOL_NAME filter = {len(df):,}/{before:,}")
                before = len(df)
            elif "DHAN_DISPLAY_NAME" in df.columns:
                df = df[df["DHAN_DISPLAY_NAME"].str.contains("NIFTY", na=False)]
                log.info(f"InstrumentMapper: after DISPLAY_NAME fallback filter = {len(df):,}/{before:,}")
                before = len(df)

            if "SEM_OPTION_TYPE" in df.columns:
                valid_opt = {"CE", "PE"}
                if df["SEM_OPTION_TYPE"].isin(valid_opt).any():
                    df = df[df["SEM_OPTION_TYPE"].isin(valid_opt)]
                    log.info(f"InstrumentMapper: after OPTION_TYPE filter = {len(df):,}/{before:,}")
                    before = len(df)

            df = df.dropna(subset=["SEM_SMST_SECURITY_ID", "SEM_EXPIRY_DATE"]).reset_index(drop=True)

            preferred_cols = [
                "SEM_SMST_SECURITY_ID",
                "SEM_INSTRUMENT_NAME",
                "SEM_EXPIRY_DATE",
                "SM_SYMBOL_NAME",
                "UNDERLYING_SYMBOL",
                "SEM_STRIKE_PRICE",
                "SEM_OPTION_TYPE",
                "SEM_EXPIRY_FLAG",
                "DHAN_DISPLAY_NAME",
                "EXCH_ID",
                "SEGMENT",
                "LOT_SIZE",
            ]
            keep_cols = [c for c in preferred_cols if c in df.columns]
            df = df[keep_cols].copy()

            df.to_parquet(DHAN_SCRIP_CACHE, index=False)
            self._write_dhan_meta({"cache_date": today_str, "rows": int(len(df))})
            self._dhan_scrip = df

            log.info(
                f"InstrumentMapper: Dhan detailed scrip master downloaded & cached "
                f"({len(df):,} NIFTY option rows)"
            )

            if df.empty:
                log.warning(
                    "InstrumentMapper: Dhan filtered dataset is EMPTY. "
                    "Check logged columns/sample/top-values above."
                )

        except Exception as e:
            log.error(
                f"InstrumentMapper: Dhan scrip master download FAILED: {e} "
                f"— resolve_dhan_contract() will fail for unknown symbols"
            )
            self._dhan_scrip = pd.DataFrame(
                columns=[
                    "SEM_SMST_SECURITY_ID",
                    "SEM_INSTRUMENT_NAME",
                    "SEM_EXPIRY_DATE",
                    "SM_SYMBOL_NAME",
                    "UNDERLYING_SYMBOL",
                    "SEM_STRIKE_PRICE",
                    "SEM_OPTION_TYPE",
                    "SEM_EXPIRY_FLAG",
                    "DHAN_DISPLAY_NAME",
                ]
            )

    def resolve_dhan_contract(self, tradingsymbol: str) -> dict:
        """
        Resolve Zerodha/NSE option symbol to the exact Dhan contract row.
        """
        if self._dhan_scrip is None or self._dhan_scrip.empty:
            raise RuntimeError("Dhan scrip master not loaded. Call mapper.load() first.")

        p = _parse_nse_symbol(tradingsymbol)
        q = self._dhan_scrip.copy()

        if "UNDERLYING_SYMBOL" in q.columns and q["UNDERLYING_SYMBOL"].eq(p["underlying"]).any():
            q = q[q["UNDERLYING_SYMBOL"] == p["underlying"]]
        elif "SM_SYMBOL_NAME" in q.columns and q["SM_SYMBOL_NAME"].eq(p["underlying"]).any():
            q = q[q["SM_SYMBOL_NAME"] == p["underlying"]]

        if "SEM_OPTION_TYPE" in q.columns:
            q = q[q["SEM_OPTION_TYPE"] == p["option_type"]]

        if "SEM_STRIKE_PRICE" in q.columns:
            q = q[q["SEM_STRIKE_PRICE"] == p["strike"]]

        if p["day"] is not None:
            expiry_date = pd.Timestamp(year=p["year"], month=p["month"], day=p["day"]).date()
            q = q[q["SEM_EXPIRY_DATE"].dt.date == expiry_date]
        else:
            q = q[
                (q["SEM_EXPIRY_DATE"].dt.year == p["year"]) &
                (q["SEM_EXPIRY_DATE"].dt.month == p["month"])
            ]

            if "SEM_EXPIRY_FLAG" in q.columns:
                monthly = q[q["SEM_EXPIRY_FLAG"] == "M"]
                if not monthly.empty:
                    q = monthly

            if len(q) > 1:
                q = q.sort_values("SEM_EXPIRY_DATE").tail(1)

        if len(q) == 0:
            tried = _zerodha_to_dhan_symbol(tradingsymbol)
            raise ValueError(
                f"Dhan security ID not found for {tradingsymbol!r}. "
                f"Parsed={p}. Debug display symbol guess={tried!r}."
            )

        if len(q) > 1:
            sample_cols = [c for c in [
                "DHAN_DISPLAY_NAME", "SEM_SMST_SECURITY_ID", "SEM_EXPIRY_DATE",
                "SEM_STRIKE_PRICE", "SEM_OPTION_TYPE", "UNDERLYING_SYMBOL", "SM_SYMBOL_NAME"
            ] if c in q.columns]
            sample = q[sample_cols].head(5).to_dict("records")
            raise ValueError(
                f"Ambiguous Dhan match for {tradingsymbol!r}. "
                f"Parsed={p}. Matches={sample}"
            )

        row = q.iloc[0]
        return {
            "dhan_security_id": str(row["SEM_SMST_SECURITY_ID"]),
            "dhan_trading_symbol": str(row.get("DHAN_DISPLAY_NAME", "")),
            "dhan_expiry": row["SEM_EXPIRY_DATE"].date().isoformat(),
            "dhan_instrument_name": str(row.get("SEM_INSTRUMENT_NAME", "")),
            "parsed": p,
        }

    def get_dhan_security_id(self, tradingsymbol: str) -> str:
        return self.resolve_dhan_contract(tradingsymbol)["dhan_security_id"]

    def get_expiry(self) -> str:
        return self._cache.get("_expiry", "")

    def get_lot_size(self) -> int:
        return LOT_SIZE

    def get_info(self) -> dict:
        return {k: v for k, v in self._cache.items() if not k.startswith("_")}

    def _load_cache(self) -> dict | None:
        if not CACHE_FILE.exists():
            return None
        try:
            data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            if data.get("_loaded_date") == str(date.today()):
                return data
        except Exception:
            pass
        return None

    def _save_cache(self):
        CACHE_FILE.write_text(json.dumps(self._cache, indent=2), encoding="utf-8")



# =============================================================================
# SECTION C — Dynamic OTM Picker
# =============================================================================


def get_available_capital(
    kite: KiteConnect,
    use_pct: float = 1.0,
    segment: str = "equity",
) -> float:
    """
    Fetches live available cash from Zerodha margins API.
    """
    try:
        margins = kite.margins(segment=segment)
        available = float(
            margins.get("available", {}).get("live_balance", 0)
            or margins.get("available", {}).get("cash", 0)
            or margins.get("net", 0)
        )
        trade_capital = round(available * use_pct, 2)

        log.info(
            f"CAPITAL: available=Rs{available:.2f}  "
            f"use_pct={use_pct * 100:.0f}%  ->  "
            f"trade_capital=Rs{trade_capital:.2f}"
        )

        if trade_capital <= 0:
            log.error("CAPITAL: available balance is zero or negative!")

        return trade_capital

    except Exception as e:
        log.error(f"CAPITAL FETCH FAILED: {e} — skipping trade for safety")
        return 0.0


def get_current_expiry(kite: KiteConnect) -> str:
    """
    Auto-detects the nearest upcoming NIFTY weekly expiry.
    Skips expiries within 2 days so we never land on same-day or next-day expiry.
    Returns expiry as 'YYYY-MM-DD' string.
    """
    instruments = pd.DataFrame(kite.instruments("NFO"))

    nf = instruments[instruments["name"] == "NIFTY"].copy()
    nf["expiry"] = pd.to_datetime(nf["expiry"])

    skip_before = pd.Timestamp.today().normalize() + pd.Timedelta(days=2)
    upcoming = nf[nf["expiry"] >= skip_before]

    if upcoming.empty:
        raise RuntimeError(
            "No upcoming NIFTY expiry found in instruments dump. "
            "Check kite.instruments('NFO') response."
        )

    nearest = upcoming["expiry"].min()
    expiry_str = nearest.strftime("%Y-%m-%d")

    if nearest.weekday() != 1:
        log.warning(
            f"get_current_expiry: resolved expiry {expiry_str} "
            f"is NOT a Tuesday (weekday={nearest.weekday()}). "
            f"Check NSE holiday schedule."
        )

    log.info(f"Auto-detected NIFTY expiry: {expiry_str} ({nearest.day_name()})")
    return expiry_str


def get_option_chain(kite: KiteConnect, expiry: str) -> pd.DataFrame:
    """
    Downloads the full NIFTY option chain for a given expiry.
    """
    instruments = pd.DataFrame(kite.instruments("NFO"))

    instruments["expiry_str"] = pd.to_datetime(
        instruments["expiry"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    nf = instruments[
        (instruments["name"] == "NIFTY") &
        (instruments["expiry_str"] == expiry) &
        (instruments["instrument_type"].isin(["CE", "PE"]))
    ].copy()

    if nf.empty:
        raise RuntimeError(
            f"No NIFTY options found for expiry={expiry}. "
            "Check expiry date or call get_current_expiry() again."
        )

    nf = nf[
        ["strike", "instrument_type", "tradingsymbol", "instrument_token", "lot_size"]
    ].reset_index(drop=True)

    log.info(f"NIFTY option chain loaded: {len(nf)} strikes | expiry={expiry}")
    return nf


def get_otm_strikes(
    chain: pd.DataFrame,
    spot: float,
    option_type: str,
    strike_range: float = STRIKE_RANGE_PTS,
) -> pd.DataFrame:
    """
    Filters strictly OTM strikes within +-strike_range points of ATM.
    Budget filtering happens later in pick_best_otm_strike().

    FIX: Both CE and PE boundaries now use ATM (not spot) so the ATM
    strike itself (distance_pts=0) is always excluded regardless of where
    spot floats between two strike levels.  Using spot as the boundary
    allowed the ATM strike to pass when spot > atm (PE case) or spot < atm
    (CE case), causing a distance_pts=0 candidate to win the score ranking
    and collide with the atm_put / atm_call Dhan feed subscription ID.
    """
    atm = round(spot / STRIKE_STEP) * STRIKE_STEP
    df = chain[chain["instrument_type"] == option_type].copy()

    if option_type == "CE":
        # FIX: was  strike > spot  — changed to  strike > atm
        # Ensures ATM strike (dist=0) is excluded even when spot < atm
        df = df[
            (df["strike"] > atm) &
            (df["strike"] <= atm + strike_range)
        ].copy()
        df["distance_pts"] = df["strike"] - atm
    else:  # PE
        # FIX: was  strike < spot  — changed to  strike < atm
        # Ensures ATM strike (dist=0) is excluded even when spot > atm
        df = df[
            (df["strike"] < atm) &
            (df["strike"] >= atm - strike_range)
        ].copy()
        df["distance_pts"] = atm - df["strike"]

    df["distance_strikes"] = df["distance_pts"] / STRIKE_STEP
    df = df.sort_values("distance_pts").reset_index(drop=True)

    log.info(
        f"OTM candidates ({option_type}): {len(df)} strikes  "
        f"spot={spot:.0f}  ATM={atm:.0f}  range=+-{strike_range:.0f}pts"
    )
    return df


def _score_strike(
    oi: int,
    volume: int,
    iv: float,
    distance_pts: float,
    total_cost: float,
    trade_capital: float,
) -> float:
    """
    Scores a single OTM strike on quality factors.
    """
    score = 0.0

    if oi >= 2_000_000:
        score += 35.0
    elif oi >= 1_000_000:
        score += 30.0
    elif oi >= 500_000:
        score += 24.0
    elif oi >= 200_000:
        score += 16.0
    elif oi >= 50_000:
        score += 8.0
    elif oi >= 10_000:
        score += 3.0

    if volume >= 100_000:
        score += 25.0
    elif volume >= 50_000:
        score += 21.0
    elif volume >= 10_000:
        score += 16.0
    elif volume >= 2_000:
        score += 10.0
    elif volume >= 500:
        score += 5.0
    elif volume >= 100:
        score += 2.0

    if distance_pts <= 50:
        score += 25.0
    elif distance_pts <= 100:
        score += 22.0
    elif distance_pts <= 200:
        score += 17.0
    elif distance_pts <= 300:
        score += 11.0
    elif distance_pts <= 400:
        score += 6.0
    elif distance_pts <= 500:
        score += 2.0

    if iv == 0:
        score += 5.0
    elif iv <= 14:
        score += 10.0
    elif iv <= 18:
        score += 8.0
    elif iv <= 22:
        score += 6.0
    elif iv <= 28:
        score += 3.0
    elif iv <= 35:
        score += 1.0

    usage_pct = total_cost / trade_capital if trade_capital > 0 else 0
    if usage_pct >= 0.50:
        score += 5.0
    elif usage_pct >= 0.25:
        score += 3.0
    elif usage_pct >= 0.10:
        score += 1.0

    return round(score, 2)


def pick_best_otm_strike(
    kite: KiteConnect,
    otm_df: pd.DataFrame,
    trade_capital: float,
    lot_size: int = LOT_SIZE,
    lots: int = MAX_LOTS,
    log_top_n: int = 5,
    mapper: "InstrumentMapper | None" = None,
) -> dict | None:
    """
    Main dynamic OTM picker. Called at signal time every trade.

    If mapper is provided, also resolves and includes:
      - dhan_security_id
      - dhan_trading_symbol
      - dhan_expiry

    livetrading.py MUST use dhan_security_id for Dhan feed matching,
    not Zerodha instrument_token.
    """
    if otm_df.empty:
        log.warning("pick_best_otm_strike: no OTM candidates provided.")
        return None

    max_ltp = trade_capital / (lot_size * lots)
    log.info(
        f"BUDGET: Rs{trade_capital:.2f} / "
        f"(lot={lot_size} x lots={lots})  ->  max LTP = Rs{max_ltp:.2f}"
    )

    sym_list = [f"NFO:{sym}" for sym in otm_df["tradingsymbol"].tolist()]
    try:
        quotes = kite.quote(sym_list)
    except Exception as e:
        log.error(f"Quote fetch failed: {e}")
        return None

    affordable = []
    skipped_cost = []
    skipped_liq = []

    for _, row in otm_df.iterrows():
        sym_key = f"NFO:{row['tradingsymbol']}"
        q = quotes.get(sym_key, {})
        ltp = float(q.get("last_price", 0))
        oi = int(q.get("oi", 0))
        vol = int(q.get("volume", 0))
        iv = float(q.get("implied_volatility", 0))

        if ltp <= 0:
            continue

        total_cost = ltp * lot_size * lots

        if total_cost > trade_capital:
            skipped_cost.append(
                f"{row['tradingsymbol']:32s} "
                f"ltp=Rs{ltp:.1f}  cost=Rs{total_cost:.0f} > Rs{trade_capital:.0f}"
            )
            continue

        if oi == 0 and vol == 0:
            skipped_liq.append(row["tradingsymbol"])
            continue

        score = _score_strike(
            oi=oi,
            volume=vol,
            iv=iv,
            distance_pts=float(row["distance_pts"]),
            total_cost=total_cost,
            trade_capital=trade_capital,
        )

        affordable.append({
            "tradingsymbol": row["tradingsymbol"],
            "instrument_token": int(row["instrument_token"]),
            "dhan_security_id": None,
            "dhan_trading_symbol": None,
            "dhan_expiry": None,
            "strike": float(row["strike"]),
            "distance_pts": float(row["distance_pts"]),
            "distance_strikes": float(row["distance_strikes"]),
            "lot_size": lot_size,
            "lots": lots,
            "ltp": ltp,
            "total_cost": total_cost,
            "oi": oi,
            "volume": vol,
            "iv": iv,
            "score": score,
        })

    if skipped_cost:
        log.info(f"TOO EXPENSIVE — {len(skipped_cost)} strikes filtered:")
        for s in skipped_cost[:5]:
            log.info(f"  x {s}")

    if skipped_liq:
        log.info(
            f"ILLIQUID (OI=0 + vol=0) — {len(skipped_liq)} strikes skipped"
        )

    if not affordable:
        log.warning(
            f"SKIP — no OTM strike fits within Rs{trade_capital:.0f} budget!"
        )
        return None

    affordable.sort(key=lambda x: x["score"], reverse=True)

    log.info(
        f"RANKED AFFORDABLE STRIKES "
        f"(top {min(log_top_n, len(affordable))}):"
    )
    for i, r in enumerate(affordable[:log_top_n]):
        log.info(
            f"  #{i+1} {r['tradingsymbol']:32s} | "
            f"strike={r['strike']:.0f} | "
            f"dist={r['distance_pts']:.0f}pts | "
            f"ltp=Rs{r['ltp']:.1f} | "
            f"cost=Rs{r['total_cost']:.0f} | "
            f"OI={r['oi']:,} | vol={r['volume']:,} | "
            f"score={r['score']:.1f}"
        )

    best = affordable[0]

    if mapper is not None:
        try:
            resolved = mapper.resolve_dhan_contract(best["tradingsymbol"])
            best["dhan_security_id"] = resolved["dhan_security_id"]
            best["dhan_trading_symbol"] = resolved["dhan_trading_symbol"]
            best["dhan_expiry"] = resolved["dhan_expiry"]

            log.info(
                f"SELECTED -> {best['tradingsymbol']} | "
                f"strike={best['strike']:.0f} | "
                f"ltp=Rs{best['ltp']:.1f} | "
                f"cost=Rs{best['total_cost']:.0f} | "
                f"score={best['score']:.1f} | "
                f"dhan_id={best['dhan_security_id']} | "
                f"dhan_symbol={best['dhan_trading_symbol']} | "
                f"dhan_expiry={best['dhan_expiry']} | "
                f"zerodha_token={best['instrument_token']}"
            )
        except (ValueError, RuntimeError) as e:
            log.warning(
                f"Could not resolve Dhan contract for {best['tradingsymbol']}: {e} "
                f"— trade will be rejected by entry flow until Dhan contract resolution works"
            )
            log.info(
                f"SELECTED -> {best['tradingsymbol']} | "
                f"strike={best['strike']:.0f} | "
                f"ltp=Rs{best['ltp']:.1f} | "
                f"cost=Rs{best['total_cost']:.0f} | "
                f"score={best['score']:.1f}"
            )
    else:
        log.info(
            f"SELECTED -> {best['tradingsymbol']} | "
            f"strike={best['strike']:.0f} | "
            f"ltp=Rs{best['ltp']:.1f} | "
            f"cost=Rs{best['total_cost']:.0f} | "
            f"score={best['score']:.1f}"
        )

    return best



# =============================================================================
# SELF-TEST
# =============================================================================
if __name__ == "__main__":
    test_cases = [
        ("NIFTY2651923550CE", 2026, 5, 19, 23550.0, "CE"),
        ("NIFTY2651923500PE", 2026, 5, 19, 23500.0, "PE"),
        ("NIFTY26101023500CE", 2026, 10, 10, 23500.0, "CE"),
        ("NIFTY26121523500CE", 2026, 12, 15, 23500.0, "CE"),
        ("NIFTY2611923550CE", 2026, 1, 19, 23550.0, "CE"),
        ("NIFTY2691923550CE", 2026, 9, 19, 23550.0, "CE"),
        ("NIFTY26101524000CE", 2026, 10, 15, 24000.0, "CE"),
        ("NIFTY2531524000CE", 2025, 3, 15, 24000.0, "CE"),
        ("NIFTY26110524500PE", 2026, 11, 5, 24500.0, "PE"),
        ("NIFTY26120724000CE", 2026, 12, 7, 24000.0, "CE"),
        ("NIFTY2511924500CE", 2025, 1, 19, 24500.0, "CE"),
        ("NIFTY26MAY23700PE", 2026, 5, None, 23700.0, "PE"),
    ]

    # ── get_otm_strikes() boundary test ───────────────────────────────
    import pandas as _pd
    print("=== get_otm_strikes() ATM-exclusion test ===")
    _chain = _pd.DataFrame({
        "strike": [23300, 23350, 23400, 23450, 23500, 23550, 23600],
        "instrument_type": ["PE", "PE", "PE", "PE", "CE", "CE", "CE"],
        "tradingsymbol": [f"NIFTY2660923{s}{'PE' if s<=23450 else 'CE'}" for s in [23300,23350,23400,23450,23500,23550,23600]],
        "instrument_token": list(range(7)),
        "lot_size": [65]*7,
    })
    for _spot, _atm in [(23472, 23450), (23428, 23450), (23450, 23450)]:
        _pe = get_otm_strikes(_chain, _spot, "PE")
        _ce = get_otm_strikes(_chain, _spot, "CE")
        _pe_strikes = _pe["strike"].tolist()
        _ce_strikes = _ce["strike"].tolist()
        _atm_in_pe  = _atm in _pe_strikes
        _atm_in_ce  = _atm in _ce_strikes
        _ok = not _atm_in_pe and not _atm_in_ce
        print(
            f"{'OK' if _ok else 'FAIL'}  spot={_spot}  ATM={_atm}  "
            f"PE_strikes={_pe_strikes}  CE_strikes={_ce_strikes}  "
            f"ATM_in_PE={_atm_in_pe}  ATM_in_CE={_atm_in_ce}"
        )

    print()
    print("=== _parse_nse_symbol + _zerodha_to_dhan_symbol self-test ===")
    all_pass = True

    for sym, exp_yr, exp_mon, exp_day, exp_strike, exp_type in test_cases:
        try:
            p = _parse_nse_symbol(sym)
            dhan = _zerodha_to_dhan_symbol(sym)
            ok = (
                p["year"] == exp_yr and
                p["month"] == exp_mon and
                p["day"] == exp_day and
                p["strike"] == exp_strike and
                p["option_type"] == exp_type
            )
            print(f"{'OK' if ok else 'FAIL'} {sym:32s} -> {p} -> {dhan}")
            if not ok:
                all_pass = False
        except Exception as e:
            all_pass = False
            print(f"EXCP {sym:32s} -> {e}")

    print(f"\n{'ALL PASS ✓' if all_pass else 'SOME FAILED ✗'}")
