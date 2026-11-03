# ══════════════════════════════════════════════════════════════════════
#  v14 LIVE PAPER TRADING — Dhan WebSocket — v12_l5 STANDALONE VARIANT
#  File: live_paper_v14_dhan_l5_standalone.py
#
#  Runs v12_l5 (3-class multiclass meta) as its OWN independent process,
#  separate Dhan feed connection, separate buffer, separate warm-start,
#  separate log file and trade CSV -- safe to run side-by-side with
#  live_paper_v14_dhan_l3.py (the unmodified v12_l3 bot) in another
#  terminal/VS Code window. The two never share state or a process.
#
#  CHANGES FROM v13:
#  1.  Momentum score gate — updated thresholds:
#        conf >= 0.75 (was 0.78), adx >= 20 (was 22),
#        trend_aligned uses OR logic (was AND),
#        consec_bars >= 2 (was 3), ATR range 1.5–8.0 (was 2.5–4.5),
#        entry_clean <= 102% of avg (was 101.5%),
#        strong_trend bonus adx >= 28 (was 30)
#  2.  Smart Exit System — replaces TP entirely. 5-priority chain:
#        Hard SL → IV Z-Score → OI Unwind → Supertrend Flip → Time Stop
#  3.  compute_sl() replaces compute_tp_sl(); SL only, no TP level
#  4.  New helpers: compute_iv_zscore(), compute_oi_change_pct(),
#        compute_supertrend(), smart_exit_check()
#  5.  get_tp_multiplier() removed (no TP in v14)
#  6.  MAX_BARS_HELD = 80 (was 15)
#  7.  SL_MIN_PCT = 1.5% (was 2.0%)
#
#  CHAIN-GRID FIX (previous revision):
#  The model was trained on the full ATM-10..ATM+10 raw chain grid, but the
#  bot only fed the 9 ATM raw columns (115 far-strike features were 0.0 on
#  every bar). chain_grid.ChainGrid now subscribes to the missing strikes,
#  builds open/volume/iv/oi per bar, and merges them into raw_row.
#    - live_warmstart.LAST_PIN_STRIKES centres the grid on the week's
#      pinned strike (matches training)
#    - resolve_instruments() keeps the weekly scrip rows (_WEEKLY_DF)
#    - process_tick(): flush at the minute boundary, on_tick per tick
#    - build_raw_row(): single source for both raw_row constructions
#    - feed_resubscribe(): never (un)subscribes grid legs
#
#  NETWORK / VIX FIX (this revision):
#  Startup showed HTTP 504 from Dhan's historical/rolling-option endpoints
#  (live_warmstart [CALL ATM], [PUT ATM], [CALL ATM-8] ...) and HTTP 503 from
#  NSE's historical-VIX endpoint (fetch_historical_vix), which left india_vix
#  constant-filled for the whole warm-started history.
#    - NET RETRY: a retrying HTTPAdapter is mounted on every requests.Session
#      for Dhan's READ-ONLY chart endpoints (/v2/charts/*) and NSE's /api/*.
#      Transient 429/5xx responses are retried with exponential backoff
#      before live_warmstart ever sees them. Order endpoints are NOT covered.
#    - VIX BACKFILL: if the warm-started india_vix column is still constant
#      after warm-start, it is rebuilt from 1-minute India VIX candles via
#      Zerodha (token 264969) and, failing that, Dhan (IDX_I id 21).
#    - RULE-PIN FALLBACK: if the pinned-CALL warm-start history comes back
#      short (e.g. because of the 504s), it is seeded from the ChainGrid leg
#      history for the SAME pinned strike, and a loud warning explains how
#      long the 'rulepinwarmup' gate will block entries otherwise.
# ══════════════════════════════════════════════════════════════════════

import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
import socket
import asyncio
import numpy as np
import pandas as pd
from datetime import time, datetime, date, timedelta
from pathlib import Path
from collections import deque
import joblib, json, inspect, time as time_mod
import urllib.request
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import dhanhq
import dhanhq.marketfeed as mf
from config import DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN, ZERODHA_CONFIG, api_key, api_secret
import logging, sys
from py_vollib.black_scholes.implied_volatility import implied_volatility as _bs_iv
from kite_auth import get_kite
from instrument_mapper import (
    InstrumentMapper,
    get_available_capital,
    get_current_expiry,
    get_option_chain,
    get_otm_strikes,
    pick_best_otm_strike,
    STRIKE_RANGE_PTS,
    LOT_SIZE,
    MAX_LOTS,
    MIN_TRADE_CAPITAL,
)
from order_manager import OrderManager
from risk_gate import RiskGate
from position_sync import PositionSync
from chain_grid import ChainGrid
GRID = None            # built in __main__ after warm-start
_WEEKLY_DF = None      # set by resolve_instruments
_WEEKLY_EXPIRY = None # ← PATCH 2: far-strike raw chain columns
logging.getLogger("asyncio").setLevel(logging.CRITICAL)


class _SpotPriceUnavailable(RuntimeError):
    pass


def _validated_spot(value, source):
    try:
        spot = float(value)
    except (TypeError, ValueError):
        raise RuntimeError(f"{source} returned an invalid NIFTY spot price: {value!r}") from None
    if not np.isfinite(spot) or spot <= 0:
        raise RuntimeError(f"{source} returned an invalid NIFTY spot price: {spot}")
    return spot


# ══════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ══════════════════════════════════════════════════════════════════════
LOG_DIR  = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"paper_trade_l3_{date.today().strftime('%Y-%m-%d')}.log"

logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt  = "%H:%M:%S",
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ]
)
log = logging.getLogger("v14_l5")

log.info("=" * 60)
log.info(f"  v14 L5 STANDALONE PAPER TRADER  started  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log.info(f"  Log file : {LOG_FILE}")
log.info("=" * 60)

log.info(f"dhanhq version : {getattr(dhanhq, '__version__', 'unknown')}")
_feed_cls = getattr(mf, "MarketFeed", None) or getattr(mf, "DhanFeed", None)
if _feed_cls:
    log.info(f"Feed class     : {_feed_cls.__name__}  sig={inspect.signature(_feed_cls.__init__)}")
else:
    log.warning("No known feed class found in dhanhq.marketfeed")
log.info(f"marketfeed dir : {[x for x in dir(mf) if not x.startswith('_')]}")


# ══════════════════════════════════════════════════════════════════════
# CREDENTIALS
# ══════════════════════════════════════════════════════════════════════
CLIENT_ID    = DHAN_CLIENT_ID
ACCESS_TOKEN = DHAN_ACCESS_TOKEN


# ══════════════════════════════════════════════════════════════════════
# NETWORK RETRY LAYER  (fixes transient HTTP 504 from Dhan / 503 from NSE)
# ══════════════════════════════════════════════════════════════════════
# live_warmstart.py / chain_grid.py / buffer_warmstart.py make their HTTP
# calls through `requests`. They log "HTTP 504 [CALL ATM]" and
# "fetch_historical_vix: HTTP 503" and give up after one or two tries.
# Rather than edit those modules, we mount a retrying adapter on every new
# requests.Session for the READ-ONLY data endpoints only:
#     https://api.dhan.co/v2/charts/*   (historical / intraday / rollingoption)
#     https://www.nseindia.com/api/*    (allIndices, historical VIX)
# Order-placement endpoints are deliberately NOT matched, so POST retries
# can never double-send an order. After the retries are exhausted the last
# response is returned unchanged (raise_on_status=False), so the callers'
# existing "HTTP 504 ..." handling still works exactly as before.
#
# NOTE: a 504 is a gateway timeout. If Dhan times out because the requested
# date window is simply too large, retrying the SAME window can keep failing;
# the permanent fix for that is to request the window in smaller chunks
# inside live_warmstart.py.
_NET_RETRY_STATUS   = (429, 500, 502, 503, 504)
_NET_RETRY_PREFIXES = (
    "https://api.dhan.co/v2/charts",
    "https://www.nseindia.com/api/",
)
NET_RETRY_TOTAL    = 3      # extra attempts per request (4 tries in all)
NET_RETRY_BACKOFF  = 2.0    # sleeps ≈ 0s, 4s, 8s between tries


def _make_retry_adapter():
    kwargs = dict(
        total=NET_RETRY_TOTAL,
        connect=NET_RETRY_TOTAL,
        read=NET_RETRY_TOTAL,
        status=NET_RETRY_TOTAL,
        backoff_factor=NET_RETRY_BACKOFF,
        status_forcelist=_NET_RETRY_STATUS,
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    try:
        retry = Retry(allowed_methods=None, **kwargs)        # urllib3 >= 1.26: any verb
    except TypeError:
        retry = Retry(method_whitelist=False, **kwargs)      # urllib3 < 1.26
    return HTTPAdapter(max_retries=retry)


def install_net_retry():
    if getattr(requests.Session, "_net_retry_patched", False):
        return
    _orig_init = requests.Session.__init__

    def _init_with_retry(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        for _prefix in _NET_RETRY_PREFIXES:
            self.mount(_prefix, _make_retry_adapter())

    requests.Session.__init__ = _init_with_retry
    requests.Session._net_retry_patched = True
    log.info(f"[NET] retry adapter installed for {list(_NET_RETRY_PREFIXES)}  "
             f"total={NET_RETRY_TOTAL} backoff={NET_RETRY_BACKOFF}s "
             f"statuses={list(_NET_RETRY_STATUS)}")


install_net_retry()


# ══════════════════════════════════════════════════════════════════════
# INSTRUMENT SECURITY IDs
# ══════════════════════════════════════════════════════════════════════
INSTRUMENTS = {
    "nifty_spot"  : ("IDX_I",   "13"),
    "atm_call"    : ("NSE_FNO", "74155"),  # NIFTY-May2026-24050-CE  ← features (ATM CE)
    "atm_put"     : ("NSE_FNO", "74156"),  # NIFTY-May2026-24050-PE  ← features (ATM PE)
    "otm_call"    : ("NSE_FNO", "74165"),  # NIFTY-May2026-24150-CE  ← execution (ATM+100 CE)
    "otm_put"     : ("NSE_FNO", "74146"),  # NIFTY-May2026-23950-PE  ← execution (ATM-100 PE)
}

# ── CORRECT THESE ──────────────────────────────────────────────────
STRIKE_STEP      = 50       # ← was 20 in your code, Nifty step is 50
OTM_OFFSET       = 1        # 2×50 = 100pts OTM (keep for fallback)



# ══════════════════════════════════════════════════════════════════════
# ATR PARAMETERS
# ══════════════════════════════════════════════════════════════════════
ATR_PERIOD  = 14

# ── SL parameters (TP removed; smart exit handles profit-side exits) ──
ATR_SL_MULT  = 3.0        # SL = entry − clamp(ATR_SL_MULT × ATR, min, max)
SL_MIN_PCT   = 0.25      # 15% minimum SL distance  (was 2.0%)
SL_MAX_PCT   = 0.30      # 30.0% maximum SL distance
# ── SL parameters (static point-based, v14.1) ──
SL_POINTS         = 20.0   # flat SL distance in premium points
SL_MIN_PRICE_FLOOR = 0.5   # never let sl_price go to/below zero

SL_PCT_BASE  = 0.005
OTM_MULT     = 1.0 + (OTM_OFFSET * 0.25)
SL_PCT_FIXED = SL_PCT_BASE * OTM_MULT    # fallback when ATR unavailable

# ── Entry gate thresholds ──────────────────────────────────────────────
MINCONF     = 0.75        # rule filter; momentum gate has own conf check
MINADX      = 22          # rule filter; momentum gate checks adx >= 20 too
MINRVOL     = 0.8
COOLDOWNBARS     = 3
SLREENTRYBARS   = 5
MOMENTUM_SCORE_THRESHOLD = 4    # out of 8 — unchanged (intentionally left as-is)

# ── Reversal guard (v14 fix, 2026-07-14) ───────────────────────────────────
# Block entries that fight a just-started underlying reversal. A SHORT bets the
# underlying FALLS, so block if it has been RISING over the last N bars; a LONG
# bets it RISES, so block if FALLING. Uses spotbuffer (underlying), which is
# direction-agnostic and NOT fed to the L3 model — no train/serve desync.
REVERSAL_GUARD_ENABLED = True
REVERSAL_GUARD_BARS    = 5      # window (bars) over which to measure the move
REVERSAL_GUARD_PCT     = 0.03   # % move over the window that counts as a reversal
RISK_FREE_RATE = 0.065   # approx India T-bill rate; adjust as needed

# ── Smart Exit parameters ──────────────────────────────────────────────
IV_ZSCORE_EXIT     = 1.8    # IV z-score threshold to trigger exit
IV_ZSCORE_LOOKBACK = 20     # bars for IV mean/std calculation
IV_MIN_PROFIT_PCT  = 1.5    # minimum pnl% before IV exit fires
OI_DROP_THRESH     = -3.0   # OI change% below this triggers exit
OI_MIN_PROFIT_PCT  = 2.0    # minimum pnl% before OI exit fires
ST_MIN_PROFIT_PCT  = 4.0    # minimum pnl% before Supertrend exit fires
MAX_BARS_HELD      = 80     # timeout (was 15 in v13)
MIN_BARS_PROFIT_EXIT = 3     # v14 fix (2026-07-14): no profit-take exit (IV/OI/Supertrend)
                             # before this many bars held. Blocks bar-3 whipsaw exits
                             # (e.g. Trade 2 SUPERTREND) and lets winners develop.
                             # Hard SL stays active the whole time.
ST_PERIOD          = 10     # Supertrend ATR period
ST_MULT            = 3.0 # Supertrend multiplier
SL_MIN_BARS_HELD   = 5  


# ══════════════════════════════════════════════════════════════════════
# STRIKE TRACKING — needed because Dhan ticks don't carry the strike price
# ══════════════════════════════════════════════════════════════════════
INSTRUMENT_STRIKES = {
    "atm_call": None,
    "atm_put":  None,
    "otm_call": None,
    "otm_put":  None,
}

# ══════════════════════════════════════════════════════════════════════
# LOAD V12 MODEL
# ══════════════════════════════════════════════════════════════════════
from v12_inference import V12Model, PaperTracker, _build_features, PERSIST_WINDOW
# B3/B4 attempt 2026-07-17 REVERTED: v12_l4 fixed the SHORT bias but dropped
# recent-data win rate 71%->55% and 2.7x trade frequency, so it was not a drop-in
# upgrade. l4 artifacts are kept in niftytrading/nifty_v12_l4_models for a later,
# win-rate-preserving bias fix. Live bot stays on l3 for now.
MODEL_DIR = Path(__file__).resolve().parent / "nifty_v12_l3_models"
model     = V12Model(MODEL_DIR)
tracker   = PaperTracker()
kite          = None
order_manager = None
expiry        = None
syncer        = None

# ══════════════════════════════════════════════════════════════════════
# CHAIN GRID STATE  (PATCH 2)
# ══════════════════════════════════════════════════════════════════════
_EXPIRY_DATE_OBJ = None  # datetime.date of the weekly expiry, set in __main__
mapper      = None   # InstrumentMapper, set in __main__

# ══════════════════════════════════════════════════════════════════════
# ROLLING BUFFER
# ══════════════════════════════════════════════════════════════════════
# BUFFER_SIZE was hardcoded 250 -- 30x too short for the long-window rank features.
from buffer_warmstart import BUFFER_SIZE, FULL_WINDOW
buffer = {k: deque(maxlen=BUFFER_SIZE) for k in [
    "datetime", "spot_close",
    "call_close", "call_high", "call_low", "call_open",
    "put_close", "put_high", "put_low", "put_open",
    "call_volume", "put_volume",
    "call_oi", "put_oi", "call_iv", "put_iv", "india_vix",
    "otm_call_high", "otm_call_low", "otm_call_close",
    "otm_put_high",  "otm_put_low",  "otm_put_close",
]}

CHAIN_OFFSETS = list(range(-10, 11))
chain_buf = deque(maxlen=BUFFER_SIZE)

# Rule-only pinned CALL series. Never substitute this into the ML buffer.
RULE_PIN_STRIKE = None
RULE_PIN_SECURITY_ID = None
RULE_PIN_CLOSES = deque(maxlen=BUFFER_SIZE)
RULE_PIN_MIN_ROWS = 175   # _compute_rule_pin_5m() needs this many closes before the 5m gate opens

def _build_buf_adapter():
    return {
        "call_close": list(buffer["call_close"]),
        "call_high":  list(buffer["call_high"]),
        "call_low":   list(buffer["call_low"]),
        "call_open":  list(buffer["call_open"]),
        "call_vol":   list(buffer["call_volume"]),
        "put_close":  list(buffer["put_close"]),
        "put_high":   list(buffer["put_high"])  if buffer.get("put_high") else [buffer["put_close"][-1]] * len(buffer["put_close"]),
        "put_low":    list(buffer["put_low"])   if buffer.get("put_low")  else [buffer["put_close"][-1]] * len(buffer["put_close"]),
        "put_open":   list(buffer["put_open"])  if buffer.get("put_open") else [buffer["put_close"][-1]] * len(buffer["put_close"]),
        "put_vol":    list(buffer["put_volume"]),
        "call_oi":    list(buffer["call_oi"]),
        "put_oi":     list(buffer["put_oi"]),
        "call_iv":    list(buffer["call_iv"]),
        "put_iv":     list(buffer["put_iv"]),
        "spot":       list(buffer["spot_close"]),
        "datetime":   list(buffer["datetime"]),
    }


def build_raw_row():
    """
    PATCH 6: single source for the raw chain columns fed to _build_features.
    The 9 ATM columns come from the live buffer; every far-strike column the
    model expects comes from the ChainGrid (previously all left at 0.0).
    """
    i = -1
    row = {
        "nifty_WEEK_ATM_CALL_1m__open":   float(buffer["call_open"][i]),
        "nifty_WEEK_ATM_CALL_1m__volume": float(buffer["call_volume"][i]),
        "nifty_WEEK_ATM_CALL_1m__iv":     float(buffer["call_iv"][i] or 18.0),
        "nifty_WEEK_ATM_CALL_1m__oi":     float(buffer["call_oi"][i] or 0.0),
        "nifty_WEEK_ATM_CALL_1m__spot":   float(buffer["spot_close"][i]),
        "nifty_WEEK_ATM_PUT_1m__open":    float(buffer["put_open"][i]),
        "nifty_WEEK_ATM_PUT_1m__volume":  float(buffer["put_volume"][i]),
        "nifty_WEEK_ATM_PUT_1m__iv":      float(buffer["put_iv"][i] or 19.0),
        "nifty_WEEK_ATM_PUT_1m__oi":      float(buffer["put_oi"][i] or 0.0),
    }
    return row

def _empty_chain_snapshot():
    return {
        off: {
            "CALL": {"open": 0.0, "volume": 0, "iv": 0.0, "oi": 0, "spot": 0.0},
            "PUT" : {"open": 0.0, "volume": 0, "iv": 0.0, "oi": 0, "spot": 0.0},
        }
        for off in CHAIN_OFFSETS
    }

def _reset_bar_acc():
    return {
        "call": {
            "open": None, "high": -np.inf, "low": np.inf,
            "close": None, "volume": 0, "oi": 0, "iv": 0.0
        },
        "put": {
            "open": None, "high": -np.inf, "low": np.inf,
            "close": None, "volume": 0, "oi": 0, "iv": 0.0
        },
        "otm_call": {
            "high": -np.inf, "low": np.inf, "close": None
        },
        "otm_put": {
            "high": -np.inf, "low": np.inf, "close": None
        },
        "trade": {
            "open": None,
            "high": -np.inf,
            "low": np.inf,
            "close": None,
            "volume": 0,
            "oi": 0,
        },
        "spot": {
            "close": None
        },
        "bar_minute": None,
    }

bar_acc = _reset_bar_acc()

# ── A1: per-bar volume tracker (train/serve skew fix) ──────────────────
# Dhan's `volume` field is CUMULATIVE day volume, not per-tick volume.
# Training ce_atm0_volume is true per-BAR volume. We track the running
# cumulative total per side and emit (cum_now - cum_prev) as the bar's
# volume. Module-level so it survives bar_acc resets on feed reconnect.
_last_cum_call = None
_last_cum_put = None

# A3: fires the serve-consistency self-check once per session at warmup end
_SERVE_CHECK_DONE = False
_WINDOW_CHECK_DONE = False   # confirms rolling_rank/vix_ma20 are reading
                              # FULL_WINDOW, not the old hardcoded 100-bar
                              # window -- fires once, right after warm-start

active_trade = None
INDIA_VIX = 16.0
FEED = None
_bar_count = 0
_cooldown_remaining = 0
_last_exit_bar = -999
_last_exit_reason = None
_last_signal_dir = None

# --- patched live traded instrument tracking ---
_ACTIVE_TRADE_FEED_ID = None
_ACTIVE_TRADE_FEED_SYMBOL = None
_LAST_TRADE_PRICE = None
_LAST_TRADE_TICK_TS = None
# --- one-time diagnostic flags for this session ---
_RAW_TICK_LOGGED = False
_ROUTING_CHECK_LOGGED = False
_LIVE_BARS_SINCE_WARMSTART = 0     # counts LIVE bars only, post-warm-start
_WARMSTART_USED = False            # set True only if warm-start actually ran
POST_WARMSTART_COOLDOWN_BARS = 20  # block entries until this many live bars
                                     # have arrived after warm-start, so ADX/
                                     # RSI/etc. are computed entirely from
                                     # connected live data, never straddling
                                     # the warm-start/live seam (the ADX=97.33
                                     # first-bar entry was the concrete case
                                     # that motivated this)


# ══════════════════════════════════════════════════════════════════════
# MOMENTUM SCORE GATE  (updated thresholds vs v13)
# ══════════════════════════════════════════════════════════════════════
def momentum_score(conf, adx, adx_dir_aligned, trend_aligned,
                   consec_bars_in_dir, spread_widening,
                   entry_clean, atr):
    """
    Score 0–8.  Entry allowed only when score >= MOMENTUM_SCORE_THRESHOLD (5).

    Factors  (updated in v14):
      1. conf >= 0.75                      — model confidence (was 0.78)
      2. adx >= 20 AND dir-aligned pdi/ndi — trend strength + direction (was adx>=22)
      3. trend_aligned (OR logic)          — 5m RSI OR MACDH agrees with signal (was AND)
      4. consec_bars_in_dir >= 2           — at least 2 consecutive rising premium bars (was 3)
      5. spread_widening                   — H-L range > 110% of 10-bar average
      6. entry_clean                       — close <= 102% of 5-bar avg (was 101.5%)
      7. 1.5 <= atr <= 8.0                — ATR in healthy option-premium range (was 2.5-4.5)
      8. (bonus) adx >= 28                — strong trend conviction (was 30)
    """
    score = 0
    if conf >= 0.75:                    score += 1
    if adx >= 20 and adx_dir_aligned:  score += 1
    if trend_aligned:                   score += 1
    if consec_bars_in_dir >= 2:         score += 1
    if spread_widening:                 score += 1
    if entry_clean:                     score += 1
    if 1.5 <= atr <= 8.0:              score += 1
    if adx >= 28:                       score += 1
    return score


def _count_consec(prem_history, lookback=6):
    """
    Count consecutive bars where premium was rising (each bar > previous).
    Iterates backwards and stops at the first non-increase.
    """
    if len(prem_history) < 2:
        return 0
    count = 0
    bars = prem_history[-(lookback + 1):]
    for i in range(len(bars) - 1, 0, -1):
        if bars[i] > bars[i - 1]:
            count += 1
        else:
            break
    return count


def compute_momentum_score_inputs(feat, signal, atr_val,
                                  prem_history, spot_history, leg=None):
    """
    Derives all 8 inputs for momentum_score() from feat dict + buffers.
    Returns (score, score_inputs_dict) for logging.
    """
    conf = feat.get("conf_proxy", 0.0)
    adx  = feat.get("adx_1m", 0)
    pdi  = feat.get("pdi_1m", 50)
    ndi  = feat.get("ndi_1m", 50)

    # Directional alignment: pdi > ndi for LONG, ndi > pdi for SHORT
    adx_dir_aligned = (pdi > ndi) if signal == "LONG" else (ndi > pdi)

    # trend_aligned: 5m RSI OR MACDH agrees with direction (v14: OR, not AND)
    rsi_5m   = feat.get("rsi_5m",  50)
    macdh_5m = feat.get("macdh_5m", 0)
    if signal == "LONG":
        trend_aligned = (rsi_5m > 50 or macdh_5m > 0)
    else:
        trend_aligned = (rsi_5m < 50 or macdh_5m < 0)

    # consecutive rising premium bars (threshold lowered to 2 in v14)
    consec = _count_consec(list(prem_history))

    # source series: the exact leg being traded (grid mode) or legacy otm buffers
    if leg is not None:
        h_buf, l_buf, c_buf = list(leg["high"]), list(leg["low"]), list(leg["close"])
    elif signal == "LONG":
        h_buf = list(buffer["otm_call_high"]); l_buf = list(buffer["otm_call_low"])
        c_buf = list(buffer["otm_call_close"])
    else:
        h_buf = list(buffer["otm_put_high"]);  l_buf = list(buffer["otm_put_low"])
        c_buf = list(buffer["otm_put_close"])

    spread_widening = False
    if len(h_buf) >= 10 and len(l_buf) >= 10:
        ranges      = np.array(h_buf) - np.array(l_buf)
        avg_range   = float(np.mean(ranges[-10:]))
        last_range  = float(ranges[-1]) if len(ranges) > 0 else 0
        spread_widening = last_range > avg_range * 1.1

    entry_clean = True
    if len(c_buf) >= 5:
        avg5 = float(np.mean(c_buf[-5:]))
        last = float(c_buf[-1])
        entry_clean = last <= avg5 * 1.02

    atr_ok = (1.5 <= atr_val <= 8.0) if atr_val else False

    score = momentum_score(
        conf              = conf,
        adx               = adx,
        adx_dir_aligned   = adx_dir_aligned,
        trend_aligned     = trend_aligned,
        consec_bars_in_dir= consec,
        spread_widening   = spread_widening,
        entry_clean       = entry_clean,
        atr               = atr_val or 0,
    )

    inputs = {
        "conf_ok"        : conf >= 0.75,
        "adx_dir_ok"     : adx >= 20 and adx_dir_aligned,
        "trend_aligned"  : trend_aligned,
        "consec_bars"    : consec,
        "spread_widening": spread_widening,
        "entry_clean"    : entry_clean,
        "atr_ok"         : atr_ok,
        "strong_trend"   : adx >= 28,
        "score"          : score,
    }
    return score, inputs


# ══════════════════════════════════════════════════════════════════════
# SMART EXIT HELPERS
# ══════════════════════════════════════════════════════════════════════

def compute_live_iv(option_price, spot, strike, expiry_date, option_type, r=RISK_FREE_RATE):
    """
    Back out implied volatility from live LTP using Black-Scholes inversion.
    Needed because Dhan's WebSocket Full packet does NOT include an
    implied_volatility field (confirmed via [DIAG] raw tick dump).

    option_type: 'c' for call, 'p' for put (py_vollib convention).
    Returns None if solver fails (deep ITM/OTM, near-zero time, bad price,
    or price below intrinsic value) — caller should fall back gracefully.
    Returned value is in PERCENTAGE POINTS (e.g. 18.5, not 0.185) to match
    the existing 18.0/19.0 fallback scale used elsewhere in the codebase.
    """
    try:
        if option_price is None or option_price <= 0:
            return None
        if spot is None or spot <= 0 or strike is None or strike <= 0:
            return None

        today = datetime.now()
        expiry_dt = datetime.combine(expiry_date, time(15, 30))
        seconds_left = (expiry_dt - today).total_seconds()
        if seconds_left <= 60:
            return None
        t_years = seconds_left / (365.0 * 24 * 3600)

        # Sanity bound: price must exceed intrinsic value or the solver
        # will fail to converge / return garbage.
        intrinsic = max(0.0, (spot - strike) if option_type == 'c' else (strike - spot))
        if option_price <= intrinsic:
            return None

        iv = _bs_iv(option_price, spot, strike, t_years, r, option_type)
        if iv is None or iv <= 0 or iv > 5.0:   # sanity clamp: 0–500% IV
            return None
        return float(iv * 100)
    except Exception:
        return None
   
_IV_FLATLINE_WARNED = False  # module-level flag, add near other session flags

def compute_iv_zscore(iv_buf, lookback=IV_ZSCORE_LOOKBACK):
    """
    Z-score of the current IV relative to the last `lookback` bars.
    Returns 0.0 if std is near-zero (flat IV) or buffer too short.
    """
    global _IV_FLATLINE_WARNED

    if len(iv_buf) < lookback + 1:
        return 0.0

    arr  = np.array(iv_buf[-lookback:], dtype=float)
    mean = float(np.mean(arr))
    std  = float(np.std(arr))

    if std < 1e-6:
        if not _IV_FLATLINE_WARNED:
            log.warning(
                f"compute_iv_zscore: IV buffer is flat (std={std:.8f} over "
                f"last {lookback} bars, mean={mean:.2f}) — IV_ZSCORE exit "
                f"will never fire until real IV data reaches iv_buf."
            )
            _IV_FLATLINE_WARNED = True
        return 0.0

    current_iv = float(iv_buf[-1])
    return float((current_iv - mean) / std)

def compute_oi_change_pct(oi_buf):
    """
    Single-bar OI change % = (oi[-1] − oi[-2]) / oi[-2] × 100.
    Returns 0.0 if buffer too short or previous OI is zero.
    """
    if len(oi_buf) < 2:
        return 0.0
    prev = float(oi_buf[-2])
    curr = float(oi_buf[-1])
    if prev == 0:
        return 0.0
    return float((curr - prev) / prev * 100)


def compute_supertrend(high_buf, low_buf, close_buf,
                       period=ST_PERIOD, mult=ST_MULT):
    """
    Supertrend computed on OTM option-premium OHLC (not spot).
    Returns +1 (premium trending up) or -1 (premium trending down).
    Uses Wilder's ATR smoothing.
    """
    h = np.array(list(high_buf),  dtype=float)
    l = np.array(list(low_buf),   dtype=float)
    c = np.array(list(close_buf), dtype=float)
    n = len(c)

    if n < period + 2:
        return 1    # default bullish (insufficient data)

    # True Range
    tr = np.zeros(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i],
                    abs(h[i] - c[i - 1]),
                    abs(l[i] - c[i - 1]))

    # Wilder's ATR
    atr      = np.zeros(n)
    atr[period - 1] = float(np.mean(tr[:period]))
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    hl2         = (h + l) / 2.0
    basic_upper = hl2 + mult * atr
    basic_lower = hl2 - mult * atr

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    direction   = np.ones(n, dtype=int)

    for i in range(1, n):
        # Final upper band
        if basic_upper[i] < final_upper[i - 1] or c[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Final lower band
        if basic_lower[i] > final_lower[i - 1] or c[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # Direction: +1 if price above lower band, -1 if below upper band
        if direction[i - 1] == 1:
            direction[i] = 1 if c[i] >= final_lower[i] else -1
        else:
            direction[i] = -1 if c[i] <= final_upper[i] else 1

    return int(direction[-1])


# ── new params — add near other Smart Exit parameters ──
IV_SPIKE_LOOKBACK_BARS = 3    # a spike anywhere in the last N bars still counts
IV_GIVEBACK_PCT        = 2.0  # pnl must have dropped at least this much from
                               # its recent peak for the spike to be "confirmed"


def smart_exit_check(signal, entry, sl_price, bar_time,
                     bar_high, bar_low, bar_close,
                     bars_held, iv_buf, oi_buf,
                     otm_h_buf, otm_l_buf, otm_c_buf):
    """
    5-priority exit chain. Runs every bar while a trade is active.
    Returns (exit_flag: bool, reason: str, exec_price: float).

    Priority order (strict):
      1. Hard SL      — bar_low based (worst-case fill)
      2. IV Z-Score   — bar_close (IV spiking, risk of crush)
      3. OI Unwind    — bar_close (institutions exiting)
      4. Supertrend   — bar_close (trend structure broken)
      5. Time Stop    — bar_close (timeout or EOD)

    LONG and SHORT both hold long option premium, so an adverse supertrend
    flip is bearish (-1) for either signal.
    """
    pnl_pct = (bar_close - entry) / entry * 100

    # ── 1. Hard SL ───────────────────────────────────────────────────
    if bar_low <= sl_price:
        return True, "SL", sl_price

    # ── 2. IV Z-Score Spike (spike-then-giveback confirmation) ─────────
    # Instead of requiring iv_z > threshold on two CONSECUTIVE bars (rare —
    # z-score spikes are usually one-bar events, which is why this almost
    # never fired), we ask a more meaningful question: did IV spike recently
    # (within IV_SPIKE_LOOKBACK_BARS), AND has price already started giving
    # back some of the peak gain since then? The giveback condition is the
    # real confirmation — it's evidence the spike is translating into an
    # actual reversal, not just noise in the z-score calc.
    iv_list = list(iv_buf)
    c_list  = list(otm_c_buf)
    n_look  = min(IV_SPIKE_LOOKBACK_BARS, bars_held, len(iv_list) - 1, len(c_list) - 1)

    spike_recent = False
    peak_since_spike = bar_close
    if n_look > 0:
        for k in range(n_look + 1):
            # iv_z computed as of (bars_held - k) bars ago
            cut = len(iv_list) - k
            if cut < IV_ZSCORE_LOOKBACK + 1:
                continue
            iv_z_k = compute_iv_zscore(iv_list[:cut], IV_ZSCORE_LOOKBACK)
            pnl_k = (c_list[-(k + 1)] - entry) / entry * 100
            if iv_z_k > IV_ZSCORE_EXIT and pnl_k >= IV_MIN_PROFIT_PCT:
                spike_recent = True
                peak_since_spike = max(peak_since_spike, c_list[-(k + 1)])

    peak_pnl_since_spike = (peak_since_spike - entry) / entry * 100
    giveback_confirmed = (peak_pnl_since_spike - pnl_pct) >= IV_GIVEBACK_PCT

    if (spike_recent and giveback_confirmed and pnl_pct >= IV_MIN_PROFIT_PCT
            and bars_held >= MIN_BARS_PROFIT_EXIT):
        return True, "IV_ZSCORE", bar_close

    # ── 3. OI Unwind (v14 fix: min-hold) ────────────────────────────
    oi_chg = compute_oi_change_pct(list(oi_buf))
    if oi_chg < OI_DROP_THRESH and pnl_pct >= OI_MIN_PROFIT_PCT and bars_held >= MIN_BARS_PROFIT_EXIT:
        return True, "OI_UNWIND", bar_close

    # ── 4. Supertrend Flip (v14 fix: min-hold + 2-bar confirmation) ──
    # Both directions hold long premium: exit when the traded option's
    # supertrend goes bearish (-1).
    if len(otm_c_buf) >= ST_PERIOD + 2:
        h_list = list(otm_h_buf)
        l_list = list(otm_l_buf)
        c_list = list(otm_c_buf)
        st_dir = compute_supertrend(h_list, l_list, c_list,
                                    period=ST_PERIOD, mult=ST_MULT)
        _st_now = (st_dir == -1) and pnl_pct >= ST_MIN_PROFIT_PCT
        _st_prev = False
        if len(c_list) >= ST_PERIOD + 3:
            try:
                # otm_h_buf / otm_l_buf / otm_c_buf are deques (buffer["otm_call_high"]
                # etc., created with deque(maxlen=BUFFER_SIZE)). Deques do NOT support
                # slicing — `otm_h_buf[:-1]` raises TypeError. Must convert to list
                # first. Previously this silently threw on every call, was swallowed
                # by the bare except below, and permanently forced _st_prev = False —
                # meaning the SUPERTREND exit could NEVER fire, no matter how long the
                # trend stayed flipped or how much profit was on the table.
                st_dir_prev = compute_supertrend(h_list[:-1], l_list[:-1], c_list[:-1],
                                                 period=ST_PERIOD, mult=ST_MULT)
                _st_prev = (st_dir_prev == -1)
            except Exception:
                log.warning(
                    "smart_exit_check: SUPERTREND prev-bar confirmation failed, "
                    "defaulting to no-exit", exc_info=True
                )
                _st_prev = False
        if _st_now and _st_prev and bars_held >= MIN_BARS_PROFIT_EXIT:
            return True, "SUPERTREND", bar_close

    # ── 5. Time Stops ────────────────────────────────────────────────
    t = bar_time.time()
    if bars_held >= MAX_BARS_HELD:
        return True, "TIMEOUT", bar_close
    if t >= time(15, 30):
        return True, "EOD", bar_close

    return False, None, None

# ══════════════════════════════════════════════════════════════════════
# ATR COMPUTATION (of option premium)
# ══════════════════════════════════════════════════════════════════════
def compute_option_atr(high_buf, low_buf, close_buf, period=ATR_PERIOD):
    h = np.array(high_buf)
    l = np.array(low_buf)
    c = np.array(close_buf)

    if len(c) < period + 1:
        return None

    h = h[-(period + 1):]
    l = l[-(period + 1):]
    c = c[-(period + 1):]

    tr = np.maximum.reduce([
        h[1:] - l[1:],
        np.abs(h[1:] - c[:-1]),
        np.abs(l[1:] - c[:-1]),
    ])

    atr = tr[0]
    for i in range(1, len(tr)):
        atr = (atr * (period - 1) + tr[i]) / period

    return float(atr)


# ══════════════════════════════════════════════════════════════════════
# SL COMPUTATION  (TP removed in v14 — smart_exit_check handles exits)
# ══════════════════════════════════════════════════════════════════════
def compute_sl(entry, atr_val=None, iv_current=None, iv_mean=None):
    """
    v14.1: static point-based SL, replacing ATR/%-based sizing.
    SL = entry - SL_POINTS, flat regardless of ATR, IV, or premium size.
    atr_val/iv_current/iv_mean kept in the signature for call-site
    compatibility but no longer influence the SL distance.

    NOTE: for premiums <= SL_POINTS, this SL never triggers before the
    premium bottoms out — the trade's real max loss is capped by the
    premium itself, not by this stop.
    """
    sl_price = max(SL_MIN_PRICE_FLOOR, entry - SL_POINTS)
    method = f"STATIC_{SL_POINTS:.0f}PTS"
    return sl_price, method


# ══════════════════════════════════════════════════════════════════════
# STRIKE RANGE & OTM VALIDATORS
# ══════════════════════════════════════════════════════════════════════
def get_atm_strike(spot_price, step=STRIKE_STEP):
    return round(spot_price / step) * step


def is_strike_in_range(spot_price, otm_offset, option_type, step=STRIKE_STEP):
    atm = get_atm_strike(spot_price, step)
    if option_type == "LONG":
        strike = atm + otm_offset * step
    else:
        strike = atm - otm_offset * step
    in_range = abs(strike - atm) <= STRIKE_RANGE_PTS
    return in_range, atm, strike


def is_strike_otm(spot_price, exec_strike, option_type):
    if option_type == "LONG":
        if exec_strike > spot_price:
            return True, f"call_OTM(strike={exec_strike:.0f}>spot={spot_price:.0f})"
        else:
            return False, f"call_ITM_BLOCKED(strike={exec_strike:.0f}<=spot={spot_price:.0f})"
    else:
        if exec_strike < spot_price:
            return True, f"put_OTM(strike={exec_strike:.0f}<spot={spot_price:.0f})"
        else:
            return False, f"put_ITM_BLOCKED(strike={exec_strike:.0f}>=spot={spot_price:.0f})"


# ══════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING  (unchanged from v13)
# ══════════════════════════════════════════════════════════════════════
def ema_arr(s, n):
    out = np.zeros(len(s))
    out[0] = s[0]
    k = 2 / (n + 1)
    for i in range(1, len(s)):
        out[i] = s[i] * k + out[i - 1] * (1 - k)
    return out


def compute_rsi_arr(s, p=14):
    d  = np.diff(s)
    g  = np.where(d > 0, d, 0)
    l  = np.where(d < 0, -d, 0)
    ag = np.convolve(g, np.ones(p) / p, mode="valid")
    al = np.convolve(l, np.ones(p) / p, mode="valid")
    rs = ag / (al + 1e-9)
    return float((100 - 100 / (1 + rs))[-1]) if len(rs) else 50.0


def compute_adx_arr(c, p=14):
    if len(c) < p + 2:
        return 25.0, 50.0, 50.0
    tr  = np.abs(np.diff(c))
    pd_ = np.clip(np.diff(c), 0, None)
    nd_ = np.clip(-np.diff(c), 0, None)
    atr = np.convolve(tr,  np.ones(p) / p, mode="valid")
    pdi = np.convolve(pd_, np.ones(p) / p, mode="valid") / (atr + 1e-9) * 100
    ndi = np.convolve(nd_, np.ones(p) / p, mode="valid") / (atr + 1e-9) * 100
    dx  = np.abs(pdi - ndi) / (pdi + ndi + 1e-9) * 100
    adx = np.convolve(dx, np.ones(p) / p, mode="valid")
    return float(adx[-1]), float(pdi[-1]), float(ndi[-1])


def compute_bb_arr(s, w=20):
    if len(s) < w:
        return 0.5, 0.02
    mid   = np.mean(s[-w:])
    std   = np.std(s[-w:])
    upper = mid + 2 * std
    lower = mid - 2 * std
    return (
        float((s[-1] - lower) / (upper - lower + 1e-9)),
        float((upper - lower) / (mid + 1e-9)),
    )


def htf(close_1m, tf):
    if len(close_1m) < tf * 2:
        return dict(rsi=50, adx=25, pdi=50, ndi=50,
                    bb_pctb=0.5, bb_bw=0.02, above_ema200=0, macdh=0.0)
    r = np.array(close_1m)[tf - 1::tf]
    if len(r) < 2:
        return dict(rsi=50, adx=25, pdi=50, ndi=50,
                    bb_pctb=0.5, bb_bw=0.02, above_ema200=0, macdh=0.0)
    ema200         = ema_arr(r, min(200, len(r)))[-1]
    ef             = ema_arr(r, 12)
    es             = ema_arr(r, 26)
    macd           = ef - es
    sig            = ema_arr(macd, min(9, len(macd)))
    adx, pdi, ndi  = compute_adx_arr(r, 14)
    bb_pctb, bb_bw = compute_bb_arr(r, 20)
    return dict(
        rsi          = compute_rsi_arr(r, 14),
        adx          = adx,
        pdi          = pdi,
        ndi          = ndi,
        bb_pctb      = bb_pctb,
        bb_bw        = bb_bw,
        above_ema200 = int(r[-1] > ema200),
        macdh        = float(macd[-1] - sig[-1]) if len(macd) > 9 else 0.0,
    )


# ══════════════════════════════════════════════════════════════════════
# EXTRA INDICATOR HELPERS (new — needed for the expanded feature set)
# ══════════════════════════════════════════════════════════════════════
def compute_macd_hist_arr(c, fast=12, slow=26, sig=9):
    if len(c) < slow + sig:
        return 0.0, 0.0
    ef = ema_arr(c, fast)
    es = ema_arr(c, slow)
    macd = ef - es
    sigl = ema_arr(macd, sig)
    hist = macd - sigl
    slope = float(hist[-1] - hist[-2]) if len(hist) > 1 else 0.0
    return float(hist[-1]), slope


def compute_stoch_arr(h, l, c, k_period=14, d_period=3):
    if len(c) < k_period + d_period:
        return 50.0, 50.0
    hh = np.array([max(h[max(0, i - k_period + 1): i + 1]) for i in range(len(h))])
    ll = np.array([min(l[max(0, i - k_period + 1): i + 1]) for i in range(len(l))])
    k = 100 * (c - ll) / (hh - ll + 1e-9)
    d = np.convolve(k, np.ones(d_period) / d_period, mode="valid")
    return float(k[-1]), float(d[-1]) if len(d) else float(k[-1])


def compute_cmf_arr(h, l, c, v, period=20):
    if len(c) < period + 1:
        return 0.0
    mfm = ((c - l) - (h - c)) / (h - l + 1e-9)
    mfv = mfm * v
    cmf = np.sum(mfv[-period:]) / (np.sum(v[-period:]) + 1e-9)
    return float(cmf)


def compute_atr_pct_arr(h, l, c, period=14):
    atr = compute_option_atr(h, l, c, period)
    if atr is None or c[-1] == 0:
        return 0.0
    return float(atr / c[-1] * 100)


def rolling_rank(arr, window=FULL_WINDOW):
    if len(arr) < 2:
        return 0.5
    w = arr[-window:] if len(arr) > window else arr
    return float((np.sum(w <= w[-1]) - 1) / max(len(w) - 1, 1))


def compute_pivots(prev_high, prev_low, prev_close):
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    return pivot, r1, s1


def compute_max_pain_pct(chain_snap, spot, step=STRIKE_STEP):
    """Approx max-pain strike vs spot, as % distance."""
    strikes = sorted(chain_snap.keys())
    pains = {}
    for s in strikes:
        strike_price = spot + s * step  # placeholder ATM ref, replaced by caller with true ATM
        pain = 0.0
        for off in strikes:
            k = spot + off * step
            call_oi = chain_snap[off]["CALL"]["oi"]
            put_oi = chain_snap[off]["PUT"]["oi"]
            pain += call_oi * max(0, strike_price - k) + put_oi * max(0, k - strike_price)
        pains[s] = pain
    if not pains:
        return 0.0
    max_pain_off = min(pains, key=pains.get)
    max_pain_strike = spot + max_pain_off * step
    return float((max_pain_strike - spot) / (spot + 1e-9) * 100)


# ══════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING  — v15 (full 245-feature schema)
# NOTE: legacy path. The live bar loop uses v12_inference._build_features
# (training-exact); this function is kept for reference only.
# ══════════════════════════════════════════════════════════════════════
def build_features(buf, chain_snap=None, india_vix=16.0):
    """
    chain_snap: dict {offset(-10..+10): {"CALL":{...}, "PUT":{...}}}
                for the CURRENT bar (from chain_buf[-1]).
    """
    if len(buf["call_close"]) < 50:
        return None

    c   = np.array(buf["call_close"]);  ch = np.array(buf["call_high"]); cl = np.array(buf["call_low"])
    cv  = np.array(buf["call_volume"]); pv = np.array(buf["put_volume"])
    coi = np.array(buf["call_oi"]);     poi = np.array(buf["put_oi"])
    civ = np.array(buf["call_iv"]);     piv = np.array(buf["put_iv"])
    pc  = np.array(buf["put_close"])
    spot = np.array(buf["spot_close"])
    vix_buf = np.array(buf["india_vix"])

    now = buf["datetime"][-1]
    t   = now.time()
    dow = now.weekday()

    row = {}

    # ── 1. WEEK STRIKE GRID (ATM-10 .. ATM+10, CALL/PUT) ───────────────
    chain_snap = chain_snap or _empty_chain_snapshot()
    for off in range(-10, 11):
        tag = "ATM" if off == 0 else (f"ATMplus{off}" if off > 0 else f"ATMminus{abs(off)}")
        leg = chain_snap.get(off, {"CALL": {}, "PUT": {}})
        for side in ("CALL", "PUT"):
            d = leg.get(side, {})
            prefix = f"nifty_WEEK_{tag}_{side}_1m__"
            row[prefix + "open"]   = float(d.get("open", 0.0))
            row[prefix + "volume"] = float(d.get("volume", 0))
            row[prefix + "iv"]     = float(d.get("iv", 0.0))
            row[prefix + "oi"]     = float(d.get("oi", 0))
            if off in (0, 10) and side == "CALL":     # spot only kept for ATM & ATM+10 CALL, per your list
                row[prefix + "spot"] = float(d.get("spot", spot[-1] if len(spot) else 0.0))
            elif off == 0 and side == "PUT":
                pass  # not in list
    # ATM+10 PUT / others not requesting spot are already covered above

    # ── 2. RVOL FAMILY ──────────────────────────────────────────────────
    vol   = float(cv[-1] + pv[-1])
    vol20 = float(np.mean((cv + pv)[-20:]) + 1e-9)
    rvol  = vol / vol20
    rvol_series = (cv + pv) / (np.convolve(cv + pv, np.ones(20) / 20, mode="same") + 1e-9)
    row["rvol"]            = rvol
    row["U02_RVOL"]        = rvol
    row["rvol_roll3_max"]  = float(np.max(rvol_series[-3:])) if len(rvol_series) >= 3 else rvol
    row["rvol_pct_rank"]   = rolling_rank(rvol_series, 100)
    row["rvol_bucket"]     = int(np.searchsorted([0.8, 1.2, 1.8, 2.5], rvol))

    # ── 3. RETURNS / MOMENTUM ───────────────────────────────────────────
    ret1  = float((c[-1] - c[-2])  / (c[-2]  + 1e-9)) if len(c) > 1  else 0.0
    ret5  = float((c[-1] - c[-6])  / (c[-6]  + 1e-9)) if len(c) > 5  else 0.0
    ret10 = float((c[-1] - c[-11]) / (c[-11] + 1e-9)) if len(c) > 10 else 0.0
    ret15 = float((c[-1] - c[-16]) / (c[-16] + 1e-9)) if len(c) > 15 else 0.0
    row["returns_1"]  = ret1
    row["returns_5"]  = ret5
    row["returns_15"] = ret15
    row["ret_1"]  = ret1
    row["ret_5"]  = ret5
    row["ret_10"] = ret10
    row["momentum_5"]  = ret5
    row["momentum_10"] = ret10

    # ── 4. 1m INDICATORS ────────────────────────────────────────────────
    rsi1m               = compute_rsi_arr(c, 14)
    rsi1m_prev          = compute_rsi_arr(c[:-1], 14) if len(c) > 15 else rsi1m
    adx1m, pdi1m, ndi1m = compute_adx_arr(c, 14)
    macdh, macdh_slope  = compute_macd_hist_arr(c)
    bb_pctb, bb_bw      = compute_bb_arr(c, 20)
    stoch_k, stoch_d     = compute_stoch_arr(ch, cl, c)
    ema200               = ema_arr(c, min(200, len(c)))[-1]
    atr1m                = compute_option_atr(ch, cl, c, ATR_PERIOD) or 0.0
    cmf1m                = compute_cmf_arr(ch, cl, c, cv, 20)

    row["rsi"]       = rsi1m
    row["rsi_slope"] = float(rsi1m - rsi1m_prev)
    row["macdh"]        = macdh
    row["macd_slope"]   = macdh_slope
    row["adx_1m"] = adx1m
    row["pdi_1m"] = pdi1m
    row["ndi_1m"] = ndi1m
    row["adx_regime"]   = int(np.searchsorted([18, 25, 35], adx1m))
    row["adx_1m_rank"]  = rolling_rank(np.array([adx1m] * len(c)), 100)
    row["atr_pct_1m"]   = float(atr1m / (c[-1] + 1e-9) * 100)
    row["bb_pctb"] = bb_pctb
    row["bb_bw"]   = bb_bw
    row["bb_squeeze"] = int(bb_bw < 0.03)
    row["stoch_k"] = stoch_k
    row["stoch_d"] = stoch_d
    row["above_ema200_1m"]     = int(c[-1] > ema200)
    row["price_position_rank"] = rolling_rank(c, 100)
    row["cmf_1m"] = cmf1m
    row["U05_CMF_Direction"] = int(cmf1m > 0)

    # ── 5. IV / OI / PCR ─────────────────────────────────────────────────
    iv_skew = float(piv[-1] - civ[-1])
    iv_skew_prev = float(piv[-6] - civ[-6]) if len(piv) > 6 else iv_skew
    row["iv_skew"]          = iv_skew
    row["iv_skew_slope"]    = iv_skew - iv_skew_prev
    row["iv_momentum_5bar"] = float(civ[-1] - civ[-6]) if len(civ) > 6 else 0.0
    row["iv_skew_ratio"]    = float(piv[-1] / (civ[-1] + 1e-9))
    row["iv_rank"]          = rolling_rank(civ + piv, 100)
    row["atm_call_iv_rank"] = rolling_rank(civ, 100)
    row["atm_put_iv_rank"]  = rolling_rank(piv, 100)

    pcr_vol = float(pv[-1] / (cv[-1] + 1e-9))
    pcr_oi  = float(poi[-1] / (coi[-1] + 1e-9))
    pcr_oi_l1 = float(poi[-2] / (coi[-2] + 1e-9)) if len(poi) > 1 else pcr_oi
    pcr_oi_l2 = float(poi[-3] / (coi[-3] + 1e-9)) if len(poi) > 2 else pcr_oi
    row["pcr_vol"] = pcr_vol
    row["pcr_oi"]  = pcr_oi
    row["pcr_oi_lag1"] = pcr_oi_l1
    row["pcr_oi_lag2"] = pcr_oi_l2
    row["pcr_oi_slope"] = pcr_oi - pcr_oi_l1

    call_oi_chg = float((coi[-1] - coi[-2]) / (coi[-2] + 1e-9)) if len(coi) > 1 else 0.0
    put_oi_chg  = float((poi[-1] - poi[-2]) / (poi[-2] + 1e-9)) if len(poi) > 1 else 0.0
    row["call_oi_chg_norm"] = call_oi_chg
    row["put_oi_chg_norm"]  = put_oi_chg
    row["net_oi_chg_norm"]  = put_oi_chg - call_oi_chg
    row["oi_buildup"]       = int((call_oi_chg > 0.02 and ret1 > 0) or (put_oi_chg > 0.02 and ret1 < 0))
    row["atm_oi_concentration"] = float((coi[-1] + poi[-1]) / (np.sum(coi[-20:]) + np.sum(poi[-20:]) + 1e-9))
    row["rvol_x_pcr"]       = rvol * pcr_oi
    row["max_pain_pct"]     = compute_max_pain_pct(chain_snap, spot[-1] if len(spot) else 0.0)

    # ── 6. CANDLE / PRICE PATTERNS ───────────────────────────────────────
    body = abs(c[-1] - buf["call_open"][-1])
    rng  = (ch[-1] - cl[-1]) + 1e-9
    row["candle_body_ratio"] = float(body / rng)
    row["P01_Strong_Candle"] = int(body / rng > 0.6)
    row["P04_HH_HL_Pattern"] = int(c[-1] > c[-2] and cl[-1] > cl[-2]) if len(c) > 2 else 0
    row["P02_Close_Position"] = bb_pctb

    spot_last = float(spot[-1]) if len(spot) else 0.0
    row["spot_mod_100"] = float(spot_last % 100)
    row["spot_mod_50"]  = float(spot_last % 50)

    if len(spot) > 375:
        prev_day = spot[-375:-1]
        piv_p, r1, s1 = compute_pivots(float(np.max(prev_day)), float(np.min(prev_day)), float(prev_day[-1]))
    else:
        piv_p, r1, s1 = spot_last, spot_last, spot_last
    row["near_pivot"] = int(abs(spot_last - piv_p) < 15)
    row["near_r1"]    = int(abs(spot_last - r1) < 15)
    row["near_s1"]    = int(abs(spot_last - s1) < 15)
    row["dist_to_pivot_pct"] = float((spot_last - piv_p) / (piv_p + 1e-9) * 100)

    # ── 7. VIX / VOLATILITY REGIME ───────────────────────────────────────
    vix_ma20 = float(np.mean(vix_buf[-FULL_WINDOW:])) if len(vix_buf) >= 100 else india_vix
    row["vix_rank"]                = min(india_vix / 30.0, 1.0)
    row["vix_ma20"]                = vix_ma20
    row["vix_above_ma"]            = int(india_vix > vix_ma20)
    row["vix_above_rolling_median"]= int(india_vix > float(np.median(vix_buf[-FULL_WINDOW:]))) if len(vix_buf) >= 100 else 0
    row["vol_regime"]              = int(rvol > 1.5)
    row["is_high_vol_regime_v2"]   = int(india_vix > 20 or rvol > 2.0)
    row["is_crisis_regime"]        = int(india_vix > 28)
    row["trend_regime"]            = int(adx1m > 25)
    row["U08_Volume_Surge_Dir"]    = int(np.sign(ret1)) if rvol > 1.5 else 0

    # ── 8. SESSION / TIME ────────────────────────────────────────────────
    def sess(t):
        if t < time(9, 30):  return 0
        if t < time(11, 30): return 1
        if t < time(13, 30): return 2
        return 3

    delta = (1 - dow) % 7
    if delta == 0 and t > time(15, 30):
        delta = 7

    row["hour"]         = now.hour
    row["minute"]       = now.minute
    row["dayofweek"]    = dow
    row["session"]      = sess(t)
    row["is_expiry"]    = int(dow == 1)
    row["is_pre_close"] = int(t >= time(14, 45))
    row["is_open_rush"] = int(time(9, 15) <= t <= time(9, 44))
    row["days_to_expiry"] = delta

    # ── 9. MULTI-TIMEFRAME (5m / 15m / 30m) ─────────────────────────────
    def build_tf(tf):
        if len(c) < tf * 2:
            return dict(rsi=50, macdh=0, adx=25, pdi=50, ndi=50, atr=0, atr_pct=0,
                        bb_pctb=0.5, bb_bw=0.02, cmf=0, ema_gap=0, st_dir=1,
                        trend=0, adx_regime=0, adx_rank=0.5, di_cross=0)
        r  = c[tf - 1::tf]
        rh = ch[tf - 1::tf]
        rl = cl[tf - 1::tf]
        rv = cv[tf - 1::tf] + pv[tf - 1::tf]
        if len(r) < 20:
            return dict(rsi=50, macdh=0, adx=25, pdi=50, ndi=50, atr=0, atr_pct=0,
                        bb_pctb=0.5, bb_bw=0.02, cmf=0, ema_gap=0, st_dir=1,
                        trend=0, adx_regime=0, adx_rank=0.5, di_cross=0)
        adx_, pdi_, ndi_ = compute_adx_arr(r, 14)
        macdh_, _        = compute_macd_hist_arr(r)
        bb_pctb_, bb_bw_ = compute_bb_arr(r, 20)
        atr_             = compute_option_atr(rh, rl, r, 14) or 0.0
        ema200_          = ema_arr(r, min(200, len(r)))[-1]
        cmf_             = compute_cmf_arr(rh, rl, r, rv, 20)
        st_dir_          = compute_supertrend(rh, rl, r, period=10, mult=3.0)
        pdi_prev, ndi_prev = pdi_, ndi_
        return dict(
            rsi=compute_rsi_arr(r, 14), macdh=macdh_, adx=adx_, pdi=pdi_, ndi=ndi_,
            atr=atr_, atr_pct=float(atr_ / (r[-1] + 1e-9) * 100),
            bb_pctb=bb_pctb_, bb_bw=bb_bw_, cmf=cmf_,
            ema_gap=float((r[-1] - ema200_) / (ema200_ + 1e-9) * 100),
            st_dir=st_dir_, trend=int(adx_ > 25),
            adx_regime=int(np.searchsorted([18, 25, 35], adx_)),
            adx_rank=rolling_rank(np.array([adx_] * len(r)), 100),
            di_cross=int(pdi_ > ndi_) - int(pdi_prev < ndi_prev),
        )

    f5, f15, f30 = build_tf(5), build_tf(15), build_tf(30)

    row.update({
        "rsi_5m": f5["rsi"], "macdh_5m": f5["macdh"], "adx_5m": f5["adx"],
        "pdi_5m": f5["pdi"], "ndi_5m": f5["ndi"], "atr_5m": f5["atr"],
        "atr_pct_5m": f5["atr_pct"], "bb_pctb_5m": f5["bb_pctb"], "bb_bw_5m": f5["bb_bw"],
        "cmf_5m": f5["cmf"], "ema200_gap_pct_5m": f5["ema_gap"], "st_dir_5m": f5["st_dir"],
        "trend_5m": f5["trend"], "adx_regime_5m": f5["adx_regime"], "adx_5m_rank": f5["adx_rank"],
        "di_cross_5m": f5["di_cross"],

        "rsi_15m": f15["rsi"], "macdh_15m": f15["macdh"], "adx_15m": f15["adx"],
        "pdi_15m": f15["pdi"], "ndi_15m": f15["ndi"], "atr_15m": f15["atr"],
        "atr_pct_15m": f15["atr_pct"], "bb_pctb_15m": f15["bb_pctb"], "bb_bw_15m": f15["bb_bw"],
        "cmf_15m": f15["cmf"], "ema200_gap_pct_15m": f15["ema_gap"], "st_dir_15m": f15["st_dir"],
        "trend_15m": f15["trend"], "adx_regime_15m": f15["adx_regime"], "adx_15m_rank": f15["adx_rank"],
        "di_cross_15m": f15["di_cross"],

        "rsi_30m": f30["rsi"], "macdh_30m": f30["macdh"], "adx_30m": f30["adx"],
        "pdi_30m": f30["pdi"], "ndi_30m": f30["ndi"], "atr_30m": f30["atr"],
        "atr_pct_30m": f30["atr_pct"], "bb_pctb_30m": f30["bb_pctb"], "bb_bw_30m": f30["bb_bw"],
        "cmf_30m": f30["cmf"], "st_dir_30m": f30["st_dir"],
        "trend_30m": f30["trend"], "adx_regime_30m": f30["adx_regime"], "adx_30m_rank": f30["adx_rank"],
        "di_cross_30m": f30["di_cross"],
    })

    # ── 10. MTF ALIGNMENT ────────────────────────────────────────────────
    row["mtf_aligned_bull"] = int(f5["rsi"] > 50 and f15["rsi"] > 50 and f15["macdh"] > 0)
    row["mtf_aligned_bear"] = int(f5["rsi"] < 50 and f15["rsi"] < 50 and f15["macdh"] < 0)
    row["mtf_bull_soft"]    = int((f5["rsi"] > 50) + (f15["rsi"] > 50) + (f15["macdh"] > 0) >= 2)

    # ── 11. Backfill anything the model still expects but we haven't set ─
    for feat in model.features:
        if feat not in row:
            row[feat] = 0.0

    return row

# ══════════════════════════════════════════════════════════════════════
# RULE-BASED SIGNAL FILTER  (rule-only pinned series is isolated from ML)
# ══════════════════════════════════════════════════════════════════════
def _compute_rule_pin_5m():
    """Calculate 5m RSI/MACD histogram using pinned CALL closes only.

    This deliberately follows the existing feature pipeline's stride-based
    1m-to-5m sampling convention so the gate's inputs remain methodologically
    consistent while changing only the underlying strike series.
    """
    if RULE_PIN_STRIKE is None or RULE_PIN_SECURITY_ID is None:
        return None, None
    if len(RULE_PIN_CLOSES) < RULE_PIN_MIN_ROWS:
        return None, None

    close = np.asarray(RULE_PIN_CLOSES, dtype=float)
    close = close[np.isfinite(close) & (close > 0)]
    if len(close) < RULE_PIN_MIN_ROWS:
        return None, None

    close_5m = close[4::5]
    if len(close_5m) < 35:
        return None, None

    rsi5m = compute_rsi_arr(close_5m, 14)
    macdh5m, _ = compute_macd_hist_arr(close_5m)
    return float(rsi5m), float(macdh5m)


def apply_rule_filters(
    signal,
    feat,
    bartime,
    lastexitreason,
    lastsignaldir,
    barssinceexit,
    confidence,
    spotbuffer=None,
    rule_rsi5m=None,
    rule_macdh5m=None,
):
    t    = bartime.time()
    adx  = feat.get("adx_1m", 0)
    rsi  = feat.get("rsi_1m", 50)
    rvol = feat.get("rvol", 1.0)

    if confidence < MINCONF:
        return "NONE", f"lowconf{confidence:.2f}<{MINCONF}"

    if t < time(9, 45):
        return "NONE", "openrush"

    # ✅ CHANGED: > instead of >= so 15:14:00 bar is now allowed through
    if t > time(15, 14):
        return "NONE", "preclose"

    # ✅ CHANGED: <= instead of < so 12:15:00 bar is also blocked
    if time(12, 0) <= t <= time(12, 15):
        return "NONE", "lunchchop"

    if adx < MINADX:
        return "NONE", f"lowadx{adx:.1f}<{MINADX}"

    if rvol < MINRVOL:
        return "NONE", f"lowrvol{rvol:.2f}<{MINRVOL}"

    if rule_rsi5m is None or rule_macdh5m is None:
        return "NONE", "rulepinwarmup"

    rsi5m = float(rule_rsi5m)
    macdh5m = float(rule_macdh5m)

    # ✅ CHANGED: double-condition AND logic replaces trend5m enum
    # LONG blocked only if rsi5m < 45 AND macdh < 0  (was: < 50 AND < 0)
    # SHORT blocked only if rsi5m > 55 AND macdh > 0  (was: > 50 AND > 0)
    if signal == "LONG"  and rsi5m < 45 and macdh5m < 0:
        return "NONE", "5mbeartrend"

    if signal == "SHORT" and rsi5m > 55 and macdh5m > 0:
        return "NONE", "5mbulltrend"

    # ⚠️ Reversal guard (v14 fix): don't enter against a just-started underlying
    # reversal. The L3 model fires on regime/timing that can lag the move — this
    # blocks a SHORT when the underlying has already turned up, and a LONG when it
    # has turned down, over the last REVERSAL_GUARD_BARS.
    if REVERSAL_GUARD_ENABLED and spotbuffer is not None and len(spotbuffer) >= REVERSAL_GUARD_BARS + 1:
        try:
            recent = [float(x) for x in spotbuffer[-(REVERSAL_GUARD_BARS + 1):]]
            chg = (recent[-1] - recent[0]) / (recent[0] + 1e-9) * 100.0
            if signal == "SHORT" and chg > REVERSAL_GUARD_PCT:
                return "NONE", f"reversalup{chg:.2f}"
            if signal == "LONG" and chg < -REVERSAL_GUARD_PCT:
                return "NONE", f"reversaldown{chg:.2f}"
        except (TypeError, ValueError):
            pass

    # ✅ CHANGED: thresholds relaxed — 78/22 instead of 75/25
    if signal == "LONG"  and rsi > 78:
        return "NONE", f"rsioverbought{rsi:.1f}"

    if signal == "SHORT" and rsi < 22:
        return "NONE", f"rsioversold{rsi:.1f}"

    if (
        lastexitreason == "SL"
        and signal == lastsignaldir
        and barssinceexit < SLREENTRYBARS
    ):
        return "NONE", f"slreentryblock{barssinceexit}<{SLREENTRYBARS}"

    return signal, "OK"


def feed_resubscribe(new_token: str, new_symbol: str,
                     old_token: str = None, old_symbol: str = None,
                     direction: str = ""):
    """
    PATCH 7: grid legs (ChainGrid) are permanent subscriptions. The default
    OTM contract (ATM±50) is ALSO a grid leg, so the entry/exit swap must
    never subscribe a token that is already live or unsubscribe one the grid
    still needs -- otherwise that grid leg silently freezes.
    """
    global FEED

    if FEED is None:
        log.warning("feed_resubscribe: FEED is None, skipping dynamic subscribe")
        return

    new_token = str(new_token).strip() if new_token is not None else ""
    old_token = str(old_token).strip() if old_token is not None else ""

    if not new_token:
        log.warning(f"feed_resubscribe: empty new_token for {new_symbol}, direction={direction}")
        return

    _grid_ids = GRID.ids if GRID is not None else set()
    segment = mf.MarketFeed.NSE_FNO
    mode    = mf.MarketFeed.Full

    # ── subscribe new — dhanhq API requires a LIST of tuples, not positional args ──
    if new_token in _grid_ids:
        log.info(f"FEED subscribe skipped (already live as grid leg) → "
                 f"{new_symbol} token={new_token} ({direction})")
    else:
        try:
            FEED.subscribe_symbols([(segment, new_token, mode)])
            log.info(f"FEED subscribed   → {new_symbol} token={new_token} ({direction})")
        except Exception as e:
            log.error(f"FEED subscribe FAILED for {new_symbol} token={new_token}: {e}")

    # ── unsubscribe old only if genuinely different AND not a grid leg ──
    try:
        if old_token and old_token != new_token and old_token not in _grid_ids:
            FEED.unsubscribe_symbols([(segment, old_token, mode)])
            log.info(f"FEED unsubscribed ← {old_symbol} token={old_token} ({direction})")
        elif old_token in _grid_ids:
            log.info(f"FEED unsubscribe skipped (grid leg kept live) "
                     f"old_token={old_token} ({direction})")
        else:
            log.info(
                f"FEED unsubscribe skipped old_token={old_token} "
                f"new_token={new_token} ({direction})"
            )
    except Exception as e:
        log.error(f"FEED unsubscribe FAILED for {old_symbol} token={old_token}: {e}")


def _grid_recenter():
    """Shift the subscribed band when ATM drifts. Runs in process_tick, OUTSIDE
    asyncio.run(on_bar_complete) -- FEED.subscribe_symbols is sync and must not be
    called from inside a running event loop."""
    if GRID is None or FEED is None:
        return
    protect = set()
    if active_trade and active_trade.get("strike"):
        protect.add((int(active_trade["strike"]), active_trade["side"]))
    if RULE_PIN_STRIKE is not None:
        protect.add((int(RULE_PIN_STRIKE), "CALL"))
    try:
        added, dropped = GRID.maybe_recenter(protect)
        if not added and not dropped:
            return
        GRID.seed_from_rest(CLIENT_ID, ACCESS_TOKEN, ids=added)
        seg, mode = mf.MarketFeed.NSE_FNO, mf.MarketFeed.Full
        for i in range(0, len(added), 50):
            FEED.subscribe_symbols([(seg, s, mode) for s in added[i:i+50]])
        for i in range(0, len(dropped), 50):
            FEED.unsubscribe_symbols([(seg, s, mode) for s in dropped[i:i+50]])
    except Exception as e:
        log.error(f"_grid_recenter failed: {e}", exc_info=True)
       
# ══════════════════════════════════════════════════════════════════════
# BAR CLOSE HANDLER  (v14 — ZERO-BAR fix applied)
# ══════════════════════════════════════════════════════════════════════

_bar_processing = False
async def on_bar_complete(bar_time, india_vix=16.0, _mapper=None):
    global active_trade, _bar_count, bar_acc
    global _cooldown_remaining, _last_exit_bar
    global _last_exit_reason, _last_signal_dir
    global order_manager, kite, expiry, syncer
    global INSTRUMENTS, _bar_processing
    global _ACTIVE_TRADE_FEED_ID, _ACTIVE_TRADE_FEED_SYMBOL
    global _LAST_TRADE_PRICE, _LAST_TRADE_TICK_TS
    global _last_cum_call, _last_cum_put

    _DEFAULT_OTM_CALL = INSTRUMENTS["otm_call"]
    _DEFAULT_OTM_PUT = INSTRUMENTS["otm_put"]

    if _bar_processing:
        log.warning(f"BAR {bar_time} skipped — previous bar still processing")
        return
    _bar_processing = True

    try:
        _bar_count += 1
        t = bar_time.time()
        if not (time(9, 16) <= t <= time(15, 14)):
            return

        otm_call_snap = bar_acc["otm_call"]["close"]
        otm_put_snap  = bar_acc["otm_put"]["close"]
        otm_call_high = bar_acc["otm_call"]["high"] if bar_acc["otm_call"]["high"] != -np.inf else None
        otm_call_low  = bar_acc["otm_call"]["low"]  if bar_acc["otm_call"]["low"]  != np.inf  else None
        otm_put_high  = bar_acc["otm_put"]["high"]  if bar_acc["otm_put"]["high"]  != -np.inf else None
        otm_put_low   = bar_acc["otm_put"]["low"]   if bar_acc["otm_put"]["low"]   != np.inf  else None

        trade_snap = bar_acc["trade"]["close"] if "trade" in bar_acc else None
        trade_high = bar_acc["trade"]["high"] if "trade" in bar_acc and bar_acc["trade"]["high"] != -np.inf else None
        trade_low  = bar_acc["trade"]["low"]  if "trade" in bar_acc and bar_acc["trade"]["low"]  != np.inf  else None

        _spot_close = bar_acc["spot"]["close"] or 0.0

        # DYNAMIC ATM: ATM = round(spot/50)*50 at this bar's close. The "call"/"put"
        # bar is the OHLCV of whichever strike is ATM *now* (same as the rolling
        # series the model was trained on), taken from the grid, not a fixed leg.
        if GRID is not None and _spot_close > 100.0:
            _atm_bar = GRID.flush_bar(_spot_close)
            if _atm_bar is None:
                log.warning(f"ATM-SKIP {bar_time}: ATM legs {GRID.atm_strike} have no price — bar NOT pushed")
                bar_acc = _reset_bar_acc()
                return
            for _sd, _k in (("CALL", "call"), ("PUT", "put")):
                _b = _atm_bar[_sd]
                bar_acc[_k].update(open=_b["open"], high=_b["high"], low=_b["low"],
                                   close=_b["close"], volume=_b["volume"],
                                   oi=_b["oi"], iv=_b["iv"])

        _call_close = bar_acc["call"]["close"] or 0.0
        _put_close  = bar_acc["put"]["close"] or 0.0

        MIN_VALID_OPTION = 1.0
        MIN_VALID_SPOT = 100.0

        if _call_close <= MIN_VALID_OPTION:
            log.warning(
                f"ZERO-CANDLE SKIPPED {bar_time} "
                f"call={_call_close:.2f} put={_put_close:.2f} "
                f"spot={_spot_close:.2f} "
                f"reason=call_feed_missing — bar NOT pushed to buffer"
            )
            bar_acc = _reset_bar_acc()
            return

        if _put_close <= MIN_VALID_OPTION:
            log.warning(
                f"ZERO-CANDLE SKIPPED {bar_time} "
                f"call={_call_close:.2f} put={_put_close:.2f} "
                f"spot={_spot_close:.2f} "
                f"reason=put_feed_missing — bar NOT pushed to buffer"
            )
            bar_acc = _reset_bar_acc()
            return

        if _spot_close <= MIN_VALID_SPOT:
            log.warning(
                f"ZERO-CANDLE SKIPPED {bar_time} "
                f"call={_call_close:.2f} put={_put_close:.2f} "
                f"spot={_spot_close:.2f} "
                f"reason=spot_feed_missing — bar NOT pushed to buffer"
            )
            bar_acc = _reset_bar_acc()
            return

        # ── Update the separate rules-only pinned CALL series.
        # GRID.flush_bar() has already closed each leg for this minute. This
        # never writes to or substitutes into the dynamic-ATM ML buffer.
        if RULE_PIN_STRIKE is not None and RULE_PIN_SECURITY_ID is not None and GRID is not None:
            _rule_pin_close = GRID.leg_close(RULE_PIN_STRIKE, "CALL")
            if (
                _rule_pin_close is not None
                and np.isfinite(float(_rule_pin_close))
                and float(_rule_pin_close) > 0
            ):
                RULE_PIN_CLOSES.append(float(_rule_pin_close))
            elif RULE_PIN_CLOSES:
                # Keep alignment to accepted bars; never fall back to dynamic ATM.
                RULE_PIN_CLOSES.append(float(RULE_PIN_CLOSES[-1]))
                if _bar_count % 10 == 0:
                    log.warning(
                        f"[RULE-PIN] missing live close for strike={RULE_PIN_STRIKE}; "
                        "carrying forward previous pin close"
                    )

        buffer["datetime"].append(bar_time)
        buffer["spot_close"].append(bar_acc["spot"]["close"] or 0)
        buffer["call_close"].append(bar_acc["call"]["close"] or 0)
        buffer["call_high"].append(bar_acc["call"]["high"] if bar_acc["call"]["high"] != -np.inf else 0)
        buffer["call_low"].append(bar_acc["call"]["low"] if bar_acc["call"]["low"] != np.inf else 0)
        buffer["call_open"].append(bar_acc["call"]["open"] or 0)
        buffer["put_open"].append(bar_acc["put"]["open"] or 0)
        buffer["put_high"].append(bar_acc["put"]["high"] if bar_acc["put"]["high"] != -np.inf else 0)
        buffer["put_low"].append(bar_acc["put"]["low"] if bar_acc["put"]["low"] != np.inf else 0)
        if GRID is not None:
            _call_vol_bar = int(max(0, bar_acc["call"]["volume"] or 0))
            _put_vol_bar  = int(max(0, bar_acc["put"]["volume"] or 0))
        else:
            _call_vol_cum = bar_acc["call"]["volume"] or 0
            _put_vol_cum = bar_acc["put"]["volume"] or 0
            _call_vol_bar = (_call_vol_cum - _last_cum_call) if _last_cum_call is not None else 0
            _put_vol_bar  = (_put_vol_cum  - _last_cum_put)  if _last_cum_put  is not None else 0
            _call_vol_bar = max(0, int(_call_vol_bar)); _put_vol_bar = max(0, int(_put_vol_bar))
            _last_cum_call = _call_vol_cum; _last_cum_put = _put_vol_cum
        buffer["call_volume"].append(_call_vol_bar)
        buffer["put_volume"].append(_put_vol_bar)
        buffer["call_oi"].append(bar_acc["call"]["oi"])
        buffer["put_oi"].append(bar_acc["put"]["oi"])
        buffer["call_iv"].append(bar_acc["call"]["iv"])
        buffer["put_iv"].append(bar_acc["put"]["iv"])
        buffer["india_vix"].append(india_vix)
        buffer["put_close"].append(bar_acc["put"]["close"] or 0)

        buffer["otm_call_high"].append(otm_call_high if otm_call_high is not None else 0)
        buffer["otm_call_low"].append(otm_call_low if otm_call_low is not None else 0)
        buffer["otm_call_close"].append(otm_call_snap if otm_call_snap is not None else 0)
        buffer["otm_put_high"].append(otm_put_high if otm_put_high is not None else 0)
        buffer["otm_put_low"].append(otm_put_low if otm_put_low is not None else 0)
        buffer["otm_put_close"].append(otm_put_snap if otm_put_snap is not None else 0)

        # ── Drive the L3 model's HTF / pivot / rvol state (served per bar)
        #    Mirrors niftybacktest4.py serving pattern: htf from ATM-call
        #    OHLCV, pivot from spot. Required by v12_inference._build_features.
        _bc = buffer["call_close"][-1];  _bh = buffer["call_high"][-1]
        _bl = buffer["call_low"][-1];    _bo = buffer["call_open"][-1]
        _bv = buffer["call_volume"][-1]; _sp = buffer["spot_close"][-1]
        _btime = bar_time
        try:
            if not (isinstance(_bc, (int, float)) and _bc > 0):
                raise ValueError("bad call close")
            model.htf.update(_btime, o=_bo, h=_bh, l=_bl, c=_bc, vol=_bv, sp=_sp)
            _bdate = _btime.date()
            model.pivot.update(_bdate, spot_high=_sp, spot_low=_sp, spot_close=_sp)
        except Exception as _e:
            log.warning(f"[MODEL] htf/pivot update skipped bar={_btime}: {_e}")

        bar_acc = _reset_bar_acc()

        atm_close = buffer["call_close"][-1]
        atm_put_close = buffer["put_close"][-1]
        spot_close = buffer["spot_close"][-1]
        buf_len = len(buffer["call_close"])

        global _LIVE_BARS_SINCE_WARMSTART
        if _WARMSTART_USED:
            _LIVE_BARS_SINCE_WARMSTART += 1

        if _cooldown_remaining > 0:
            _cooldown_remaining -= 1

        bars_since_exit = _bar_count - _last_exit_bar
        warmup_left = max(0, 50 - buf_len)
        trade_state = (
            f"IN {active_trade['signal'][:5]} entry={active_trade['entry']:.2f} bars={active_trade['bar_held']}"
            if active_trade
            else f"COOLDOWN={_cooldown_remaining}" if _cooldown_remaining > 0
            else "FLAT"
        )
        log.info(
        f"BAR {_bar_count:4d} {bar_time} "
            f"call={atm_close:8.2f} put={atm_put_close:8.2f} "
            f"spot={spot_close:9.2f} buf={buf_len:4d}/{BUFFER_SIZE} "
            f"vix={india_vix:.1f} "
            f"{'WARMUP ' + str(warmup_left) + ' bars' if warmup_left else trade_state}"
        )

        # ── [GRID] coverage heartbeat: how many far-strike legs have a live price
        if GRID is not None and _bar_count % 10 == 0:
            _g_live, _g_tot = GRID.coverage()
            log.info(f"  [GRID] legs with live price: {_g_live}/{_g_tot}")

        # ── A3: Serve-consistency self-check (fires once when warmup ends) ──
        # Guarantees the raw chain columns the model expects are actually being
        # fed live. If any ATM column is zero we are back to the train/serve skew
        # that caused the SHORT bias — log it loudly instead of failing silently.
        global _SERVE_CHECK_DONE
        _buf_adapter = _build_buf_adapter()
        if warmup_left == 0 and not _SERVE_CHECK_DONE:
            _SERVE_CHECK_DONE = True
            try:
                _raw_now = build_raw_row()
                if GRID is not None:
                    _raw_now = GRID.merge_into(_raw_now)
                _feat_now = _build_features(
                    raw_row=_raw_now, buf=_buf_adapter,
                    htf=model.htf, pivot=model.pivot, rvol_hist=model.rvol_hist,
                    india_vix=india_vix, vix_chg=0.0, feat_cols=model.features)
                _expected = [c for c in model.features if "nifty_WEEK_ATM" in c]
                _zero = [c for c in _expected if not _feat_now or float(_feat_now.get(c, 0.0)) == 0.0]
                _ATM_COLS = {
                    "nifty_WEEK_ATM_CALL_1m__open", "nifty_WEEK_ATM_CALL_1m__volume",
                    "nifty_WEEK_ATM_CALL_1m__iv", "nifty_WEEK_ATM_CALL_1m__oi",
                    "nifty_WEEK_ATM_CALL_1m__spot", "nifty_WEEK_ATM_PUT_1m__open",
                    "nifty_WEEK_ATM_PUT_1m__volume", "nifty_WEEK_ATM_PUT_1m__iv",
                    "nifty_WEEK_ATM_PUT_1m__oi",
                }
                _atm_zero = [c for c in _zero if c in _ATM_COLS]
                _atm_zero_nonvol = [
                    c for c in _atm_zero if not c.endswith("__volume")
                ]
                _atm_zero_vol = [c for c in _atm_zero if c.endswith("__volume")]
                _far_zero = [c for c in _zero if c not in _ATM_COLS]
                # Volume is a per-bar delta of cumulative exchange volume. A zero
                # delta can be valid when there were no trades in that minute.
                _far_zero_nonvol = [c for c in _far_zero if not c.endswith("__volume")]
                if _atm_zero_nonvol:
                    log.warning(f"[SERVE-CHECK] ⚠️ {len(_atm_zero_nonvol)} ATM non-volume "
                                f"raw cols are 0 (skew risk): {_atm_zero_nonvol}")
                else:
                    log.info(f"[SERVE-CHECK] ✅ ATM non-volume raw cols populated")
                if _atm_zero_vol:
                    log.info(f"[SERVE-CHECK] ATM per-bar volume delta is 0 for "
                             f"{_atm_zero_vol} (valid when no trades/volume change "
                             "occurred in the bar)")
                if _far_zero_nonvol:
                    log.warning(f"[SERVE-CHECK] ⚠️ {len(_far_zero_nonvol)}/{len(_expected)} "
                                f"far-strike non-volume cols still 0 (grid leg not feeding?): "
                                f"{_far_zero_nonvol[:12]}{'...' if len(_far_zero_nonvol) > 12 else ''}")
                else:
                    log.info(f"[SERVE-CHECK] ✅ far-strike grid populated; "
                             f"{len(_expected)} raw cols expected, {len(_zero)} zero "
                             f"(volume-only zeros are normal for illiquid strikes)")
            except Exception as _e:
                log.warning(f"[SERVE-CHECK] could not run: {_e}")

        # ── [WINDOW-CHECK]: confirms rolling_rank/vix_ma20 are actually
        # reading FULL_WINDOW bars, not the old hardcoded 100 -- fires once.
        # A window_used far below FULL_WINDOW just means the buffer hasn't
        # accumulated that much yet (expected pre-warm-start); after a
        # successful warm-start it should read close to FULL_WINDOW.
        global _WINDOW_CHECK_DONE
        if warmup_left == 0 and not _WINDOW_CHECK_DONE:
            _WINDOW_CHECK_DONE = True
            try:
                _cc_arr = np.array(list(buffer["call_close"]))
                _window_used = min(len(_cc_arr), FULL_WINDOW)
                _sample_rank = rolling_rank(_cc_arr)
                _vix_arr = np.array(list(buffer["india_vix"]))
                _sample_vix_ma = (float(np.mean(_vix_arr[-FULL_WINDOW:]))
                                  if len(_vix_arr) >= 100 else None)
                log.info(
                    f"[WINDOW-CHECK] rolling_rank window_used={_window_used} bars "
                    f"(target FULL_WINDOW={FULL_WINDOW}; was hardcoded 100 before "
                    f"this fix) sample_price_position_rank={_sample_rank:.4f} "
                    f"vix_ma20={_sample_vix_ma if _sample_vix_ma is not None else 'N/A (buf<100)'}"
                )
                if _window_used < FULL_WINDOW * 0.5:
                    log.warning(
                        f"[WINDOW-CHECK] ⚠️ window_used={_window_used} is well below "
                        f"FULL_WINDOW={FULL_WINDOW} -- warm-start may not have "
                        f"populated the buffer as expected, check for errors above"
                    )
            except Exception as _e:
                log.warning(f"[WINDOW-CHECK] could not run: {_e}")



        # ── 1. Sync active_trade with order_manager ───────────────────────
        if order_manager is not None:
            om_trade = order_manager.get_active_trade()
            if active_trade is not None and om_trade is None:
                log.warning(
                    f"SYNC: broker closed trade externally signal={active_trade['signal']} "
                    f"resetting local state"
                )
                _cooldown_remaining = COOLDOWNBARS
                _last_exit_bar = _bar_count
                _last_exit_reason = "BROKER_CLOSE"
                _last_signal_dir = active_trade["signal"]
                active_trade = None
                _ACTIVE_TRADE_FEED_ID = None
                _ACTIVE_TRADE_FEED_SYMBOL = None
                _LAST_TRADE_PRICE = None
                _LAST_TRADE_TICK_TS = None
            elif om_trade is not None:
                active_trade = om_trade

        # ── Guard: sanitize active_trade against coroutine poisoning ──────
        if active_trade is not None:
            entry_val = active_trade.get("entry")
            if not isinstance(entry_val, (int, float)):
                log.error(
                    f"CORRUPTED active_trade detected — "
                    f"entry={type(entry_val).__name__}. "
                    f"Clearing state. CHECK ZERODHA POSITIONS IMMEDIATELY."
                )
                active_trade = None
                if order_manager is not None:
                    order_manager.active_trade = None
                _cooldown_remaining = COOLDOWNBARS
                _ACTIVE_TRADE_FEED_ID = None
                _ACTIVE_TRADE_FEED_SYMBOL = None
                _LAST_TRADE_PRICE = None
                _LAST_TRADE_TICK_TS = None

        # ── 2. EXIT LOGIC ─────────────────────────────────────────────────
        if active_trade:
            at = active_trade
            at_signal = at["signal"]
            at_entry = at["entry"]
            at_sl = at["sl"]
            at_sl_meth = at.get("sl_method", "ATR")
            at_prob_long = at.get("prob_long", 0.5)
            at_prob_short = at.get("prob_short", 0.5)
            at["bar_held"] = at.get("bar_held", 0) + 1
            at_bar_held = at["bar_held"]

            if at_signal == "LONG":
                bar_high = trade_high if trade_high is not None else (otm_call_high if otm_call_high and otm_call_high != -np.inf else None)
                bar_low = trade_low if trade_low is not None else (otm_call_low if otm_call_low and otm_call_low != np.inf else None)
                bar_close = trade_snap if trade_snap and trade_snap > 0 else (otm_call_snap if otm_call_snap and otm_call_snap > 0 else None)
                raw_trade_close = trade_snap if trade_snap and trade_snap > 0 else otm_call_snap
                iv_buf_x = buffer["call_iv"]
                oi_buf_x = buffer["call_oi"]
                otm_h_x = buffer["otm_call_high"]
                otm_l_x = buffer["otm_call_low"]
                otm_c_x = buffer["otm_call_close"]
                otm_close_buf = buffer["otm_call_close"]
            else:
                bar_high = trade_high if trade_high is not None else (otm_put_high if otm_put_high and otm_put_high != -np.inf else None)
                bar_low = trade_low if trade_low is not None else (otm_put_low if otm_put_low and otm_put_low != np.inf else None)
                bar_close = trade_snap if trade_snap and trade_snap > 0 else (otm_put_snap if otm_put_snap and otm_put_snap > 0 else None)
                raw_trade_close = trade_snap if trade_snap and trade_snap > 0 else otm_put_snap
                iv_buf_x = buffer["put_iv"]
                oi_buf_x = buffer["put_oi"]
                otm_h_x = buffer["otm_put_high"]
                otm_l_x = buffer["otm_put_low"]
                otm_c_x = buffer["otm_put_close"]
                otm_close_buf = buffer["otm_put_close"]

            # DYNAMIC ATM: the traded option is ONE fixed strike. IV z-score, OI change,
            # supertrend and SL must use ITS history, not the ATM series (which hops strikes
            # and would fake OI/IV changes every time ATM rolls).
            if GRID is not None and at.get("strike"):
                _ls = GRID.leg_series(at["strike"], at["side"])
                if _ls is not None:
                    bar_high, bar_low, bar_close = _ls["high"][-1], _ls["low"][-1], _ls["close"][-1]
                    raw_trade_close = bar_close
                    iv_buf_x, oi_buf_x = _ls["iv"], _ls["oi"]
                    otm_h_x, otm_l_x, otm_c_x = _ls["high"], _ls["low"], _ls["close"]
                    otm_close_buf = _ls["close"]

            skip_hard_sl_this_bar = False

            if not raw_trade_close or raw_trade_close <= 0:
                at["zero_bar_count"] = at.get("zero_bar_count", 0) + 1
                zbc = at["zero_bar_count"]

                if at_bar_held <= 5:
                    log.warning(
                        f"ZERO-BAR #{zbc} signal={at_signal} "
                        f"bar_held={at_bar_held} "
                        f"feed warming up after entry — skipping exit eval"
                    )
                    return
                else:
                    if (
                        _LAST_TRADE_PRICE is not None
                        and _LAST_TRADE_PRICE > 0
                        and _LAST_TRADE_TICK_TS is not None
                        and (datetime.now() - _LAST_TRADE_TICK_TS).total_seconds() <= 90
                    ):
                        bar_close = _LAST_TRADE_PRICE
                        bar_low = _LAST_TRADE_PRICE
                        bar_high = _LAST_TRADE_PRICE
                        skip_hard_sl_this_bar = True
                        log.warning(
                            f"ZERO-BAR #{zbc} signal={at_signal} "
                            f"bar_held={at_bar_held} "
                            f"no new traded tick — using last traded price "
                            f"Rs{_LAST_TRADE_PRICE:.2f} (flat bar), hard SL skipped"
                        )
                    else:
                        last_valid_otm = next(
                            (p for p in reversed(list(otm_close_buf)) if p and p > 0),
                            None
                        )
                        if last_valid_otm is not None:
                            bar_close = last_valid_otm
                            bar_low = last_valid_otm
                            bar_high = last_valid_otm
                            skip_hard_sl_this_bar = True
                            log.warning(
                                f"ZERO-BAR #{zbc} signal={at_signal} "
                                f"bar_held={at_bar_held} "
                                f"no new tick — using last valid OTM price "
                                f"Rs{last_valid_otm:.2f} (flat bar), hard SL skipped"
                            )
                        else:
                            bar_close = atm_close if at_signal == "LONG" else atm_put_close
                            bar_low = bar_close
                            bar_high = bar_close
                            skip_hard_sl_this_bar = True
                            log.warning(
                                f"ZERO-BAR #{zbc} signal={at_signal} "
                                f"bar_held={at_bar_held} "
                                f"no OTM price in buffer at all — "
                                f"ATM fallback Rs{bar_close:.2f}, hard SL skipped"
                            )
            else:
                at["zero_bar_count"] = 0

            if not bar_close or bar_close <= 0:
                bar_close = atm_close if at_signal == "LONG" else atm_put_close
            if not bar_low or bar_low <= 0:
                bar_low = bar_close
            if not bar_high or bar_high <= 0:
                bar_high = bar_close

            iv_z = compute_iv_zscore(list(iv_buf_x), IV_ZSCORE_LOOKBACK)
            oi_chg = compute_oi_change_pct(list(oi_buf_x))

            # ── [DIAG] per-bar exit visibility — pnl uses whatever bar_close
            #    is at this point (post zero-bar handling above) ───────────
            _diag_pnl_pct = (bar_close - at_entry) / at_entry * 100
            _diag_st_dir = None
            if len(otm_c_x) >= ST_PERIOD + 2:
                try:
                    _diag_st_dir = compute_supertrend(otm_h_x, otm_l_x, otm_c_x, period=ST_PERIOD, mult=ST_MULT)
                except Exception:
                    _diag_st_dir = None
            log.info(
                f"  [EXIT-CHECK] signal={at_signal} bar_held={at_bar_held} "
                f"close={bar_close:.2f} entry={at_entry:.2f} pnl={_diag_pnl_pct:.2f}% "
                f"sl={at_sl:.2f} ivz={iv_z:.2f} oichg={oi_chg:.2f} "
                f"st_dir={_diag_st_dir} skip_hard_sl={skip_hard_sl_this_bar}"
            )

            if skip_hard_sl_this_bar:
                exit_flag = False
                exit_reason = None
                exec_price = None

                pnl_pct_now = (bar_close - at_entry) / at_entry * 100

                if iv_z >= IV_ZSCORE_EXIT and pnl_pct_now >= IV_MIN_PROFIT_PCT:
                    exit_flag, exit_reason, exec_price = True, "IV_ZSCORE", bar_close
                elif oi_chg <= OI_DROP_THRESH and pnl_pct_now >= OI_MIN_PROFIT_PCT:
                    exit_flag, exit_reason, exec_price = True, "OI_UNWIND", bar_close
                else:
                    if len(otm_c_x) >= ST_PERIOD + 2:
                        st_dir = compute_supertrend(otm_h_x, otm_l_x, otm_c_x,
                                                    period=ST_PERIOD, mult=ST_MULT)
                        # Long premium on both sides -> adverse flip is always -1.
                        if st_dir == -1 and pnl_pct_now >= ST_MIN_PROFIT_PCT:
                            exit_flag, exit_reason, exec_price = True, "SUPERTREND", bar_close

                if not exit_flag:
                    if at_bar_held >= MAX_BARS_HELD:
                        exit_flag, exit_reason, exec_price = True, "TIME_STOP", bar_close
                    elif t >= time(15, 14):
                        exit_flag, exit_reason, exec_price = True, "EOD", bar_close
            else:
                exit_flag, exit_reason, exec_price = smart_exit_check(
                    signal=at_signal,
                    entry=at_entry,
                    sl_price=at_sl,
                    bar_time=bar_time,
                    bar_high=bar_high,
                    bar_low=bar_low,
                    bar_close=bar_close,
                    bars_held=at_bar_held,
                    iv_buf=iv_buf_x,
                    oi_buf=oi_buf_x,
                    otm_h_buf=otm_h_x,
                    otm_l_buf=otm_l_x,
                    otm_c_buf=otm_c_x,
                )

            if exit_flag:
                exit_confirmed = True   # assume success unless order_manager tells us otherwise

                if order_manager is not None:
                    summary = await order_manager.place_exit(
                        reason=exit_reason,
                        exit_price_hint=exec_price,
                        bar_dt=bar_time,
                    )

                    if summary and isinstance(summary, dict):
                        exec_price = summary.get("exit_price", exec_price)
                        # If order_manager signals failure explicitly via a
                        # "success" key, respect it. If the key isn't present,
                        # a returned dict with an exit_price is treated as success
                        # (preserves current behavior for order_manager versions
                        # that don't yet return a success flag).
                        if "success" in summary:
                            exit_confirmed = bool(summary.get("success"))
                    elif summary is None:
                        # No dict returned at all — treat as failure signal
                        exit_confirmed = False

                    if not exit_confirmed:
                        # ── Exit order failed — verify against ACTUAL broker
                        #    position before deciding what to do. Prevents an
                        #    infinite retry loop if the position was already
                        #    closed externally (e.g. squared off manually on
                        #    Kite), which otherwise causes repeated
                        #    "Invalid quantity" rejections forever.
                        try:
                            broker_positions = kite.positions()
                            net_positions = broker_positions.get("net", [])
                            live_qty = 0
                            at_symbol = at.get("tradingsymbol")
                            for p in net_positions:
                                if p.get("tradingsymbol") == at_symbol:
                                    live_qty = p.get("quantity", 0)
                                    break

                            if live_qty == 0:
                                log.warning(
                                    f"EXIT FAILED but broker shows qty=0 for "
                                    f"{at_symbol} — position was already closed "
                                    f"externally (e.g. manually on Kite). "
                                    f"Clearing local active_trade to stop the "
                                    f"retry loop. No broker action taken."
                                )
                                tracker.log_trade(
                                    bar_time, at_signal, at_entry, bar_close,
                                    "EXTERNAL_CLOSE_DETECTED", at_prob_long, at_prob_short,
                                )
                                _cooldown_remaining = COOLDOWNBARS
                                _last_exit_bar = _bar_count
                                _last_exit_reason = "EXTERNAL_CLOSE_DETECTED"
                                _last_signal_dir = at_signal
                                active_trade = None
                                order_manager.active_trade = None
                                _ACTIVE_TRADE_FEED_ID = None
                                _ACTIVE_TRADE_FEED_SYMBOL = None
                                _LAST_TRADE_PRICE = None
                                _LAST_TRADE_TICK_TS = None
                                return
                            else:
                                log.error(
                                    f"EXIT FAILED and broker STILL shows live "
                                    f"qty={live_qty} for {at_symbol} — genuine "
                                    f"order failure, not a stale-state issue. "
                                    f"Will retry next bar. CHECK MANUALLY if "
                                    f"this repeats."
                                )
                                return
                        except Exception as _pos_check_err:
                            log.error(
                                f"EXIT FAILED and position verification ALSO "
                                f"failed: {_pos_check_err}. Cannot confirm "
                                f"broker state — will retry next bar. "
                                f"CHECK ZERODHA POSITIONS MANUALLY."
                            )
                            return

                pnl_pct = (exec_price - at_entry) / at_entry * 100

                if order_manager is not None:
                    if at_signal == "LONG":
                        old_token = _ACTIVE_TRADE_FEED_ID or INSTRUMENTS["otm_call"][1]
                        old_symbol = _ACTIVE_TRADE_FEED_SYMBOL or INSTRUMENTS["otm_call"][0]
                        INSTRUMENTS["otm_call"] = _DEFAULT_OTM_CALL
                        INSTRUMENT_STRIKES["otm_call"] = INSTRUMENT_STRIKES.get("atm_call", 0) + 50  # ← ADD
                        log.info(f"INSTRUMENTS restored: otm_call → {_DEFAULT_OTM_CALL[1]} (default)")
                        feed_resubscribe(
                            new_token=_DEFAULT_OTM_CALL[1],
                            new_symbol=_DEFAULT_OTM_CALL[0],
                            old_token=old_token,
                            old_symbol=old_symbol,
                            direction="EXIT LONG → default",
                        )
                    else:
                        old_token = _ACTIVE_TRADE_FEED_ID or INSTRUMENTS["otm_put"][1]
                        old_symbol = _ACTIVE_TRADE_FEED_SYMBOL or INSTRUMENTS["otm_put"][0]
                        INSTRUMENTS["otm_put"] = _DEFAULT_OTM_PUT
                        INSTRUMENT_STRIKES["otm_put"] = INSTRUMENT_STRIKES.get("atm_put", 0) - 50  # ← ADD
                        log.info(f"INSTRUMENTS restored: otm_put → {_DEFAULT_OTM_PUT[1]} (default)")
                        feed_resubscribe(
                            new_token=_DEFAULT_OTM_PUT[1],
                            new_symbol=_DEFAULT_OTM_PUT[0],
                            old_token=old_token,
                            old_symbol=old_symbol,
                            direction="EXIT SHORT → default",
                        )

                tracker.log_trade(
                    bar_time, at_signal, at_entry, exec_price,
                    exit_reason, at_prob_long, at_prob_short,
                )

                # (A) EXIT-GATE audit — mirrors ENTRY-GATE so every cut is diagnosable.
                # adx_1m recomputed on the live call_close buffer (same series _build_features
                # uses) so it's directly comparable to the entry; spot_chg shows the move
                # into the exit; reason names which of the 5-priority chain fired.
                try:
                    _ex_adx, _ex_pdi, _ex_ndi = compute_adx_arr(list(buffer["call_close"]))
                except Exception:
                    _ex_adx, _ex_pdi, _ex_ndi = 0.0, 50.0, 50.0
                try:
                    _ex_rg = list(buffer["spot_close"])[-(REVERSAL_GUARD_BARS + 1):]
                    _ex_rg_chg = (_ex_rg[-1] - _ex_rg[0]) / (_ex_rg[0] + 1e-9) * 100.0 if len(_ex_rg) >= 2 else 0.0
                except Exception:
                    _ex_rg_chg = 0.0
                log.info(
                    f"EXIT-GATE reason={exit_reason} adx_1m={_ex_adx:.2f} "
                    f"pdi={_ex_pdi:.1f} ndi={_ex_ndi:.1f} adx_src=call_close "
                    f"nbars={len(buffer['call_close'])} "
                    f"spot_chg_{REVERSAL_GUARD_BARS}bar={_ex_rg_chg:+.2f}%"
                )

                log.info(
                    f"EXIT {at_signal:<5} {exec_price:8.2f} "
                    f"entry={at_entry:8.2f} pnl={pnl_pct:.3f}% "
                    f"{exit_reason} bars_held={at_bar_held} "
                    f"sl={at_sl:.2f} ivz={iv_z:.2f} "
                    f"oichg={oi_chg:.2f} sl_method={at_sl_meth}"
                )

                _cooldown_remaining = COOLDOWNBARS
                _last_exit_bar = _bar_count
                _last_exit_reason = exit_reason
                _last_signal_dir = at_signal
                active_trade = None
                _ACTIVE_TRADE_FEED_ID = None
                _ACTIVE_TRADE_FEED_SYMBOL = None
                _LAST_TRADE_PRICE = None
                _LAST_TRADE_TICK_TS = None

                if len(tracker.trades) > 0 and len(tracker.trades) % 20 == 0:
                    tracker.status(window=20)
                    tracker.save("paper_trades_v14dhan_l5.csv")
                    log.info(f"saved total {len(tracker.trades)} trades")

                return

        # ── 3. Guard: skip entry if in trade or cooldown ──────────────────
        if active_trade or _cooldown_remaining > 0:
            return

        # ── Guard: post-warm-start seam cooldown ───────────────────────────
        if _WARMSTART_USED and _LIVE_BARS_SINCE_WARMSTART < POST_WARMSTART_COOLDOWN_BARS:
            if _bar_count % 5 == 0:  # don't spam every bar
                log.info(f"ENTRY BLOCKED post-warmstart cooldown "
                        f"{_LIVE_BARS_SINCE_WARMSTART}/{POST_WARMSTART_COOLDOWN_BARS} live bars")
            return

        # ── 4. Build features (training-exact pipeline via v12_inference)
        #    The bot's old build_features() drifted from what the model was
        #    trained on and never populated the chain grid. We route through
        #    the SAME _build_features used in training, fed by the bot's
        #    buffer (key adapter below).
        _buf_adapter = _build_buf_adapter()
        # ── A2 + CHAIN-GRID FIX: train/serve skew ───────────────────────
        # The model was trained on the raw `nifty_WEEK_ATM*` chain columns
        # (36.6% of importance). The 9 ATM columns come from the live buffer;
        # the ~115 far-strike columns now come from the ChainGrid (live
        # ticks on the ATM-10..ATM+10 strikes) instead of being left at 0.0.
        # call/put_iv are already in percent units (compute_live_iv returns
        # iv*100), matching the training scale.
        if buffer.get("call_open") and len(buffer["call_open"]) > 0:
            raw_row = build_raw_row()
        else:
            raw_row = {}
        if GRID is not None:
            raw_row = GRID.merge_into(raw_row)

        if GRID is not None:
            live_legs, total_legs = GRID.coverage()
            if live_legs < 0.9 * total_legs:
                if _bar_count % 5 == 0:
                    log.warning(f"ENTRY BLOCKED grid coverage {live_legs}/{total_legs}")
                return

        feat = _build_features(
            raw_row=raw_row,
            buf=_buf_adapter,
            htf=model.htf,
            pivot=model.pivot,
            rvol_hist=model.rvol_hist,
            india_vix=india_vix,
            vix_chg=0.0,
            feat_cols=model.features,
        )
        if feat is None:
            return

        # ── 5. Model predict ──────────────────────────────────────────────
        # v12_l5_inference.V12L5Model.predict_bar() returns the same dict
        # shape as v12_inference.V12Model.predict_bar() (signal/prob_long/
        # prob_short/confidence/...), so no adapter needed -- every line
        # below (rule filters, momentum gate, entry/exit, logging, tracker)
        # is byte-identical to the l3 file.
        result = model.predict_bar(
            feature_dict=feat,
            india_vix=INDIA_VIX,
            vol_regime=int(feat.get("vol_regime", 0)),
            is_pre_close=int(feat.get("is_pre_close", 0)),
        )

        if _bar_count % 10 == 0:
            log.info(
                f"  [MODEL] pl={result['prob_long']:.3f} ps={result['prob_short']:.3f} "
                f"sig={result['signal']} adx={feat.get('adx_1m',0):.1f} "
                f"rvol={feat.get('rvol',0):.2f}"
            )

        if result["signal"] == "NONE":
            return

        raw_signal = result["signal"]
        confidence = result["confidence"]
        spot_closes = buffer.get("spot_close")

        # Indicators used only by rule gates. `feat` remains the model's
        # original dynamic-ATM feature dictionary and is not mutated.
        rule_rsi5m, rule_macdh5m = _compute_rule_pin_5m()
        rule_feat = dict(feat)
        if rule_rsi5m is not None and rule_macdh5m is not None:
            rule_feat["rsi_5m"] = rule_rsi5m
            rule_feat["macdh_5m"] = rule_macdh5m
        if _bar_count % 10 == 0:
            log.info(
                f"[RULE-PIN] strike={RULE_PIN_STRIKE} rows={len(RULE_PIN_CLOSES)} "
                f"rsi5m={rule_rsi5m if rule_rsi5m is not None else 'WARMUP'} "
                f"macdh5m={rule_macdh5m if rule_macdh5m is not None else 'WARMUP'}"
            )

        # ── 6. Rule filters ───────────────────────────────────────────────
        filtered_signal, filter_reason = apply_rule_filters(
            signal          = raw_signal,
            feat            = feat,
            bartime         = bar_time,
            lastexitreason  = _last_exit_reason,
            lastsignaldir   = _last_signal_dir,
            barssinceexit   = bars_since_exit,
            confidence      = confidence,
            spotbuffer      = spot_closes,
            rule_rsi5m      = rule_rsi5m,
            rule_macdh5m    = rule_macdh5m,
        )

        if filtered_signal == "NONE":
            log.info(f"SIGNAL FILTERED raw={raw_signal} conf={confidence:.3f} reason={filter_reason}")
            return

        # ── 7. Strike range guard ─────────────────────────────────────────
        in_range, atm_strike, exec_strike = is_strike_in_range(
            spot_close, OTM_OFFSET, filtered_signal, STRIKE_STEP
        )
        if not in_range:
            log.info(
                f"STRIKE OUT OF RANGE spot={spot_close:.0f} "
                f"ATM={atm_strike} exec={exec_strike} "
                f"limit=ATM+-{STRIKE_RANGE_PTS}"
            )
            return

        # ── 8. OTM-only guard ─────────────────────────────────────────────
        otm_ok, otm_reason = is_strike_otm(spot_close, exec_strike, filtered_signal)
        if not otm_ok:
            log.info(
                f"ITM BLOCK signal={filtered_signal} {otm_reason} "
                f"spot={spot_close:.0f} instrument has gone ITM, skipping"
            )
            return

        # ── 9. ATR for momentum score ─────────────────────────────────────
        _side = "CALL" if filtered_signal == "LONG" else "PUT"
        _leg = GRID.leg_series(exec_strike, _side) if GRID is not None else None
        if GRID is not None and _leg is None:
            log.info(f"ENTRY BLOCKED no grid history for exec leg {exec_strike} {_side}")
            return
        if _leg is not None:
            atr_val  = compute_option_atr(_leg["high"], _leg["low"], _leg["close"], ATR_PERIOD)
            prem_buf = _leg["close"]
        elif filtered_signal == "LONG":
            atr_val = compute_option_atr(buffer["otm_call_high"], buffer["otm_call_low"],
                                         buffer["otm_call_close"], ATR_PERIOD)
            prem_buf = buffer["otm_call_close"]
        else:
            atr_val = compute_option_atr(buffer["otm_put_high"], buffer["otm_put_low"],
                                         buffer["otm_put_close"], ATR_PERIOD)
            prem_buf = buffer["otm_put_close"]

        # ── 10. Momentum score gate ───────────────────────────────────────
        rule_feat["conf_proxy"] = confidence
        spot_list = list(buffer["spot_close"])
        mscore, mscore_inputs = compute_momentum_score_inputs(
            rule_feat, filtered_signal, atr_val or 0, prem_buf, spot_list, leg=_leg)
        if mscore < MOMENTUM_SCORE_THRESHOLD:
            log.info(
                f"MOMENTUM GATE BLOCKED signal={filtered_signal} "
                f"score={mscore}<{MOMENTUM_SCORE_THRESHOLD} "
                f"confok={mscore_inputs['conf_ok']} "
                f"adxdirok={mscore_inputs['adx_dir_ok']} "
                f"trend={mscore_inputs['trend_aligned']} "
                f"consec={mscore_inputs['consec_bars']} "
                f"spread={mscore_inputs['spread_widening']} "
                f"clean={mscore_inputs['entry_clean']} "
                f"atrok={mscore_inputs['atr_ok']} "
                f"strong={mscore_inputs['strong_trend']}"
            )
            return

        # ── 11. Capital check ─────────────────────────────────────────────
        if order_manager is not None:
            trade_capital = get_available_capital(kite, use_pct=1.0, segment="equity")
            if trade_capital < MIN_TRADE_CAPITAL:
                log.warning(f"SKIP insufficient_capital available=Rs{trade_capital:.0f}")
                return
        else:
            trade_capital = 100000.0

        # ── 12. Fetch option chain + filter OTM candidates ────────────────
        option_type = "CE" if filtered_signal == "LONG" else "PE"

        if order_manager is not None:
            try:
                chain = get_option_chain(kite, expiry)
                otm_df = get_otm_strikes(
                    chain,
                    spot=spot_close,
                    option_type=option_type,
                    strike_range=STRIKE_RANGE_PTS,
                )
            except Exception as e:
                log.warning(f"SKIP option_chain_error {e} signal={filtered_signal}")
                return

            # ── 13. Pick best OTM strike within budget ────────────────────
            if _mapper is None:
                log.error(
                    "SKIP: _mapper is None in on_bar_complete — "
                    "mapper not passed from process_tick. "
                    "Check asyncio.run(on_bar_complete(..., _mapper=mapper)) call site."
                )
                return

            best = pick_best_otm_strike(
                kite,
                otm_df,
                trade_capital=trade_capital,
                lot_size=LOT_SIZE,
                lots=MAX_LOTS,
                log_top_n=5,
                mapper=_mapper,
            )
            if best is None:
                log.warning(
                    f"SKIP no_affordable_strike "
                    f"capital=Rs{trade_capital:.0f} signal={filtered_signal}"
                )
                return

            exec_entry = best["ltp"]
            best_symbol = best["tradingsymbol"]
            best_token_confirmed = int(best["instrument_token"])
            best_dhan_id = best.get("dhan_security_id")
            best_dhan_symbol = best.get("dhan_trading_symbol")
            best_dhan_expiry = best.get("dhan_expiry")

            if not best_dhan_id:
                try:
                    resolved = _mapper.resolve_dhan_contract(best_symbol)
                    best_dhan_id = resolved["dhan_security_id"]
                    best_dhan_symbol = resolved["dhan_trading_symbol"]
                    best_dhan_expiry = resolved["dhan_expiry"]
                    best["dhan_security_id"] = best_dhan_id
                    best["dhan_trading_symbol"] = best_dhan_symbol
                    best["dhan_expiry"] = best_dhan_expiry
                    log.warning(
                        f"dhan_security_id was None in best dict — "
                        f"resolved via direct lookup: {best_dhan_id} "
                        f"({best_dhan_symbol}, expiry={best_dhan_expiry})"
                    )
                except Exception as _id_err:
                    log.error(
                        f"CRITICAL: cannot resolve Dhan security_id for "
                        f"{best_symbol}: {_id_err} "
                        f"— rejecting trade to avoid ZERO-BAR. "
                        f"Check Dhan scrip master."
                    )
                    return

            if not best_dhan_id:
                log.error(
                    f"CRITICAL: Dhan security_id still missing for {best_symbol} "
                    f"after resolution attempt — rejecting trade."
                )
                return

        else:
            if GRID is not None:
                exec_entry = GRID.leg_close(exec_strike, _side)
                if not exec_entry:
                    log.warning(f"SKIP no price for exec leg {exec_strike} {_side}")
                    return
            else:
                exec_entry = ((otm_call_snap or atm_close) if filtered_signal == "LONG"
                              else (otm_put_snap or atm_put_close))
            best_symbol = f"PAPER_{filtered_signal}_{exec_strike:.0f}{option_type}"
            best_token_confirmed = None
            best_dhan_id = None
            best_dhan_symbol = None
            best_dhan_expiry = None
            best = {
                "lot_size": LOT_SIZE,
                "lots": MAX_LOTS,
                "total_cost": exec_entry * LOT_SIZE,
                "distance_pts": abs(exec_strike - spot_close),
                "oi": 0,
            }

        # ── 14. Compute SL (IV-adjusted ATR) ──────────────────────────────
        if filtered_signal == "LONG":
            iv_buf_entry = list(buffer["call_iv"])
        else:
            iv_buf_entry = list(buffer["put_iv"])

        iv_curr = iv_buf_entry[-1] if iv_buf_entry else None
        iv_mean = float(np.mean(iv_buf_entry[-20:])) if len(iv_buf_entry) >= 20 else None

        sl_price, sl_method = compute_sl(exec_entry, atr_val, iv_curr, iv_mean)

        atm_call_feed_id = str(INSTRUMENTS["atm_call"][1]).strip()
        atm_put_feed_id  = str(INSTRUMENTS["atm_put"][1]).strip()
        chosen_feed_id   = str(best_dhan_id).strip() if best_dhan_id is not None else ""

        if filtered_signal == "LONG" and chosen_feed_id == atm_call_feed_id:
            log.warning(
                f"NOTE same_token_as_atm_call symbol={best_symbol} "
                f"dhan_id={chosen_feed_id} atm_call_id={atm_call_feed_id} -> taking trade anyway"
            )

        if filtered_signal == "SHORT" and chosen_feed_id == atm_put_feed_id:
            log.warning(
                f"NOTE same_token_as_atm_put symbol={best_symbol} "
                f"dhan_id={chosen_feed_id} atm_put_id={atm_put_feed_id} -> taking trade anyway"
            )

        # ── 15. Place entry ───────────────────────────────────────────────
        if order_manager is not None:
            try:
                active_trade = await order_manager.place_entry(
                    signal=filtered_signal,
                    snap_price=exec_entry,
                    sl_price=sl_price,
                    conf=confidence,
                    agree=mscore,
                    method=sl_method,
                    bar_dt=bar_time,
                    bar_count=_bar_count,
                    cooldown_remaining=_cooldown_remaining,
                    tradingsymbol=best_symbol,
                    inst_token=best_token_confirmed,
                    lot_size=best["lot_size"],
                    lots=best["lots"],
                    total_cost=best["total_cost"],
                )
                if active_trade is None:
                    log.warning(
                        f"ENTRY REJECTED by order_manager signal={filtered_signal} "
                        f"conf={confidence:.3f} symbol={best_symbol}"
                    )
                    return
            except Exception as e:
                log.error(
                    f"ENTRY EXCEPTION signal={filtered_signal} "
                    f"symbol={best_symbol} err={e}",
                    exc_info=True
                )
                active_trade = None
                return

            if filtered_signal == "LONG" and best_token_confirmed:
                old_token = INSTRUMENTS["otm_call"][1]
                old_symbol = INSTRUMENTS["otm_call"][0]

                INSTRUMENTS["otm_call"] = (best_symbol, str(best_dhan_id))
                INSTRUMENT_STRIKES["otm_call"] = exec_strike   # ← ADD THIS LINE
                _ACTIVE_TRADE_FEED_ID = str(best_dhan_id).strip()
                _ACTIVE_TRADE_FEED_SYMBOL = str(best_dhan_symbol or best_symbol)
                _LAST_TRADE_PRICE = None
                _LAST_TRADE_TICK_TS = None

                log.info(
                    f"INSTRUMENTS updated: otm_call → {best_symbol} "
                    f"dhan_id={best_dhan_id} "
                    f"dhan_symbol={best_dhan_symbol} "
                    f"dhan_expiry={best_dhan_expiry} "
                    f"trade_feed_id={_ACTIVE_TRADE_FEED_ID} "
                    f"(zerodha_token={best_token_confirmed} used for order only)"
                )

                feed_resubscribe(
                    new_token=str(best_dhan_id),
                    new_symbol=str(best_dhan_symbol or best_symbol),
                    old_token=str(old_token),
                    old_symbol=str(old_symbol),
                    direction="ENTRY LONG",
                )

            elif filtered_signal == "SHORT" and best_token_confirmed:
                old_token = INSTRUMENTS["otm_put"][1]
                old_symbol = INSTRUMENTS["otm_put"][0]

                INSTRUMENTS["otm_put"] = (best_symbol, str(best_dhan_id))
                INSTRUMENT_STRIKES["otm_put"] = exec_strike   # ← ADD THIS LINE
                _ACTIVE_TRADE_FEED_ID = str(best_dhan_id).strip()
                _ACTIVE_TRADE_FEED_SYMBOL = str(best_dhan_symbol or best_symbol)
                _LAST_TRADE_PRICE = None
                _LAST_TRADE_TICK_TS = None

                log.info(
                    f"INSTRUMENTS updated: otm_put → {best_symbol} "
                    f"dhan_id={best_dhan_id} "
                    f"dhan_symbol={best_dhan_symbol} "
                    f"dhan_expiry={best_dhan_expiry} "
                    f"trade_feed_id={_ACTIVE_TRADE_FEED_ID} "
                    f"(zerodha_token={best_token_confirmed} used for order only)"
                )

                feed_resubscribe(
                    new_token=str(best_dhan_id),
                    new_symbol=str(best_dhan_symbol or best_symbol),
                    old_token=str(old_token),
                    old_symbol=str(old_symbol),
                    direction="ENTRY SHORT",
                )

            tracker.log_trade(
                bar_time,
                filtered_signal,
                active_trade.get("entry", exec_entry),
                None,
                None,
                result.get("prob_long", 0.5),
                result.get("prob_short", 0.5),
            )

        else:
            active_trade = {
                "signal": filtered_signal,
                "entry": exec_entry,
                "sl": sl_price,
                "sl_method": sl_method,
                "strike": int(exec_strike),
                "side": _side,
                "bar_held": 0,
                "prob_long": result.get("prob_long", 0.5),
                "prob_short": result.get("prob_short", 0.5),
            }
            tracker.log_trade(
                bar_time, filtered_signal, exec_entry, None, None,
                result.get("prob_long", 0.5),
                result.get("prob_short", 0.5),
            )

        # (A) ADX audit line — makes the trend gate verifiable per entry.
        # adx_1m is computed in _build_features on the ATM CALL close series
        # (v12_inference.py). We also show the spot move that the reversal guard
        # evaluated, so a mistimed entry is diagnosable straight from the log.
        try:
            _rg = list(buffer["spot_close"])[-(REVERSAL_GUARD_BARS + 1):]
            _rg_chg = (_rg[-1] - _rg[0]) / (_rg[0] + 1e-9) * 100.0 if len(_rg) >= 2 else 0.0
        except Exception:
            _rg_chg = 0.0
        log.info(
            f"ENTRY-GATE adx_1m={feat.get('adx_1m', 0):.2f} "
            f"pdi={feat.get('pdi_1m', 0):.1f} ndi={feat.get('ndi_1m', 0):.1f} "
            f"adx_src=call_close nbars={len(buffer['call_close'])} "
            f"spot_chg_{REVERSAL_GUARD_BARS}bar={_rg_chg:+.2f}% "
            f"revguard={'on' if REVERSAL_GUARD_ENABLED else 'off'}"
        )

        atr_log = f"ATR={atr_val:.2f}" if atr_val else "ATR=NA(fallback)"
        log.info(
            f"ENTRY {filtered_signal:<5} {exec_entry:8.2f} "
            f"symbol={best_symbol} "
            f"PL={result.get('prob_long', 0):.3f} "
            f"PS={result.get('prob_short', 0):.3f} "
            f"conf={confidence:.3f} SL={sl_price:.2f} "
            f"momscore={mscore} {atr_log} "
            f"OTM={OTM_OFFSET} ATM={atm_strike} strike={best.get('strike_price', exec_strike)} "
            f"cost=Rs{best['total_cost']:.0f} dist={best['distance_pts']:.0f}pts "
            f"OI={best['oi']:,} filter={filter_reason}"
        )
        _last_signal_dir = filtered_signal

    finally:
        _bar_processing = False
                                             
# ══════════════════════════════════════════════════════════════════════
# VIX FETCH
# ══════════════════════════════════════════════════════════════════════
def fetch_india_vix(retries=5, base_delay=2.0):
    url = "https://www.nseindia.com/api/allIndices"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept"    : "application/json",
        "Referer"   : "https://www.nseindia.com/",
    }
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as r:
                data = json.loads(r.read())
            for item in data.get("data", []):
                if item.get("index") == "INDIA VIX":
                    vix = float(item["last"])
                    log.info(f"India VIX from NSE: {vix:.2f} (attempt {attempt + 1})")
                    return vix
            log.warning("VIX fetch: INDIA VIX not present in NSE response")
        except Exception as e:
            log.warning(f"VIX fetch attempt {attempt + 1}/{retries} failed: {e}")
        if attempt < retries - 1:
            delay = base_delay * (2 ** attempt)
            log.info(f"VIX fetch retrying in {delay:.0f}s...")
            time_mod.sleep(delay)
    log.error("VIX fetch FAILED after all retries — using default 16.0 "
              "(vix_ma20 / vix_above_ma features will be degraded)")
    return 16.0


# ══════════════════════════════════════════════════════════════════════
# HISTORICAL VIX BACKFILL  (fallback when live_warmstart.fetch_historical_vix
# gets HTTP 503 from NSE and constant-fills india_vix for the whole buffer)
# ══════════════════════════════════════════════════════════════════════
KITE_INDIA_VIX_TOKEN = 264969   # Zerodha instrument_token for NSE "INDIA VIX"
DHAN_INDIA_VIX_ID    = "21"     # Dhan security_id for INDIA VIX (segment IDX_I)


def _vix_is_constant(vix_buf, min_len=20):
    """True when the whole india_vix column is one repeated value (= constant-filled)."""
    try:
        arr = np.asarray(list(vix_buf), dtype=float)
    except Exception:
        return False
    return len(arr) >= min_len and float(np.nanstd(arr)) < 1e-9


def _to_naive_ist(ts):
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("Asia/Kolkata").tz_localize(None)
    return ts


def _fetch_vix_minutes_kite(start, end):
    """1-minute India VIX closes from Zerodha. Returns a minute-indexed Series or None."""
    if kite is None:
        return None
    rows = []
    end_dt = end + pd.Timedelta(hours=23, minutes=59)
    cur = start
    while cur < end_dt:
        nxt = min(cur + pd.Timedelta(days=25), end_dt)
        data = kite.historical_data(KITE_INDIA_VIX_TOKEN,
                                    cur.to_pydatetime(), nxt.to_pydatetime(), "minute")
        rows.extend(data or [])
        cur = nxt + pd.Timedelta(seconds=1)
    if not rows:
        return None
    idx = pd.DatetimeIndex([_to_naive_ist(r["date"]).floor("min") for r in rows])
    s = pd.Series([float(r["close"]) for r in rows], index=idx)
    return s[~s.index.duplicated(keep="last")].sort_index()


def _fetch_vix_minutes_dhan(start, end):
    """1-minute India VIX closes from Dhan. Returns a minute-indexed Series or None."""
    from dhanhq import DhanContext, dhanhq as DhanHQ
    client = DhanHQ(DhanContext(CLIENT_ID, ACCESS_TOKEN))
    closes, stamps = [], []
    cur = start
    while cur <= end:
        nxt = min(cur + pd.Timedelta(days=20), end)
        resp = client.intraday_minute_data(
            security_id=DHAN_INDIA_VIX_ID,
            exchange_segment="IDX_I",
            instrument_type="INDEX",
            from_date=cur.strftime("%Y-%m-%d"),
            to_date=(nxt + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        )
        if isinstance(resp, dict) and resp.get("status") == "success":
            d = resp.get("data") or {}
            if isinstance(d, dict) and d.get("close") is not None and d.get("timestamp") is not None:
                closes.extend(list(d["close"]))
                stamps.extend(list(d["timestamp"]))
        else:
            log.warning(f"[VIX-BACKFILL] Dhan VIX chunk {cur.date()}..{nxt.date()} failed: "
                        f"{resp.get('remarks') if isinstance(resp, dict) else resp!r}")
        cur = nxt + pd.Timedelta(days=1)
    if not closes:
        return None

    raw = np.asarray(stamps, dtype=float)
    lo, hi = start - pd.Timedelta(days=3), end + pd.Timedelta(days=3)

    def _in_session(ix):   # share of stamps inside 09:15-15:30 IST
        tod = ix.hour * 60 + ix.minute
        return float(((tod >= 555) & (tod <= 930)).mean())

    candidates = (
        lambda a: pd.to_datetime(a, unit="s", utc=True).tz_convert("Asia/Kolkata").tz_localize(None),
        lambda a: pd.to_datetime(a, unit="s"),
        lambda a: pd.to_datetime(a + 315532800, unit="s", utc=True).tz_convert("Asia/Kolkata").tz_localize(None),
        lambda a: pd.to_datetime(a + 315532800, unit="s"),
    )
    chosen = None
    for fn in candidates:
        try:
            ix = fn(raw)
        except Exception:
            continue
        if lo <= ix[len(ix) // 2] <= hi and _in_session(ix) >= 0.9:
            chosen = ix
            break
    if chosen is None:
        log.warning("[VIX-BACKFILL] could not decode Dhan VIX timestamps — discarding")
        return None
    s = pd.Series(np.asarray(closes, dtype=float), index=chosen.floor("min"))
    return s[~s.index.duplicated(keep="last")].sort_index()


def backfill_vix_history(buf):
    """
    Rebuild buf['india_vix'] (aligned to buf['datetime']) from real 1-minute VIX
    candles. Tries Zerodha first, then Dhan. Leaves the buffer untouched on any
    failure. Returns True when the buffer was replaced.
    """
    try:
        dts = [pd.Timestamp(d) for d in buf["datetime"]]
        n = len(dts)
        if n < 20 or len(buf["india_vix"]) != n:
            log.warning(f"[VIX-BACKFILL] skipped: datetime len={n} vs "
                        f"india_vix len={len(buf['india_vix'])}")
            return False
        idx = pd.DatetimeIndex(dts).floor("min")
        start, end = idx.min().normalize(), idx.max().normalize()

        series, source = None, None
        for name, fn in (("Zerodha", _fetch_vix_minutes_kite),
                         ("Dhan",    _fetch_vix_minutes_dhan)):
            try:
                s = fn(start, end)
            except Exception as e:
                log.warning(f"[VIX-BACKFILL] {name} VIX fetch failed: {e}")
                s = None
            if s is not None and len(s) > 0:
                series, source = s, name
                break

        if series is None:
            log.warning("[VIX-BACKFILL] no VIX source worked — india_vix stays "
                        "constant-filled for the warm-started history")
            return False

        aligned = series.reindex(idx, method="ffill", tolerance=pd.Timedelta("15min"))
        coverage = float(aligned.notna().mean())
        if coverage < 0.8:
            log.warning(f"[VIX-BACKFILL] {source} VIX covers only {coverage:.0%} of the "
                        f"buffer — discarding")
            return False
        vals = aligned.ffill().bfill().to_numpy(dtype=float)
        if not (np.all(vals > 3.0) and np.all(vals < 100.0)):
            log.warning(f"[VIX-BACKFILL] {source} VIX values out of range "
                        f"[{vals.min():.2f}, {vals.max():.2f}] — discarding")
            return False

        buf["india_vix"].clear()
        buf["india_vix"].extend(vals.tolist())
        log.info(f"[VIX-BACKFILL] ✅ india_vix rebuilt from {source}: {n} bars, "
                 f"coverage={coverage:.0%}, range={vals.min():.2f}..{vals.max():.2f}, "
                 f"last={vals[-1]:.2f}")
        return True
    except Exception as e:
        log.warning(f"[VIX-BACKFILL] unexpected error: {e}", exc_info=True)
        return False


def fetch_nifty_spot_nse(retries=2, base_delay=1.0):
    url = "https://www.nseindia.com/api/allIndices"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Referer": "https://www.nseindia.com/",
    }
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as response:
                payload = json.loads(response.read())

            indices = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(indices, list):
                raise RuntimeError("NSE allIndices response has no data list")

            nifty = next(
                (item for item in indices
                 if isinstance(item, dict) and item.get("index") == "NIFTY 50"),
                None,
            )
            if nifty is None:
                raise RuntimeError("NIFTY 50 is missing from NSE allIndices response")

            spot = _validated_spot(nifty.get("last"), "NSE allIndices")
            log.info(f"NIFTY spot from NSE: {spot:.2f} (attempt {attempt + 1})")
            return spot
        except Exception as error:
            log.warning(
                f"NSE NIFTY spot fetch attempt {attempt + 1}/{retries} failed: {error}"
            )
            if attempt < retries - 1:
                time_mod.sleep(base_delay * (2 ** attempt))
    return None


# ══════════════════════════════════════════════════════════════════════
# LTT PARSER
# ══════════════════════════════════════════════════════════════════════
def _parse_ltt(ltt):
    if isinstance(ltt, (int, float)):
        return (
            pd.Timestamp(int(ltt), unit="s")
              .tz_localize("UTC")
              .tz_convert("Asia/Kolkata")
              .replace(tzinfo=None)
        )
    ts = pd.Timestamp(str(ltt))
    if ts.tzinfo is None:
        ts = ts.tz_localize("Asia/Kolkata")
    return ts.tz_convert("Asia/Kolkata").replace(tzinfo=None)

def parse_ltt(ltt):
    """
    Parse Dhan LTT (Last Trade Time) to a naive IST datetime.
    Handles both integer (epoch ms) and string formats.
    """
    if isinstance(ltt, (int, float)):
        return (
            pd.Timestamp(int(ltt), unit="s")
            .tz_localize("UTC")
            .tz_convert("Asia/Kolkata")
            .replace(tzinfo=None)
        )
    # String format fallback
    ts = pd.Timestamp(str(ltt))
    if ts.tzinfo is None:
        ts = ts.tz_localize("Asia/Kolkata")
    return ts.tz_convert("Asia/Kolkata").replace(tzinfo=None)

_mapper_ready = False
_MAIN_LOOP = None   # set in __main__ before feed.run_forever()
_UNRECOGNIZED_ID_LOGGED = set()
# ══════════════════════════════════════════════════════════════════════
# TICK PROCESSOR  — called with each dict from feed.get_data()
# ══════════════════════════════════════════════════════════════════════

def _update_bar_leg(acc_leg: dict, price: float):
    if price is None or price <= 0:
        return
    if acc_leg.get("open") is None:
        acc_leg["open"] = price
    acc_leg["close"] = price
    acc_leg["high"] = max(acc_leg.get("high", -np.inf), price)
    acc_leg["low"] = min(acc_leg.get("low", np.inf), price)


def _update_bar_leg_min(acc_leg: dict, price: float):
    if price is None or price <= 0:
        return
    acc_leg["close"] = price
    acc_leg["high"] = max(acc_leg.get("high", -np.inf), price)
    acc_leg["low"] = min(acc_leg.get("low", np.inf), price)
   
def process_tick(message):
    """
    Called synchronously in the main thread with each parsed tick dict
    from feed.get_data(). Accumulates OHLCV into bar_acc, fires
    on_bar_complete() at minute boundary using asyncio.run() — safe
    because this runs in the main thread (polling pattern).

    IMPORTANT:
    - sec_id is always Dhan security_id from MarketFeed.
    - INSTRUMENTS[*][1] must therefore also be Dhan security_id strings.
    - Zerodha instrument_token must never be compared here.
    """
    global INDIA_VIX, _mapper_ready, bar_acc
    global _ACTIVE_TRADE_FEED_ID, _ACTIVE_TRADE_FEED_SYMBOL
    global _LAST_TRADE_PRICE, _LAST_TRADE_TICK_TS
    global _RAW_TICK_LOGGED, _ROUTING_CHECK_LOGGED

    if not message or not isinstance(message, dict):
        return

    sec_id = str(message.get("security_id", "")).strip()
    ltp = float(message.get("LTP", 0) or 0)
    ltt = message.get("LTT", 0)
    volume = int(message.get("volume", 0) or 0)
    oi = int(message.get("OI", 0) or 0)

    if not sec_id:
        return

    if ltp <= 0 or not ltt:
        return

    tracked_ids = set()
    for value in INSTRUMENTS.values():
        if isinstance(value, (tuple, list)) and len(value) >= 2:
            tracked_ids.add(str(value[1]).strip())
    if GRID is not None:
        tracked_ids |= {str(sid).strip() for sid in GRID.ids}
    if _ACTIVE_TRADE_FEED_ID is not None:
        tracked_ids.add(str(_ACTIVE_TRADE_FEED_ID).strip())

    if sec_id not in tracked_ids:
        if len(_UNRECOGNIZED_ID_LOGGED) > 200:
            _UNRECOGNIZED_ID_LOGGED.clear()
        if sec_id not in _UNRECOGNIZED_ID_LOGGED:
            log.warning(
                "[FEED] Packet ignored for untracked security_id=%s; "
                "expected ids=%s (active trade=%s, grid=%s)",
                sec_id,
                sorted(tracked_ids)[:20],
                str(_ACTIVE_TRADE_FEED_ID).strip() if _ACTIVE_TRADE_FEED_ID is not None else None,
                len(GRID.ids) if GRID is not None else 0,
            )
            _UNRECOGNIZED_ID_LOGGED.add(sec_id)
        return

    try:
        tick_time = parse_ltt(ltt)
    except Exception:
        return

    bar_min = tick_time.replace(second=0, microsecond=0)

    call_id = str(INSTRUMENTS["atm_call"][1]).strip()
    put_id = str(INSTRUMENTS["atm_put"][1]).strip()
    spot_id = str(INSTRUMENTS["nifty_spot"][1]).strip()
    otm_call_id = str(INSTRUMENTS["otm_call"][1]).strip()
    otm_put_id = str(INSTRUMENTS["otm_put"][1]).strip()
    trade_id = str(_ACTIVE_TRADE_FEED_ID).strip() if _ACTIVE_TRADE_FEED_ID is not None else ""

    # ── ONE-TIME DIAG: dump raw tick keys for the ATM call leg ──────────
    # Confirms whether Dhan's Full packet actually includes an IV field.
    if not _RAW_TICK_LOGGED and sec_id == call_id:
        log.info(f"[DIAG] RAW TICK keys (call leg): {list(message.keys())}")
        log.info(f"[DIAG] RAW TICK full payload (call leg): {message}")
        _RAW_TICK_LOGGED = True

    # ── ONE-TIME DIAG: confirm trade_id routing matches otm ids on first
    #    tick after an entry (only fires once per active trade) ─────────
    if (
        not _ROUTING_CHECK_LOGGED
        and trade_id
        and sec_id in (trade_id, otm_call_id, otm_put_id)
    ):
        log.info(
            f"[DIAG] routing check: sec_id={sec_id!r} trade_id={trade_id!r} "
            f"otm_call_id={otm_call_id!r} otm_put_id={otm_put_id!r} "
            f"match_trade={sec_id == trade_id} "
            f"match_otm_call={sec_id == otm_call_id} "
            f"match_otm_put={sec_id == otm_put_id}"
        )
        _ROUTING_CHECK_LOGGED = True

    def update_ohlcv(side, price, vol, oi_val):
        a = bar_acc[side]
        if a["open"] is None:
            a["open"] = price
        a["high"] = max(a["high"], price)
        a["low"] = min(a["low"], price)
        a["close"] = price
        # A1: `vol` is the feed's CUMULATIVE day volume, not per-tick volume.
        # Store it; convert to true per-BAR volume at bar flush (per-bar delta).
        a["volume"] = vol
        a["oi"] = oi_val

    def update_exec_only(side, price):
        a = bar_acc[side]
        if a["high"] == -np.inf:
            a["high"] = price
        else:
            a["high"] = max(a["high"], price)

        if a["low"] == np.inf:
            a["low"] = price
        else:
            a["low"] = min(a["low"], price)

        a["close"] = price

    # Ensure patched trade bucket exists even if _reset_bar_acc() is older
    if "trade" not in bar_acc:
        bar_acc["trade"] = {
            "open": None,
            "high": -np.inf,
            "low": np.inf,
            "close": None,
            "volume": 0,
            "oi": 0,
        }

    # ── Initialize bar minute first ───────────────────────────────────
    if bar_acc["bar_minute"] is None:
        bar_acc["bar_minute"] = bar_min

    # ── Minute boundary check BEFORE consuming current tick into new bar ──
    if bar_min > bar_acc["bar_minute"]:
        completed = bar_acc["bar_minute"]

        spot_ok = (bar_acc["spot"]["close"] or 0) > 0
        call_ok = (bar_acc["call"]["close"] or 0) > 0
        put_ok = (bar_acc["put"]["close"] or 0) > 0

        if spot_ok and (GRID is not None or call_ok or put_ok):

            # ── One-time mapper readiness check (fires on bar 1 only) ──
            if not _mapper_ready:
                if mapper is not None:
                    try:
                        mapper.load()
                        log.info(
                            "process_tick: mapper reloaded for "
                            "process_tick scope (late init)"
                        )
                    except Exception as _me:
                        log.error(
                            f"process_tick: mapper.load() failed "
                            f"in late init: {_me}"
                        )
                _mapper_ready = True

            try:
                asyncio.run(
                    on_bar_complete(
                        completed,
                        india_vix=INDIA_VIX,
                        _mapper=mapper,
                    )
                )
            except Exception as exc:
                log.error(
                    f"on_bar_complete EXCEPTION for {completed}: {exc}",
                    exc_info=True,
                )
            _grid_recenter()
        else:
            log.debug(
                f"SKIP incomplete bar {completed} "
                f"spot={bar_acc['spot']['close']} "
                f"call={bar_acc['call']['close']} "
                f"put={bar_acc['put']['close']}"
            )

        # reset old bar completely, then start a fresh bar for current minute
        bar_acc = _reset_bar_acc()

        # make sure trade bucket survives reset
        if "trade" not in bar_acc:
            bar_acc["trade"] = {
                "open": None,
                "high": -np.inf,
                "low": np.inf,
                "close": None,
                "volume": 0,
                "oi": 0,
            }

        bar_acc["bar_minute"] = bar_min

    if GRID is not None:
        GRID.on_tick(sec_id, ltp, volume, oi)

    # ── First: always capture active traded instrument tick ───────────
    if trade_id and sec_id == trade_id:
        update_ohlcv("trade", ltp, volume, oi)
        _LAST_TRADE_PRICE = ltp
        _LAST_TRADE_TICK_TS = tick_time

    # ── Now consume current tick into named buckets ───────────────────
    if sec_id == call_id:
        update_ohlcv("call", ltp, volume, oi)
        spot_now = bar_acc["spot"]["close"]
        strike_now = INSTRUMENT_STRIKES.get("atm_call")
        iv_val = None
        if spot_now and strike_now and _EXPIRY_DATE_OBJ:
            iv_val = compute_live_iv(ltp, spot_now, strike_now, _EXPIRY_DATE_OBJ, 'c')
        bar_acc["call"]["iv"] = iv_val if iv_val is not None else (bar_acc["call"]["iv"] or 18.0)

    elif sec_id == put_id:
        update_ohlcv("put", ltp, volume, oi)
        spot_now = bar_acc["spot"]["close"]
        strike_now = INSTRUMENT_STRIKES.get("atm_put")
        iv_val = None
        if spot_now and strike_now and _EXPIRY_DATE_OBJ:
            iv_val = compute_live_iv(ltp, spot_now, strike_now, _EXPIRY_DATE_OBJ, 'p')
        bar_acc["put"]["iv"] = iv_val if iv_val is not None else (bar_acc["put"]["iv"] or 19.0)

    elif sec_id == spot_id:
        bar_acc["spot"]["close"] = ltp

    elif sec_id == otm_call_id:
        update_exec_only("otm_call", ltp)

    elif sec_id == otm_put_id:
        update_exec_only("otm_put", ltp)

    else:
        return


def _flush_idle_minute():
    """Close an accumulated minute if the feed is quiet across its boundary."""
    global bar_acc
    if bar_acc["bar_minute"] is None or not (bar_acc["spot"]["close"] or 0) > 0:
        return False

    now = pd.Timestamp.now(tz="Asia/Kolkata")
    current_minute = now.tz_localize(None).replace(second=0, microsecond=0)
    if current_minute <= bar_acc["bar_minute"]:
        return False

    process_tick({
        "security_id": str(INSTRUMENTS["nifty_spot"][1]),
        "LTP": bar_acc["spot"]["close"],
        "LTT": now.strftime("%H:%M:%S"),
        "volume": 0,
        "OI": 0,
    })

    # The synthetic tick above is only a boundary signal. Start the new bar
    # empty so stale prices cannot be reused as live market data.
    bar_acc = _reset_bar_acc()
    if "trade" not in bar_acc:
        bar_acc["trade"] = {
            "open": None,
            "high": -np.inf,
            "low": np.inf,
            "close": None,
            "volume": 0,
            "oi": 0,
        }
    bar_acc["bar_minute"] = current_minute
    return True

           
# ══════════════════════════════════════════════════════════════════════
# FEED FACTORY
# ══════════════════════════════════════════════════════════════════════
def build_feed(subscription):
    global FEED

    try:
        from dhanhq import DhanContext, MarketFeed as _MF
        feed = mf.MarketFeed(
            DhanContext(CLIENT_ID, ACCESS_TOKEN),
            subscription,
            version="v2",
        )
        log.info("Using MarketFeed (dhanhq v2.1+ API)")
        FEED = feed   # ✅ store globally so feed_resubscribe() can access it

        # ── one-time method availability check ──
        log.info(f"FEED has subscribe_symbols: {hasattr(FEED, 'subscribe_symbols')}")
        log.info(f"FEED has unsubscribe_symbols: {hasattr(FEED, 'unsubscribe_symbols')}")

        return feed, "market_feed"
    except (ImportError, AttributeError):
        pass

    sig    = inspect.signature(mf.DhanFeed.__init__)
    params = list(sig.parameters.keys())
    log.info(f"   Falling back to DhanFeed  params={params}")

    param_map = {
        "client_id"   : CLIENT_ID,
        "access_token": ACCESS_TOKEN,
        "instruments" : subscription,
        "version"     : "v2",
    }
    kwargs = {p: param_map[p] for p in params if p in param_map}
    feed   = mf.DhanFeed(**kwargs)
    log.info("Using DhanFeed (legacy polling)")
    FEED = feed   # ✅ store globally so feed_resubscribe() can access it

    # ── one-time method availability check ──
    log.info(f"FEED has subscribe_symbols: {hasattr(FEED, 'subscribe_symbols')}")
    log.info(f"FEED has unsubscribe_symbols: {hasattr(FEED, 'unsubscribe_symbols')}")

    return feed, "dhan_feed"


# ══════════════════════════════════════════════════════════════════════
# INSTRUMENT RESOLVER
# ══════════════════════════════════════════════════════════════════════
def resolve_instruments(kite_or_none=None):
    """
    Auto-resolve Dhan security IDs for the NEXT weekly expiry.
    NIFTY 50 expiry is Tuesday (changed from Thursday in Sep 2025).
    Downloads scrip master, finds ATM +/- 50pt strikes.
    Falls back to hardcoded IDs if download fails.

    Also stores the weekly expiry's scrip rows in the module-level
    _WEEKLY_DF so ChainGrid can resolve the far-strike security IDs
    without a second scrip-master download.

    Spot price priority:
        1. Zerodha kite.ltp()            — already authenticated, fastest
        2. Dhan intraday_minute_data     — latest candle close
        3. NSE allIndices                — independent market-data fallback
        4. Zerodha kite.quote()          — different endpoint, broader coverage
    """
    global _WEEKLY_DF, _WEEKLY_EXPIRY
    import pandas as pd
    from datetime import date

    _WEEKLY_DF = None
    _WEEKLY_EXPIRY = None

    # ── Local fallback — avoids circular reference with module-level INSTRUMENTS ──
    _FALLBACK = {
        "nifty_spot": ("IDX_I",   "13"),
        "atm_call":   ("NSE_FNO", "74155"),  # NIFTY-May2026-24050-CE
        "atm_put":    ("NSE_FNO", "74156"),  # NIFTY-May2026-24050-PE
        "otm_call":   ("NSE_FNO", "74165"),  # NIFTY-May2026-24150-CE
        "otm_put":    ("NSE_FNO", "74146"),  # NIFTY-May2026-23950-PE
    }

    try:
        df = pd.read_csv(
            "https://images.dhan.co/api-data/api-scrip-master.csv",
            low_memory=False
        )
        nifty = df[
            df["SEM_TRADING_SYMBOL"].str.startswith("NIFTY") &
            df["SEM_INSTRUMENT_NAME"].isin(["OPTIDX"])
        ].copy()
        nifty["SEM_EXPIRY_DATE"] = pd.to_datetime(nifty["SEM_EXPIRY_DATE"])

        # Skip today + tomorrow → land on next Tuesday expiry
        today       = pd.Timestamp(date.today())
        skip_before = today + pd.Timedelta(days=2)
        future      = nifty[nifty["SEM_EXPIRY_DATE"] >= skip_before]
        next_exp    = future["SEM_EXPIRY_DATE"].min()
        weekly      = nifty[nifty["SEM_EXPIRY_DATE"] == next_exp]

        # Keep the weekly rows for ChainGrid.
        _WEEKLY_DF = weekly.copy()
        _WEEKLY_EXPIRY = next_exp.date()

        # Sanity check: must be Tuesday
        if next_exp.weekday() != 1:
            log.warning(
                f"resolve_instruments: next_exp {next_exp.date()} "
                f"is NOT a Tuesday (weekday={next_exp.weekday()}). "
                f"Check NSE holiday schedule."
            )

        # ── Spot price — 4-level priority chain ──────────────────────────────
        spot = None

        # ── Priority 1: Zerodha kite.ltp() — already authenticated in the bot ──
        if kite_or_none is not None:
            try:
                ltp_data = kite_or_none.ltp(["NSE:NIFTY 50"])
                spot = _validated_spot(
                    ltp_data["NSE:NIFTY 50"]["last_price"], "Zerodha kite.ltp"
                )
                log.info(f"resolve_instruments: spot from Zerodha kite.ltp = {spot:.0f}")
            except Exception as kite_ltp_err:
                log.warning(
                    f"resolve_instruments: kite.ltp failed ({kite_ltp_err}) "
                    "— trying Dhan intraday"
                )

        # ── Priority 2: Dhan intraday_minute_data ────────────────────────────
        if spot is None:
            try:
                from dhanhq import DhanContext, dhanhq as DhanHQ
                client = DhanHQ(DhanContext(CLIENT_ID, ACCESS_TOKEN))
                today_str = date.today().strftime("%Y-%m-%d")
                spot_data = client.intraday_minute_data(
                    security_id      = "13",
                    exchange_segment = "IDX_I",
                    instrument_type  = "INDEX",
                    from_date        = today_str,
                    to_date          = today_str,
                )

                if not isinstance(spot_data, dict) or spot_data.get("status") != "success":
                    raise RuntimeError(
                        f"Dhan intraday API returned failure: "
                        f"{spot_data.get('remarks') if isinstance(spot_data, dict) else spot_data!r}"
                    )

                candles = spot_data.get("data")
                if isinstance(candles, dict):
                    closes = candles.get("close")
                    if isinstance(closes, (list, tuple, np.ndarray)):
                        if len(closes) == 0:
                            raise RuntimeError("Dhan intraday response has no close prices")
                        spot = float(closes[-1])
                    elif closes is not None:
                        spot = float(closes)
                    else:
                        raise RuntimeError(
                            f"Dhan intraday response is missing close prices: "
                            f"{list(candles.keys())}"
                        )
                elif isinstance(candles, list) and candles:
                    spot = float(candles[-1]["close"])
                else:
                    raise RuntimeError(
                        f"Unexpected Dhan intraday response data: {candles!r}"
                    )

                if not np.isfinite(spot) or spot <= 0:
                    raise RuntimeError(f"Dhan returned invalid NIFTY spot price: {spot}")
                log.info(f"resolve_instruments: spot from Dhan intraday = {spot:.0f}")
            except Exception as dhan_err:
                log.warning(
                    f"resolve_instruments: Dhan spot fetch failed ({dhan_err}) "
                    "— trying NSE spot fallback"
                )

        # ── Priority 3: NSE allIndices — independent market-data fallback ────
        if spot is None:
            spot = fetch_nifty_spot_nse()
            if spot is not None:
                log.info(f"resolve_instruments: spot from NSE = {spot:.0f}")

        # ── Priority 4: Zerodha kite.quote() — different endpoint, broader coverage ──
        if spot is None and kite_or_none is not None:
            try:
                quote_data = kite_or_none.quote(["NSE:NIFTY 50"])
                spot = _validated_spot(
                    quote_data["NSE:NIFTY 50"]["last_price"], "Zerodha kite.quote"
                )
                log.info(f"resolve_instruments: spot from Zerodha kite.quote = {spot:.0f}")
            except Exception as kite_quote_err:
                log.warning(
                    f"resolve_instruments: kite.quote failed ({kite_quote_err}) "
                    "— instrument resolution will abort if no valid spot is available"
                )

        # ── Never calculate ATM from a stale guessed spot ────────────────────
        if spot is None:
            log.error(
                "resolve_instruments: all spot fetch methods failed; "
                "aborting instrument resolution to avoid selecting the wrong ATM. "
                "Verify Dhan/Kite market-data permissions and credentials."
            )
            raise _SpotPriceUnavailable(
                "Could not fetch a valid NIFTY spot price from Kite, Dhan, or NSE"
            )

        atm = round(spot / 50) * 50

        def _get_id(strike, opt_type):
            row = weekly[
                (weekly["SEM_STRIKE_PRICE"] == float(strike)) &
                (weekly["SEM_OPTION_TYPE"]  == opt_type)
            ]
            if not len(row):
                log.error(
                    f"resolve_instruments: NO ID found for "
                    f"strike={strike} type={opt_type} expiry={next_exp.date()}"
                )
                return None
            sid = str(int(row["SEM_SMST_SECURITY_ID"].values[0]))
            sym = row["SEM_TRADING_SYMBOL"].values[0]
            log.info(f"  {sym:<35} → ID={sid}")
            return sid

        log.info(
            f"resolve_instruments  spot={spot:.0f}  ATM={atm}  "
            f"expiry={next_exp.date()}  ({next_exp.day_name()})"
        )

        ids = {
            "nifty_spot" : ("IDX_I",   "13"),
            "atm_call"   : ("NSE_FNO", _get_id(atm,      "CE")),
            "atm_put"    : ("NSE_FNO", _get_id(atm,      "PE")),
            "otm_call"   : ("NSE_FNO", _get_id(atm + 50, "CE")),
            "otm_put"    : ("NSE_FNO", _get_id(atm - 50, "PE")),
        }

        # Guard: if any ID missing, use local fallback (not module-level INSTRUMENTS)
        if any(v[1] is None for v in ids.values()):
            log.warning(
                "resolve_instruments: one or more IDs missing — "
                "falling back to hardcoded _FALLBACK"
            )
            _WEEKLY_DF = None
            _WEEKLY_EXPIRY = None
            return _FALLBACK

        # ── keep strike tracking in sync with resolved instruments ──
        global INSTRUMENT_STRIKES
        INSTRUMENT_STRIKES["atm_call"] = atm
        INSTRUMENT_STRIKES["atm_put"]  = atm
        INSTRUMENT_STRIKES["otm_call"] = atm + 50
        INSTRUMENT_STRIKES["otm_put"]  = atm - 50

        log.info(f"resolve_instruments SUCCESS  IDs={ids}")
        return ids

    except _SpotPriceUnavailable:
        _WEEKLY_DF = None
        _WEEKLY_EXPIRY = None
        raise
    except Exception as e:
        _WEEKLY_DF = None
        _WEEKLY_EXPIRY = None
        log.warning(f"resolve_instruments failed: {e} — using hardcoded fallback")
        return _FALLBACK   # ← local dict, guaranteed to have all 5 keys
    
def repin_instruments_to_week_pin(pin):
    """
    Re-point atm_call/atm_put (and default otm legs) at the week's PINNED strike so the
    live series continues the warm-start history with no strike-level jump at the seam.
    Uses _WEEKLY_DF captured by resolve_instruments(). Returns True on success.
    """
    if _WEEKLY_DF is None or pin is None:
        log.warning("repin: _WEEKLY_DF or pin unavailable — keeping startup-ATM instruments")
        return False

    def _id(strike, opt):
        row = _WEEKLY_DF[(_WEEKLY_DF["SEM_STRIKE_PRICE"] == float(strike)) &
                         (_WEEKLY_DF["SEM_OPTION_TYPE"] == opt)]
        return str(int(row["SEM_SMST_SECURITY_ID"].values[0])) if len(row) else None

    new = {
        "atm_call": _id(pin,      "CE"),
        "atm_put" : _id(pin,      "PE"),
        "otm_call": _id(pin + 50, "CE"),
        "otm_put" : _id(pin - 50, "PE"),
    }
    if any(v is None for v in new.values()):
        log.error(f"repin: missing scrip IDs for pin={pin} {new} — keeping startup-ATM instruments")
        return False

    for k, v in new.items():
        INSTRUMENTS[k] = ("NSE_FNO", v)
    INSTRUMENT_STRIKES["atm_call"] = pin
    INSTRUMENT_STRIKES["atm_put"]  = pin
    INSTRUMENT_STRIKES["otm_call"] = pin + 50
    INSTRUMENT_STRIKES["otm_put"]  = pin - 50
    log.info(f"repin: atm/otm legs re-pointed to week pin {pin}  ids={new}")
    return True
   
# ══════════════════════════════════════════════════════════════════════
# MAIN  — Official Dhan polling pattern: run_forever() + get_data()
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import random
    import warnings
    from websockets.exceptions import InvalidStatus

    warnings.filterwarnings("ignore", message="coroutine.*GeneratorExit")

    INDIA_VIX = fetch_india_vix()

    # ── Zerodha init ──────────────────────────────────────────────────
    kite         = get_kite()
    kite.timeout = 30

    _load_attempts = 0
    mapper = InstrumentMapper(kite)
    while True:
        try:
            mapper.load()
            log.info("mapper.load() ✅ Dhan scrip master loaded")
            break
        except Exception as _e:
            _load_attempts += 1
            if _load_attempts >= 5:
                log.error(f"mapper.load() failed {_load_attempts} times — giving up: {_e}")
                raise
            log.warning(f"mapper.load() attempt {_load_attempts} failed ({_e}) — retrying in 5s...")
            time_mod.sleep(5)

    _expiry_attempts = 0
    expiry = None
    while expiry is None:
        try:
            expiry = get_current_expiry(kite)
        except Exception as _e:
            _expiry_attempts += 1
            if _expiry_attempts >= 5:
                log.error(f"get_current_expiry failed {_expiry_attempts} times — giving up: {_e}")
                raise
            log.warning(f"get_current_expiry attempt {_expiry_attempts} failed ({_e}) — retrying in 5s...")
            time_mod.sleep(5)

    # ── parse expiry into a date object for compute_live_iv() ──
    if isinstance(expiry, str):
        _EXPIRY_DATE_OBJ = datetime.strptime(expiry, "%Y-%m-%d").date()
    elif isinstance(expiry, datetime):
        _EXPIRY_DATE_OBJ = expiry.date()
    else:
        _EXPIRY_DATE_OBJ = expiry  # already a date object
    log.info(f"_EXPIRY_DATE_OBJ set to {_EXPIRY_DATE_OBJ} for live IV computation")

    # ══════════════════════════════════════════════════════════════════
    # PAPER TRADING MODE — no real orders are ever placed.
    #
    # order_manager is intentionally kept as None (its module-level default).
    # Every code path in on_bar_complete() already branches on
    # `if order_manager is not None:` for entries/exits/capital/broker-sync,
    # with a matching `else:` branch that simulates the trade using live
    # buffer/tick prices instead. Leaving order_manager (and syncer) unset
    # here simply routes 100% of execution through those existing paper
    # branches — no other logic (model, gates, SL, smart-exit chain,
    # tracker) is touched.
    #
    # RiskGate / OrderManager / PositionSync are NOT instantiated, so no
    # broker connection for order placement or position sync is made, and
    # no real capital is ever checked or consumed. `kite` stays connected
    # only for market-data use (spot LTP in resolve_instruments, etc.).
    # ══════════════════════════════════════════════════════════════════
    risk_gate     = None
    order_manager = None
    syncer        = None
    log.info("PAPER TRADING MODE — order_manager/risk_gate/syncer disabled. "
              "No real orders will be placed; trades are simulated and "
              "monitored against the smart-exit (SL / IV / OI / Supertrend / "
              "time-stop) chain only.")

    log.info(f"Zerodha broker connection used for MARKET DATA ONLY  expiry={expiry}  "
              f"dry_run={ZERODHA_CONFIG['dry_run']}  paper_mode=True")

    INSTRUMENTS = resolve_instruments(kite_or_none=kite)

    # ── Warm-start buffer + model state before the feed connects, so the
    # bot predicts meaningfully from bar 1 instead of needing ~20 trading
    # days of live accumulation. The dynamic ATM grid is seeded from the
    # current weekly expiry and does not depend on a pinned week-strike.
    from live_warmstart import (
        warmstart_live_bot,
        fetch_pinned_call_history,
        LAST_PIN_STRIKES,
    )
    from buffer_warmstart import build_extended_buffer
    log.info("Warm-starting model buffer from recent dynamic-ATM history...")
    _warm_buf = warmstart_live_bot(
        build_extended_buffer, model, CLIENT_ID, ACCESS_TOKEN,
        india_vix_default=INDIA_VIX, log=log,
    )
    if _warm_buf is not None:
        buffer = _warm_buf
        _WARMSTART_USED = True
        log.info(f"Warm-start SUCCESS: buffer at {len(buffer['call_close'])}/{BUFFER_SIZE} bars")
        log.info(f"Entries blocked for the next {POST_WARMSTART_COOLDOWN_BARS} live bars "
                f"(seam-artifact guard)")

        # ── VIX BACKFILL: live_warmstart constant-fills india_vix with today's
        # value whenever NSE's historical-VIX endpoint fails (HTTP 503). Rebuild
        # it from real 1-minute VIX candles so vix_ma20 / vix_above_ma /
        # vix_above_rolling_median match what the model was trained on.
        if _vix_is_constant(buffer["india_vix"]):
            log.warning("[VIX-BACKFILL] india_vix is constant across the warm-started "
                        "buffer (NSE historical VIX failed) — rebuilding from Zerodha/Dhan")
            backfill_vix_history(buffer)
        else:
            log.info("[VIX-BACKFILL] india_vix already has real history — nothing to do")
    else:
        log.warning("Warm-start FAILED -- bot starting with empty buffer, "
                    "will need ~20 trading days of live accumulation for "
                    "long-window features to become accurate")

    # ── Separate rule-only pinned CALL warm-start; ML buffer above stays dynamic.
    RULE_PIN_CLOSES.clear()
    RULE_PIN_STRIKE = None
    RULE_PIN_SECURITY_ID = None

    _rule_pin_df = fetch_pinned_call_history(
        CLIENT_ID, ACCESS_TOKEN, log=log
    )

    # Exclude the current incomplete minute from the historical seed.
    _startup_minute = (
        pd.Timestamp.now(tz="Asia/Kolkata")
        .tz_localize(None)
        .floor("min")
    )
    if _rule_pin_df is not None and not _rule_pin_df.empty:
        _rule_pin_df = _rule_pin_df[
            _rule_pin_df["datetime"] < _startup_minute
        ]
        _rule_pin_values = pd.to_numeric(
            _rule_pin_df["close"], errors="coerce"
        ).dropna()
        _rule_pin_values = _rule_pin_values[_rule_pin_values > 0]
        RULE_PIN_CLOSES.extend(
            _rule_pin_values.tail(BUFFER_SIZE).astype(float).tolist()
        )

    # LAST_PIN_STRIKES is keyed by the W-TUE period containing each
    # historical bar. Do NOT key this lookup by _WEEKLY_EXPIRY: on Monday,
    # for example, the next option expiry may be the following Tuesday and
    # therefore belongs to a different W-TUE period with no historical rows.
    # Use the current session's calendar period so a valid current-week pin
    # is selected and the rule gate can resolve its matching CALL contract.
    _pin_period = _startup_minute.to_period("W-TUE")
    _pin_value = LAST_PIN_STRIKES.get(_pin_period)
    if _pin_value is not None:
        try:
            _pin_number = float(_pin_value)
            if np.isfinite(_pin_number) and _pin_number > 0:
                RULE_PIN_STRIKE = int(round(_pin_number))
            else:
                log.error(
                    f"[RULE-PIN] Invalid strike {_pin_value!r} for current "
                    f"period {_pin_period}; trend gate will fail closed"
                )
        except (TypeError, ValueError):
            log.error(
                f"[RULE-PIN] Could not parse strike {_pin_value!r} for "
                f"current period {_pin_period}; trend gate will fail closed"
            )
    else:
        _available_pin_periods = sorted(str(k) for k in LAST_PIN_STRIKES.keys())
        log.error(
            f"[RULE-PIN] No historical pin found for current period "
            f"{_pin_period}. Available periods: {_available_pin_periods}. "
            "Trend gate will fail closed; check the historical ATM warm-start."
        )

    log.info(
        f"[RULE-PIN] selected strike={RULE_PIN_STRIKE} "
        f"historical_rows={len(RULE_PIN_CLOSES)}"
    )

    from dhanhq import DhanContext, MarketFeed as _MF

    dhan_context = DhanContext(CLIENT_ID, ACCESS_TOKEN)
    _center = int(INSTRUMENT_STRIKES["atm_call"])

    subscription = [
        (_MF.IDX,     INSTRUMENTS["nifty_spot"][1], _MF.Ticker),
        (_MF.NSE_FNO, INSTRUMENTS["atm_call"][1],   _MF.Full),
        (_MF.NSE_FNO, INSTRUMENTS["atm_put"][1],    _MF.Full),
        (_MF.NSE_FNO, INSTRUMENTS["otm_call"][1],   _MF.Full),
        (_MF.NSE_FNO, INSTRUMENTS["otm_put"][1],    _MF.Full),
    ]

    # ══════════════════════════════════════════════════════════════════
    # Build the dynamic ATM grid from the current weekly expiry.
    # ══════════════════════════════════════════════════════════════════
    if _WEEKLY_DF is not None:
        GRID = ChainGrid(model.features, _WEEKLY_DF, _WEEKLY_EXPIRY,
                         iv_fn=compute_live_iv, center_strike=_center,
                         step=STRIKE_STEP, half_width=14, recenter_at=2)

        if RULE_PIN_STRIKE is not None:
            RULE_PIN_SECURITY_ID = GRID.ensure_leg(RULE_PIN_STRIKE, "CALL")
            if RULE_PIN_SECURITY_ID is None:
                log.error(
                    f"[RULE-PIN] Could not resolve CALL contract for "
                    f"strike={RULE_PIN_STRIKE}; rule trend gate will fail closed"
                )

        GRID.seed_from_rest(CLIENT_ID, ACCESS_TOKEN)

        # ── RULE-PIN FALLBACK SEED: if the pinned-CALL warm-start history came
        # back short (e.g. Dhan 504s on the rolling-option calls), seed it from
        # the ChainGrid's own history for the SAME pinned strike. This is still
        # the pinned strike (never the dynamic ATM), so the rule gate keeps its
        # intended meaning; it just opens earlier than 175 live bars.
        if RULE_PIN_STRIKE is not None and len(RULE_PIN_CLOSES) < RULE_PIN_MIN_ROWS:
            try:
                _pin_ls = GRID.leg_series(RULE_PIN_STRIKE, "CALL")
                if _pin_ls is not None:
                    _g_closes = []
                    for _x in list(_pin_ls["close"]):
                        try:
                            _xf = float(_x)
                        except (TypeError, ValueError):
                            continue
                        if np.isfinite(_xf) and _xf > 0:
                            _g_closes.append(_xf)
                    if len(_g_closes) > len(RULE_PIN_CLOSES):
                        _old_n = len(RULE_PIN_CLOSES)
                        RULE_PIN_CLOSES.clear()
                        RULE_PIN_CLOSES.extend(_g_closes[-BUFFER_SIZE:])
                        log.warning(
                            f"[RULE-PIN] warm-start history was short ({_old_n} rows); "
                            f"re-seeded from ChainGrid leg history for strike="
                            f"{RULE_PIN_STRIKE}: {len(RULE_PIN_CLOSES)} rows"
                        )
            except Exception as _pe:
                log.warning(f"[RULE-PIN] ChainGrid fallback seed failed: {_pe}")

        if RULE_PIN_STRIKE is not None and len(RULE_PIN_CLOSES) < RULE_PIN_MIN_ROWS:
            log.warning(
                f"[RULE-PIN] only {len(RULE_PIN_CLOSES)} pinned-CALL closes available "
                f"(< {RULE_PIN_MIN_ROWS}). The 'rulepinwarmup' gate will block ALL "
                f"entries until ~{RULE_PIN_MIN_ROWS - len(RULE_PIN_CLOSES)} more live "
                f"bars have arrived. Fix the Dhan 504s (see [NET] / live_warmstart "
                f"warnings above) to avoid this wait."
            )

        known = [v[1] for v in INSTRUMENTS.values()]
        subscription += GRID.subscription_tuples(_MF, exclude_ids=known)
        log.info(f"GRID coverage after seed: {GRID.coverage()}  subscription={len(subscription)}")
        log.info(
            f"[RULE-PIN] strike={RULE_PIN_STRIKE} "
            f"security_id={RULE_PIN_SECURITY_ID}"
        )
    else:
        log.warning("GRID disabled: no weekly scrip slice")

    log.info(f"Subscription built: {subscription}")
    log.info("=" * 60)
    log.info("Starting v14 Nifty50 Live Bot — Dhan MarketFeed")
    log.info(f"India VIX     : {INDIA_VIX:.2f}")
    log.info(f"eff_long             : {model.eff_long:.4f}  (prob_long  >= this -> LONG)")
    log.info(f"eff_short            : {model.eff_short:.4f}  (prob_short >= this -> SHORT)")
    log.info(f"persist_window       : {PERSIST_WINDOW} bars")

    # v12_inference.predict_bar() only emits a signal when
    #   prob_long >= eff_long (+0.01 pre-close)   or
    #   prob_short >= eff_short (+0.02 pre-close, -0.04 if VIX > 22),
    # and confidence = max(pl, ps). So any signal reaching apply_rule_filters()
    # already has confidence >= the lowest of these thresholds.
    _model_conf_floor = min(model.eff_long, model.eff_short - 0.04)  # worst case: VIX > 22
    if MINCONF <= _model_conf_floor:
        log.info(f"MINCONF={MINCONF:.4f} -- NO-OP (model's lowest effective "
                 f"threshold {_model_conf_floor:.4f} is already >= MINCONF, so "
                 f"this rule-filter gate never rejects)")
    else:
        log.warning(f"MINCONF={MINCONF:.4f} > model's lowest effective threshold "
                    f"{_model_conf_floor:.4f} -- this rule-filter gate is ACTIVELY "
                    f"vetoing signals in the [{_model_conf_floor:.4f}, {MINCONF:.4f}) "
                    f"confidence band. Confirm this is intentional.")
    log.info(f"ATM Call      : {INSTRUMENTS['atm_call'][1]}  (features)")
    log.info(f"ATM Put       : {INSTRUMENTS['atm_put'][1]}   (features)")
    log.info(f"OTM offset    : ±{OTM_OFFSET}×{STRIKE_STEP} = ±{OTM_OFFSET*STRIKE_STEP}pts  (±500pt chain search)")
    log.info(f"Expiry        : {expiry}")
    log.info(f"Lot size      : {LOT_SIZE}  Max lots: {MAX_LOTS}")
    log.info(f"SL method     : static {SL_POINTS:.0f} points below entry")
    log.info(f"Exit system   : Smart 5-priority chain  no fixed TP")
    log.info(f"  1. Hard SL    barlow ≤ sl_price")
    log.info(f"  2. IV Z-Score z≥{IV_ZSCORE_EXIT}  AND pnl≥{IV_MIN_PROFIT_PCT}%")
    log.info(f"  3. OI Unwind  oichg≤{OI_DROP_THRESH}%  AND pnl≥{OI_MIN_PROFIT_PCT}%")
    log.info(f"  4. Supertrend period={ST_PERIOD}, mult={ST_MULT}  AND pnl≥{ST_MIN_PROFIT_PCT}%")
    log.info(f"  5. Time stop  bars≥{MAX_BARS_HELD}  or EOD 15:30")
    log.info(f"Momentum gate : score≥{MOMENTUM_SCORE_THRESHOLD}/8 to enter")
    log.info(f"MINCONF={MINCONF}  MINADX={MINADX}  COOLDOWN={COOLDOWNBARS}  SL_REENTRY={SLREENTRYBARS}")
    log.info("=" * 60)

    # ══════════════════════════════════════════════════════════════════
    # ✅ Official Dhan polling pattern:
    #    feed.run_forever()  — connects the WebSocket (blocks briefly)
    #    feed.get_instrument_data() — returns one parsed packet, blocks
    #                                 until data arrives or connection drops
    # The same feed event loop is used for both calls; callbacks are not needed.
    # ══════════════════════════════════════════════════════════════════
    _BASE_DELAY             = 5
    _MAX_DELAY              = 120
    _reconnect_delay        = _BASE_DELAY
    _connected_successfully = False
    _subscription_ids = {str(item[1]).strip() for item in subscription}

    feed = _MF(dhan_context, subscription, "v2")
    FEED = feed   # ← store globally on first creation too
    log.info(f"Feed created  class={type(feed).__name__}")
    log.info(f"FEED has subscribe_symbols: {hasattr(FEED, 'subscribe_symbols')}")
    log.info(f"FEED has unsubscribe_symbols: {hasattr(FEED, 'unsubscribe_symbols')}")

    while True:
        try:
            feed.run_forever()
            log.info("WebSocket connected — waiting for the first market-data packet")
            _connected_successfully = True
            _reconnect_delay = _BASE_DELAY
            _packet_count = 0
            _spot_tick_seen = False
            _next_feed_heartbeat = time_mod.monotonic() + 30

            # ── Tick polling loop ─────────────────────────────────────
            while True:
                try:
                    response = feed.loop.run_until_complete(
                        asyncio.wait_for(
                            feed.get_instrument_data(),
                            timeout=5.0,
                        )
                    )
                except asyncio.TimeoutError:
                    if _flush_idle_minute():
                        log.warning(
                            "[FEED] Clock boundary flush closed the accumulated bar"
                        )
                    if time_mod.monotonic() >= _next_feed_heartbeat:
                        if _packet_count == 0:
                            log.warning(
                                "[FEED] No websocket packets received since connect; "
                                "check Dhan feed entitlements, token, and subscriptions"
                            )
                        elif not _spot_tick_seen:
                            log.warning(
                                f"[FEED] Received {_packet_count} websocket packets, "
                                "but no valid NIFTY spot tick; minute bars cannot be built"
                            )
                        else:
                            log.info(
                                f"[FEED] Market-data heartbeat: {_packet_count} packets "
                                "received; NIFTY spot ticks are flowing"
                            )
                        _next_feed_heartbeat = time_mod.monotonic() + 30
                    continue

                # feed.on_close is set True on server disconnect
                if feed.on_close:
                    log.warning("Feed on_close flag set — reconnecting...")
                    break

                if response is None:
                    if _flush_idle_minute():
                        log.warning(
                            "[FEED] Clock boundary flush closed the accumulated bar"
                        )
                    continue

                _packet_count += 1
                if _packet_count == 1:
                    log.info(
                        f"[FEED] First websocket packet: type={type(response).__name__} "
                        f"payload={str(response)[:240]}"
                    )
                if isinstance(response, dict):
                    _response_id = str(response.get("security_id", "")).strip()
                    if (
                        not _spot_tick_seen
                        and _response_id == str(INSTRUMENTS["nifty_spot"][1]).strip()
                        and response.get("LTP")
                        and response.get("LTT")
                    ):
                        _spot_tick_seen = True
                        log.info(
                            f"[FEED] First NIFTY spot tick received: "
                            f"security_id={_response_id} LTT={response.get('LTT')}"
                        )
                    elif _response_id:
                        # _subscription_ids is a startup snapshot. Grid recentering
                        # and trade-contract swaps can add valid IDs later, so check
                        # the current grid and active trade before calling a packet
                        # unrecognized. This is logging/routing validation only; it
                        # does not change tick processing or trading decisions.
                        _known_feed_ids = set(_subscription_ids)
                        for _instrument in INSTRUMENTS.values():
                            if isinstance(_instrument, (tuple, list)) and len(_instrument) >= 2:
                                _known_feed_ids.add(str(_instrument[1]).strip())
                        if GRID is not None:
                            _known_feed_ids.update(str(_sid).strip() for _sid in GRID.ids)
                        if _ACTIVE_TRADE_FEED_ID is not None:
                            _known_feed_ids.add(str(_ACTIVE_TRADE_FEED_ID).strip())

                        if _response_id not in _known_feed_ids:
                            # Warn once per unknown ID rather than once per packet.
                            # The same set is used by process_tick() for its guard.
                            if len(_UNRECOGNIZED_ID_LOGGED) > 500:
                                _UNRECOGNIZED_ID_LOGGED.clear()
                            if _response_id not in _UNRECOGNIZED_ID_LOGGED:
                                log.warning(
                                    f"[FEED] Packet has unrecognized security_id={_response_id}; "
                                    f"keys={list(response.keys())}"
                                )
                                _UNRECOGNIZED_ID_LOGGED.add(_response_id)
                if _flush_idle_minute():
                    log.warning(
                        "[FEED] Clock boundary flush closed the accumulated bar"
                    )
                process_tick(response)

        except KeyboardInterrupt:
            log.info("Stopped by user")
            tracker.status()
            tracker.save("paper_trades_v14dhan_l5.csv")
            log.info(f"Total trades: {len(tracker.trades)}")
            log.info(f"Log: {LOG_FILE}")
            if syncer:
                syncer.stop()
            try:
                feed.close_connection()
            except Exception:
                pass
            break

        except InvalidStatus as e:
            err_str = str(e)
            if "429" in err_str:
                jitter = random.uniform(0, 3)
                wait   = _reconnect_delay + jitter
                log.warning(
                    f"HTTP 429 — Dhan rate-limited. "
                    f"Waiting {wait:.1f}s before reconnect "
                    f"(backoff={_reconnect_delay}s, jitter={jitter:.1f}s)..."
                )
                time_mod.sleep(wait)
                _reconnect_delay = min(_reconnect_delay * 2, _MAX_DELAY)
            else:
                log.error(f"Feed WS handshake error: {e}  reconnecting in {_reconnect_delay}s...")
                time_mod.sleep(_reconnect_delay)
                _reconnect_delay = min(_reconnect_delay * 2, _MAX_DELAY)

            bar_acc = _reset_bar_acc()
            log.info("Bar accumulator reset")
            try:
                feed.close_connection()
            except Exception:
                pass
            log.info(f"Reconnecting...  next_backoff={_reconnect_delay}s")
            feed = _MF(dhan_context, subscription, "v2")
            FEED = feed   # ← FIX 2a: update global FEED on InvalidStatus reconnect
            log.info(f"Feed recreated  class={type(feed).__name__}")
            log.info(f"FEED has subscribe_symbols: {hasattr(FEED, 'subscribe_symbols')}")
            log.info(f"FEED has unsubscribe_symbols: {hasattr(FEED, 'unsubscribe_symbols')}")

        except Exception as e:
            log.error(f"Feed error: {e}  reconnecting in {_reconnect_delay}s...")
            bar_acc = _reset_bar_acc()
            log.info("Bar accumulator reset")
            try:
                feed.close_connection()
            except Exception:
                pass
            log.info(f"Waiting {_reconnect_delay}s before reconnect...")
            time_mod.sleep(_reconnect_delay)
            if _connected_successfully:
                _reconnect_delay = _BASE_DELAY
                log.info("Last connection was healthy — backoff reset to 5s")
                _connected_successfully = False
            else:
                _reconnect_delay = min(_reconnect_delay * 2, _MAX_DELAY)
            log.info(f"Reconnecting...  next_backoff={_reconnect_delay}s")
            try:
                feed.close_connection()
            except Exception:
                pass
            feed = _MF(dhan_context, subscription, "v2")
            FEED = feed   # ← FIX 2b: update global FEED on generic Exception reconnect
            log.info(f"Feed recreated  class={type(feed).__name__}")
            log.info(f"FEED has subscribe_symbols: {hasattr(FEED, 'subscribe_symbols')}")
            log.info(f"FEED has unsubscribe_symbols: {hasattr(FEED, 'unsubscribe_symbols')}")