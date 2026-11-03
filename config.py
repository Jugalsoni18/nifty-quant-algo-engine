"""Public starter configuration for NIFTY Quant Algo Engine.

Replace placeholder credentials only in your local copy.
Never commit real broker credentials, passwords, TOTP secrets, or tokens.
"""

import os
from datetime import time

# ── Broker credentials (placeholders only) ─────────────────────
request_token = "ZERODHA_REQUEST_TOKEN"
api_key = "ZERODHA_API_KEY"
api_secret = "ZERODHA_API_SECRET"
ZERODHA_USER_ID = "ZERODHA_USER_ID"
ZERODHA_PASSWORD = "ZERODHA_PASSWORD"
ZERODHA_TOTP_SECRET = "ZERODHA_TOTP_SECRET"

dhan_api_key = "DHAN_API_KEY"
dhan_api_secret = "DHAN_API_SECRET"
DHAN_CLIENT_ID = "DHAN_CLIENT_ID"
DHAN_ACCESS_TOKEN = "DHAN_ACCESS_TOKEN"

# ── Project-relative paths ─────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "nifty_v12_l3_models")
ACCESS_TOKEN_FILE = os.path.join(BASE_DIR, "token.json")
TOKEN_FILE = ACCESS_TOKEN_FILE
LOG_FILE = os.path.join(BASE_DIR, "bot_log.csv")

# ── Trade parameters ───────────────────────────────────────────
SL_PCT = "20 points"  # Confirm expected type/units in the consuming code.
MAX_HOLD = 180         # Maximum hold in minutes.
LOT_SIZE = 65          # Keep aligned with the applicable NIFTY contract lot.

# ── Signal parameters ──────────────────────────────────────────
WARMUP_BARS = 500
FETCH_BARS = 550
MIN_ADX = 18
MIN_CONFLUENCE = 5
CONF_BUFFER = 0.06
POLL_SECS = 15
PAPER_TRADE = True

# ── VIX gates ──────────────────────────────────────────────────
VIX_MAX_TRADE = 29.0
VIX_CAUTION = 20.0
VIX_SL_MULT = 1.5
VIX_SPIKE_PCT = 0.15

# ── Extra signal filters ────────────────────────────────────────
SHORT_MIN_CONF = 0.65
BB5M_BULL_MIN = 0.45
BB5M_BEAR_MAX = 0.55
ADX_EXHAUST = 50
BB5M_EXTREME_H = 1.00
BB5M_EXTREME_L = 0.00
VWAP_TOLERANCE = 0.0003

# ── Zerodha execution and risk configuration ──────────────────
# Safe starter default: keep dry_run enabled until live execution has
# been explicitly reviewed and deliberately enabled in a local config.
ZERODHA_CONFIG = {
    "dry_run": True,
    "lot_size": 65,
    "max_lots": 1,
    "product": "MIS",
    "max_trades_per_day": 5,
    "daily_loss_limit": -10000.0,
    "max_capital_per_trade": 20000.0,
    "fill_timeout_secs": 30,
    "place_sl_order": False,
    "slippage_buffer": 0.5,
    "market_open": time(9, 15),
    "market_close": time(15, 15),
    "market_protection": 2.0,
}
