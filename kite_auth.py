"""
kite_auth.py — Token loaded from token.json (manual mode)
─────────────────────────────────────────────────────────
token.json format (create this file manually):
{
    "access_token": "your_access_token_here",
    "api_key": "your_api_key_here"
}

How to get access_token:
  1. Login to kite.zerodha.com
  2. Open browser DevTools → Network tab
  3. Filter by "user" → look for /session/token request
  4. Copy access_token from the response JSON
  OR use Kite Connect login URL + generate_session() once manually
─────────────────────────────────────────────────────────
"""

import json
import logging
from pathlib import Path
from kiteconnect import KiteConnect

log = logging.getLogger("kite_auth")

TOKEN_FILE = Path(__file__).parent / "token.json"


def get_kite() -> KiteConnect:
    """
    Loads access_token from token.json and returns authenticated KiteConnect.
    No browser, no TOTP — just reads the file.
    """
    if not TOKEN_FILE.exists():
        raise FileNotFoundError(
            f"token.json not found at {TOKEN_FILE}\n"
            "Create it with:\n"
            '{\n'
            '    "access_token": "your_access_token_here",\n'
            '    "api_key": "your_api_key_here"\n'
            '}'
        )

    data = json.loads(TOKEN_FILE.read_text())
    api_key      = data.get("api_key")
    access_token = data.get("access_token")

    if not api_key or not access_token:
        raise ValueError(
            "token.json must have both 'api_key' and 'access_token' fields."
        )

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)

    # Quick validation
    try:
        profile = kite.profile()
        log.info(f"kite_auth: ✅ Authenticated as {profile['user_name']} ({profile['user_id']})")
    except Exception as e:
        raise RuntimeError(
            f"kite_auth: Token validation failed: {e}\n"
            "Your access_token may be expired. Generate a new one and update token.json."
        )

    return kite


# Keep this for future use when TOTP is fixed
def get_kite_manual() -> KiteConnect:
    return get_kite()