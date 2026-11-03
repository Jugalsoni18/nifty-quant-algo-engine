from kiteconnect import KiteConnect
from config import api_key, api_secret
import json

kite = KiteConnect(api_key=api_key)

print("\nLogin URL:")
print(kite.login_url())

request_token = input("\nPaste NEW request_token here: ").strip()

data = kite.generate_session(
    request_token,
    api_secret=api_secret
)

kite.set_access_token(data["access_token"])

with open("token.json", "w") as f:
    json.dump({
        "access_token": data["access_token"]
    }, f, indent=4)

print("✅ Access Token Generated Successfully")
print(data["access_token"])

