# utils.py
import os
import base64
import secrets
import hmac
import hashlib
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY_FILE = ".secret_key"
ENV_SECRET = os.getenv("SECRET_KEY")

def get_secret_key() -> str:
    if ENV_SECRET:
        return ENV_SECRET
    if os.path.exists(SECRET_KEY_FILE):
        return open(SECRET_KEY_FILE, "r").read().strip()
    k = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
    open(SECRET_KEY_FILE, "w").write(k)
    return k

SECRET_KEY = get_secret_key()

def hmac_hash(token: str) -> str:
    """HMAC-SHA256 of token using SECRET_KEY (store this in DB)."""
    return hmac.new(SECRET_KEY.encode(), token.encode(), hashlib.sha256).hexdigest()

def generate_token() -> str:
    """Generate a one-time token to show user (store only hashed)."""
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
