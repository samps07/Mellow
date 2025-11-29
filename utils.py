import os
import base64
import secrets
import hmac
import hashlib
import streamlit as st

def get_secret_key() -> str:
    # 1. Try Streamlit Secrets
    if hasattr(st, "secrets") and "SECRET_KEY" in st.secrets:
        return st.secrets["SECRET_KEY"]
    
    # 2. Try Env Var
    if os.environ.get("SECRET_KEY"):
        return os.environ["SECRET_KEY"]
        
    # 3. Fallback: Generate a temporary one for this session
    # Note: If the app restarts, tokens generated with this will become invalid.
    if "temp_secret_key" not in st.session_state:
        st.session_state.temp_secret_key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
    
    return st.session_state.temp_secret_key

SECRET_KEY = get_secret_key()

def hmac_hash(token: str) -> str:
    return hmac.new(SECRET_KEY.encode(), token.encode(), hashlib.sha256).hexdigest()

def generate_token() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()