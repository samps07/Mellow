import os
import google.generativeai as genai
from dotenv import load_dotenv
import logging
import streamlit as st

# Load .env for local development (won't do anything on cloud if file is missing)
load_dotenv()

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. GET API KEY ---
API_KEY = None

# Check Streamlit Secrets first (Cloud)
if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
    API_KEY = st.secrets["GEMINI_API_KEY"]

MODEL_NAME = "gemini-2.5-flash"

IS_CONFIGURED = False
model = None

if API_KEY:
    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)
        IS_CONFIGURED = True
        logging.info(f"Gemini model '{MODEL_NAME}' loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to configure Gemini: {e}", exc_info=True)
else:
    logging.warning("GEMINI_API_KEY not found. Gemini features (RAG, Moderation) will be disabled.")
