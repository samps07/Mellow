# gemini.py
import os
import google.generativeai as genai
from dotenv import load_dotenv
import logging

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

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