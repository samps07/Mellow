# moderation.py
import os
import json
import logging

# Import the centralized Gemini model
import gemini

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Local profanity list preserved
PROFANITY = ["fuck", "shit", "bitch", "asshole"]


def simple_moderation(text: str) -> (bool, str):
    """
    Local quick checks, then call gemini_moderation if key present.
    Returns (allowed: bool, reason: str)
    """
    lowered = (text or "").lower()
    for w in PROFANITY:
        if w in lowered:
            return False, "contains profanity"
    if len((text or "").strip()) < 3:
        return False, "too short"

    if gemini.IS_CONFIGURED:
        try:
            logger.info("Attempting Gemini moderation...")
            gm = gemini_moderation(text)
            if gm and isinstance(gm, dict) and "allowed" in gm and "reason" in gm:
                allowed = bool(gm.get("allowed", True))
                reason = gm.get("reason", "unknown")
                logger.info("Gemini moderation result: Allowed=%s Reason=%s", allowed, reason)
                if not allowed:
                    return allowed, reason
            else:
                logger.warning("Unexpected response structure from gemini_moderation: %s", gm)
        except Exception as e:
            logger.error("Gemini moderation call failed: %s", e)
            # Do not block on moderation API failure
            pass
    else:
        logger.warning("Gemini moderation skipped: Gemini is not configured.")

    return True, "ok"


def gemini_moderation(text: str) -> dict:
    """
    Ask Gemini to return a strict JSON: {"allowed": bool, "reason": str}
    Uses model.generate_content and parses response.text.
    """
    if not gemini.IS_CONFIGURED:
        logger.error("gemini_moderation called but Gemini is not configured.")
        return {"allowed": True, "reason": "internal_error_key_missing"}

    prompt = (
        "You are a content-safety assistant. Given the text below, "
        "return a valid JSON object with keys: allowed (true/false) and reason (short string). "
        "Do not add extra commentary. Example output: {\"allowed\": false, \"reason\": \"contains profanity\"}\n\n"
        f"Text:\n'''{text}'''\n\n"
        "JSON response:"
    )

    try:
        resp = gemini.model.generate_content(prompt)
        content = getattr(resp, "text", "") or ""
        content = content.strip()
        logger.debug("Gemini moderation raw text: %s", content)

        if not content:
            logger.warning("Gemini moderation response missing text content.")
            return {"allowed": True, "reason": "gemini_no_text"}

        # (Your original JSON parsing logic is good, keeping it)
        cleaned = content
        if cleaned.startswith("```") and cleaned.endswith("```"):
            parts = cleaned.split("\n")
            if len(parts) >= 3:
                cleaned = "\n".join(parts[1:-1]).strip()

        cleaned = cleaned.strip().strip('`').strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and "allowed" in parsed and "reason" in parsed:
                return {"allowed": bool(parsed["allowed"]), "reason": str(parsed["reason"])}
            else:
                logger.warning("Gemini moderation JSON parsed but unexpected keys: %s", parsed)
                return {"allowed": True, "reason": "gemini_bad_json_format"}
        except json.JSONDecodeError as jde:
            logger.warning("Gemini moderation JSON decode error: %s. Raw: %s", jde, content)
            low = content.lower()
            if any(w in low for w in PROFANITY + ["abuse", "threat", "not allowed", "disallowed", "blocked"]):
                return {"allowed": False, "reason": "gemini_unsafe_text_fallback"}
            return {"allowed": True, "reason": "gemini_json_decode_error"}

    except Exception as e:
        logger.error("Gemini moderation unexpected error: %s", e)
        lowered = (text or "").lower()
        for w in PROFANITY:
            if w in lowered:
                return {"allowed": False, "reason": "contains profanity (fallback)"}
    return {"allowed": True, "reason": "ok_after_api_attempt"}






# # moderation.py
# import os
# from dotenv import load_dotenv
# import json
# import logging

# # Load .env exactly as your working reference
# load_dotenv()

# # Strict reference initialization
# import google.generativeai as genai
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# model = genai.GenerativeModel("gemini-2.5-flash")

# # Logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Local profanity list preserved
# PROFANITY = ["fuck", "shit", "bitch", "asshole"]


# def simple_moderation(text: str) -> (bool, str):
#     """
#     Local quick checks, then call gemini_moderation if key present.
#     Returns (allowed: bool, reason: str)
#     """
#     lowered = (text or "").lower()
#     for w in PROFANITY:
#         if w in lowered:
#             return False, "contains profanity"
#     if len((text or "").strip()) < 3:
#         return False, "too short"

#     if os.getenv("GEMINI_API_KEY"):
#         try:
#             logger.info("Attempting Gemini moderation via model.generate_content")
#             gm = gemini_moderation(text)
#             if gm and isinstance(gm, dict) and "allowed" in gm and "reason" in gm:
#                 allowed = bool(gm.get("allowed", True))
#                 reason = gm.get("reason", "unknown")
#                 logger.info("Gemini moderation result: Allowed=%s Reason=%s", allowed, reason)
#                 if not allowed:
#                     return allowed, reason
#             else:
#                 logger.warning("Unexpected response structure from gemini_moderation: %s", gm)
#         except Exception as e:
#             logger.error("Gemini moderation call failed: %s", e)
#             # Do not block on moderation API failure
#             pass
#     else:
#         logger.warning("Gemini moderation skipped: GEMINI_API_KEY not set.")

#     return True, "ok"


# def gemini_moderation(text: str) -> dict:
#     """
#     Ask Gemini to return a strict JSON: {"allowed": bool, "reason": str}
#     Uses model.generate_content and parses response.text.
#     """
#     if not os.getenv("GEMINI_API_KEY"):
#         logger.error("gemini_moderation called but GEMINI_API_KEY missing.")
#         return {"allowed": True, "reason": "internal_error_key_missing"}

#     prompt = (
#         "You are a content-safety assistant. Given the text below, "
#         "return a valid JSON object with keys: allowed (true/false) and reason (short string). "
#         "Do not add extra commentary. Example output: {\"allowed\": false, \"reason\": \"contains profanity\"}\n\n"
#         f"Text:\n'''{text}'''\n\n"
#         "JSON response:"
#     )

#     try:
#         resp = model.generate_content(
#             prompt
#         )
#         content = getattr(resp, "text", "") or ""
#         content = content.strip()
#         logger.debug("Gemini moderation raw text: %s", content)

#         if not content:
#             logger.warning("Gemini moderation response missing text content.")
#             return {"allowed": True, "reason": "gemini_no_text"}

#         # Remove triple-fence code blocks if present
#         cleaned = content
#         if cleaned.startswith("```") and cleaned.endswith("```"):
#             parts = cleaned.split("\n")
#             if len(parts) >= 3:
#                 cleaned = "\n".join(parts[1:-1]).strip()

#         cleaned = cleaned.strip().strip('`').strip()

#         try:
#             parsed = json.loads(cleaned)
#             if isinstance(parsed, dict) and "allowed" in parsed and "reason" in parsed:
#                 return {"allowed": bool(parsed["allowed"]), "reason": str(parsed["reason"])}
#             else:
#                 logger.warning("Gemini moderation JSON parsed but unexpected keys: %s", parsed)
#                 return {"allowed": True, "reason": "gemini_bad_json_format"}
#         except json.JSONDecodeError as jde:
#             logger.warning("Gemini moderation JSON decode error: %s. Raw: %s", jde, content)
#             low = content.lower()
#             if any(w in low for w in PROFANITY + ["abuse", "threat", "not allowed", "disallowed", "blocked"]):
#                 return {"allowed": False, "reason": "gemini_unsafe_text_fallback"}
#             return {"allowed": True, "reason": "gemini_json_decode_error"}

#     except Exception as e:
#         logger.error("Gemini moderation unexpected error: %s", e)
#         lowered = (text or "").lower()
#         for w in PROFANITY:
#             if w in lowered:
#                 return {"allowed": False, "reason": "contains profanity (fallback)"}
#     return {"allowed": True, "reason": "ok_after_api_attempt"}
