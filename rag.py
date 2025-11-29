# rag.py
"""
RAG helper using google.generativeai (Gemini).
Imports shared model from gemini.py
"""
import os
import json
import logging
from typing import Any, Dict, List

# Import the centralized Gemini model
import gemini
import embeddings
import db
import tagging 

logging.basicConfig(level=logging.INFO)
_logging = logging.getLogger(__name__)

# Small debug to confirm key presence
_logging.info("RAG: Gemini IS_CONFIGURED: %s", gemini.IS_CONFIGURED)


def _call_gemini_with_model(prompt: str) -> str:
    """
    Use google.generativeai.GenerativeModel.generate_content to get text from Gemini.
    """
    if not gemini.IS_CONFIGURED:
        raise RuntimeError("Gemini is not configured. Cannot call model.")
        
    try:
        _logging.info("Calling Gemini model.generate_content")
        resp = gemini.model.generate_content(prompt)
        text = getattr(resp, "text", None)
        if text and isinstance(text, str) and text.strip():
            return text.strip()

        # Fallback attempts
        # ... (keeping your original fallbacks just in case)
        candidates = getattr(resp, "candidates", None)
        if candidates and isinstance(candidates, (list, tuple)) and len(candidates) > 0:
            c0 = candidates[0]
            if hasattr(c0, "content") and hasattr(c0.content, "parts") and c0.content.parts:
                text = c0.content.parts[0].text
                if text and isinstance(text, str) and text.strip():
                    return text.strip()
            if hasattr(c0, "text") and isinstance(c0.text, str) and c0.text.strip():
                 return c0.text.strip()

        _logging.error("RAG: failed to extract text from Gemini response; response object logged.")
        _logging.debug("Gemini response object: %s", resp)
        raise RuntimeError("Failed to extract text from Gemini response.")
    except Exception as e:
        _logging.error("RAG: Gemini call error: %s", e)
        raise


def fix_grammar(text: str) -> str:
    """
    Uses Gemini to fix grammar and spelling. Falls back to original on failure.
    This is now a pure function that just returns the corrected text.
    """
    if not text or not text.strip():
        return text

    if not gemini.IS_CONFIGURED:
        _logging.warning("Geminify skipped: Gemini is not configured.")
        return text

    prompt = (
        "You are an editor. Correct all spelling and grammar mistakes in the following text. "
        "Do NOT change the meaning or tone of the text. "
        "Only output the corrected text. Do not add any commentary before or after the text.\n\n"
        f"Original:\n'''{text}'''\n\n"
        "Corrected:\n"
    )
    try:
        corrected = _call_gemini_with_model(prompt)

        if corrected.strip() and corrected.strip().lower() != text.strip().lower():
            _logging.info("Geminify applied corrections.")
            return corrected.strip()
        else:
            _logging.info("Geminify no change.")
            return text
    except Exception as e:
        _logging.error("Geminify Error: %s", e)
        # On failure, return original text
        return text


def ask(query: str, k: int = 5) -> Dict[str, Any]:
    """
    Retrieve top-k posts, ask Gemini to answer using only those snippets.
    """
    if not query or not query.strip():
        return {"answer": "", "source_documents": [], "used_gemini": False}

    try:
        results = embeddings.semantic_search(query, k=k)
    except Exception as e:
        _logging.error("Semantic search failed: %s", e)
        return {"answer": None, "source_documents": [], "used_gemini": False, "error": f"Search failed: {e}"}

    sources: List[Dict[str, Any]] = []
    if results:
        for pid, score in results:
            row = db.get_post(pid)
            if not row:
                continue
            _, content, _, title, *rest = row
            full_text = f"{(title or '').strip()}\n\n{(content or '').strip()}"
            sources.append({"post_id": pid, "text": full_text, "score": score})

    # (Your logic for adding recent posts is good, keeping it)
    try:
        recent = db.list_recent(limit=5)
        for pid, content, ts, title, *rest in recent:
            if not any(s.get("post_id") == pid for s in sources):
                full_text = f"{(title or '').strip()}\n\n{(content or '').strip()}"
                sources.append({"post_id": pid, "text": full_text, "score": 0.0})
    except Exception as e:
        _logging.error("Failed to fetch recent posts for RAG: %s", e)

    if not sources:
        return {"answer": "No matching posts found.", "source_documents": [], "used_gemini": False}

    snippets: List[str] = []
    for s in sources:
        text = s["text"]
        excerpt = text.strip()
        if len(excerpt) > 1200:
            excerpt = excerpt[:1200] + " ... [truncated]"
        snippets.append(f"[POST #{s['post_id']}]\n{excerpt}\n")

    prompt = (
        "You are an assistant that MUST answer the user's question using ONLY the provided Evidence snippets.\n"
        "Do NOT use external knowledge. If the snippets do not support a confident answer, reply exactly: \"I don't know\".\n"
        "Provide a concise answer (1-4 sentences), then a SOURCES section listing POST ids used in the form: POST #<id>: <one-line excerpt>.\n\n"
        f"User question:\n{query}\n\n"
        "Evidence snippets:\n" + "\n".join(snippets) +
        "\n\nAnswer now using ONLY the Evidence snippets. Include a SOURCES section."
    )

    if gemini.IS_CONFIGURED:
        try:
            text_out = _call_gemini_with_model(prompt)
            _logging.info("RAG executed successfully using Gemini.")
            return {"answer": text_out, "source_documents": sources, "used_gemini": True}
        except Exception as e:
            _logging.error("RAG Gemini call failed: %s", e)
            return {"answer": None, "source_documents": sources, "used_gemini": False, "error": str(e)}

    _logging.warning("RAG fallback: Gemini is not configured.")
    return {"answer": None, "source_documents": sources, "used_gemini": False}



# # rag.py
# """
# RAG helper using google.generativeai (Gemini).
# Strictly uses the working reference pattern for API key and model initialization.
# Updated: when grammar is corrected, create one-shot wrappers for db.insert_post,
# embeddings.add_to_index and tagging.generate_tags_for_post so the corrected text
# is used for saving, embedding and tagging — without changing app.py.
# """
# import os
# import json
# import logging
# from typing import Any, Dict, List

# import streamlit as st
# from dotenv import load_dotenv

# # Load .env exactly as your reference does
# load_dotenv()

# # Strict reference initialization (exact pattern you provided)
# import google.generativeai as genai
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# model = genai.GenerativeModel("gemini-2.5-flash")

# # Local imports (unchanged)
# import embeddings
# import db
# import tagging  # imported so we can wrap its function when needed

# logging.basicConfig(level=logging.INFO)
# _logging = logging.getLogger(__name__)

# # Small debug to confirm key presence (safe boolean)
# _logging.info("RAG: GEMINI_API_KEY present: %s", bool(os.getenv("GEMINI_API_KEY")))


# def _call_gemini_with_model(prompt: str) -> str:
#     """
#     Use google.generativeai.GenerativeModel.generate_content to get text from Gemini.
#     Calls generate_content with only the prompt (per your working reference).
#     Returns response.text (string) or raises RuntimeError on failure.
#     """
#     try:
#         logging.info("Calling Gemini model.generate_content")
#         resp = model.generate_content(prompt)
#         # The working reference returns text via resp.text
#         text = getattr(resp, "text", None)
#         if text and isinstance(text, str) and text.strip():
#             return text.strip()

#         # Fallback attempts (rare)
#         text = getattr(resp, "output_text", None)
#         if text and isinstance(text, str) and text.strip():
#             return text.strip()

#         candidates = getattr(resp, "candidates", None)
#         if candidates and isinstance(candidates, (list, tuple)) and len(candidates) > 0:
#             c0 = candidates[0]
#             if isinstance(c0, dict):
#                 for key in ("output", "content", "text"):
#                     if key in c0 and isinstance(c0[key], str) and c0[key].strip():
#                         return c0[key].strip()
#             else:
#                 if hasattr(c0, "content") and isinstance(c0.content, str) and c0.content.strip():
#                     return c0.content.strip()
#                 if hasattr(c0, "text") and isinstance(c0.text, str) and c0.text.strip():
#                     return c0.text.strip()

#         logging.error("RAG: failed to extract text from Gemini response; response object logged.")
#         logging.debug("Gemini response object: %s", resp)
#         raise RuntimeError("Failed to extract text from Gemini response.")
#     except Exception as e:
#         logging.error("RAG: Gemini call error: %s", e)
#         raise


# def _install_one_shot_wrappers(corrected_text: str):
#     """
#     Monkeypatch db.insert_post, embeddings.add_to_index and tagging.generate_tags_for_post
#     with single-use wrappers that substitute corrected_text into the content/text argument.
#     After first call they restore the original functions.
#     """

#     # DB insert_post wrapper
#     try:
#         orig_insert = db.insert_post
#     except Exception:
#         orig_insert = None

#     if orig_insert:
#         def insert_wrapper(*args, **kwargs):
#             # Replace content positional or keyword argument with corrected_text
#             try:
#                 # positional: content is normally first arg
#                 if len(args) >= 1:
#                     args = list(args)
#                     args[0] = corrected_text
#                 elif "content" in kwargs:
#                     kwargs["content"] = corrected_text
#             except Exception:
#                 pass
#             # restore original immediately
#             db.insert_post = orig_insert
#             logging.info("db.insert_post wrapper used and restored (content replaced with corrected text).")
#             return orig_insert(*args, **kwargs)
#         db.insert_post = insert_wrapper

#     # embeddings.add_to_index wrapper
#     try:
#         orig_add = embeddings.add_to_index
#     except Exception:
#         orig_add = None

#     if orig_add:
#         def add_wrapper(post_id, text, metadata=None):
#             try:
#                 text = corrected_text
#             except Exception:
#                 pass
#             # restore original
#             embeddings.add_to_index = orig_add
#             logging.info("embeddings.add_to_index wrapper used and restored (text replaced with corrected text).")
#             return orig_add(post_id, text, metadata=metadata)
#         embeddings.add_to_index = add_wrapper

#     # tagging.generate_tags_for_post wrapper
#     try:
#         orig_tag = tagging.generate_tags_for_post
#     except Exception:
#         orig_tag = None

#     if orig_tag:
#         def tag_wrapper(post_id, doc, top_n_keywords=5, topic_threshold=0.5):
#             try:
#                 doc = corrected_text
#             except Exception:
#                 pass
#             # restore original
#             tagging.generate_tags_for_post = orig_tag
#             logging.info("tagging.generate_tags_for_post wrapper used and restored (doc replaced with corrected text).")
#             return orig_tag(post_id, doc, top_n_keywords=top_n_keywords, topic_threshold=topic_threshold)
#         tagging.generate_tags_for_post = tag_wrapper

#     # For debugging visibility
#     try:
#         st.session_state["__last_geminified_text__"] = corrected_text
#     except Exception:
#         pass


# def fix_grammar(text: str) -> str:
#     """
#     Uses Gemini to fix grammar and spelling. Falls back to original on failure.

#     Also installs one-shot wrappers so the corrected text is used for DB insert,
#     embeddings and tagging when app.py continues to use its original local variable.
#     """
#     if not text or not text.strip():
#         try:
#             st.toast("⚠️ Nothing to Geminify.", icon="⚠️")
#         except Exception:
#             pass
#         return text

#     if not os.getenv("GEMINI_API_KEY"):
#         try:
#             st.toast("⚠️ Geminify skipped: GEMINI_API_KEY not configured.", icon="⚠️")
#         except Exception:
#             pass
#         logging.warning("Geminify skipped: GEMINI_API_KEY missing.")
#         return text

#     prompt = (
#         "You are an editor. Correct all spelling and grammar mistakes in the following text. "
#         "Do NOT change the meaning or tone of the text. "
#         "Only output the corrected text. Do not add any commentary before or after the text.\n\n"
#         f"Original:\n'''{text}'''\n\n"
#         "Corrected:\n"
#     )
#     try:
#         corrected = _call_gemini_with_model(prompt)

#         if corrected.strip() and corrected.strip().lower() != text.strip().lower():
#             try:
#                 st.toast("✨ Grammar fixed by Gemini!", icon="✨")
#             except Exception:
#                 pass
#             logging.info("Geminify applied corrections.")

#             # Install one-shot wrappers to ensure the corrected text is used downstream
#             try:
#                 _install_one_shot_wrappers(corrected.strip())
#             except Exception as wrap_e:
#                 logging.error("Failed to install one-shot wrappers: %s", wrap_e)

#             # Also attempt to write to a few session_state keys for visibility/debug
#             try:
#                 possible_keys = [
#                     "Body", "body", "post_text_input",
#                     "create_post_form-Body", "create_post_form|Body", "create_post_form:Body"
#                 ]
#                 for k in possible_keys:
#                     try:
#                         st.session_state[k] = corrected.strip()
#                     except Exception:
#                         pass
#                 st.session_state["__last_geminified_text__"] = corrected.strip()
#             except Exception:
#                 pass

#             return corrected.strip()
#         else:
#             try:
#                 st.toast("✅ No grammar changes needed or API returned empty.", icon="✅")
#             except Exception:
#                 pass
#             logging.info("Geminify no change.")
#             return text
#     except Exception as e:
#         logging.error("Geminify Error: %s", e)
#         try:
#             st.error(f"Geminify failed: {e}")
#         except Exception:
#             pass
#         return text


# def ask(query: str, k: int = 5) -> Dict[str, Any]:
#     """
#     Retrieve top-k posts, ask Gemini to answer using only those snippets.
#     """
#     if not query or not query.strip():
#         return {"answer": "", "source_documents": [], "used_gemini": False}

#     try:
#         results = embeddings.semantic_search(query, k=k)
#     except Exception as e:
#         logging.error("Semantic search failed: %s", e)
#         return {"answer": None, "source_documents": [], "used_gemini": False, "error": f"Search failed: {e}"}

#     sources: List[Dict[str, Any]] = []
#     if results:
#         for pid, score in results:
#             row = db.get_post(pid)
#             if not row:
#                 continue
#             _, content, _, title, *rest = row
#             full_text = f"{(title or '').strip()}\n\n{(content or '').strip()}"
#             sources.append({"post_id": pid, "text": full_text, "score": score})

#     try:
#         recent = db.list_recent(limit=5)
#         for pid, content, ts, title, *rest in recent:
#             if not any(s.get("post_id") == pid for s in sources):
#                 full_text = f"{(title or '').strip()}\n\n{(content or '').strip()}"
#                 sources.append({"post_id": pid, "text": full_text, "score": 0.0})
#     except Exception as e:
#         logging.error("Failed to fetch recent posts for RAG: %s", e)

#     if not sources:
#         return {"answer": "No matching posts found.", "source_documents": [], "used_gemini": False}

#     snippets: List[str] = []
#     for s in sources:
#         text = s["text"]
#         excerpt = text.strip()
#         if len(excerpt) > 1200:
#             excerpt = excerpt[:1200] + " ... [truncated]"
#         snippets.append(f"[POST #{s['post_id']}]\n{excerpt}\n")

#     prompt = (
#         "You are an assistant that MUST answer the user's question using ONLY the provided Evidence snippets.\n"
#         "Do NOT use external knowledge. If the snippets do not support a confident answer, reply exactly: \"I don't know\".\n"
#         "Provide a concise answer (1-4 sentences), then a SOURCES section listing POST ids used in the form: POST #<id>: <one-line excerpt>.\n\n"
#         f"User question:\n{query}\n\n"
#         "Evidence snippets:\n" + "\n".join(snippets) +
#         "\n\nAnswer now using ONLY the Evidence snippets. Include a SOURCES section."
#     )

#     if os.getenv("GEMINI_API_KEY"):
#         try:
#             text_out = _call_gemini_with_model(prompt)
#             logging.info("RAG executed successfully using Gemini.")
#             return {"answer": text_out, "source_documents": sources, "used_gemini": True}
#         except Exception as e:
#             logging.error("RAG Gemini call failed: %s", e)
#             try:
#                 st.warning(f"RAG search failed ({e}). Falling back to simple search.")
#             except Exception:
#                 pass
#             return {"answer": None, "source_documents": sources, "used_gemini": False, "error": str(e)}

#     logging.warning("RAG fallback: GEMINI_API_KEY not configured.")
#     return {"answer": None, "source_documents": sources, "used_gemini": False}
