# rag.py
"""
RAG helper using google.generativeai (Gemini).
Imports shared model from gemini.py
"""
import logging
from typing import Any, Dict, List
import streamlit as st # Added for UI feedback (Toasts)

# Import the centralized Gemini model
import gemini
import embeddings
import db
import tagging 

logging.basicConfig(level=logging.INFO)
_logging = logging.getLogger(__name__)

def _call_gemini_with_model(prompt: str) -> str:
    """
    Use google.generativeai.GenerativeModel.generate_content to get text from Gemini.
    """
    if not gemini.IS_CONFIGURED:
        raise RuntimeError("Gemini is not configured. Cannot call model.")
        
    try:
        _logging.info("Calling Gemini model.generate_content")
        
        # --- CRITICAL FIX: Relax Safety Settings ---
        # Gemini defaults to blocking anything slightly "unsafe". 
        # For a grammar fixer, we need it to process everything.
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        resp = gemini.model.generate_content(
            prompt, 
            safety_settings=safety_settings
        )
        
        # Standard text extraction
        text = getattr(resp, "text", None)
        if text and isinstance(text, str) and text.strip():
            return text.strip()

        # Fallback extraction (for edge cases)
        candidates = getattr(resp, "candidates", None)
        if candidates and len(candidates) > 0:
            c0 = candidates[0]
            if hasattr(c0, "content") and hasattr(c0.content, "parts") and c0.content.parts:
                text = c0.content.parts[0].text
                if text: return text.strip()
                
        raise RuntimeError("Gemini returned an empty response (likely filtered).")

    except Exception as e:
        _logging.error("RAG: Gemini call error: %s", e)
        raise

def fix_grammar(text: str) -> str:
    """
    Uses Gemini to fix grammar and spelling. 
    """
    if not text or not text.strip():
        return text

    if not gemini.IS_CONFIGURED:
        st.toast("⚠️ Geminify skipped: API Key missing.", icon="⚠️")
        return text

    # Improved prompt to prevent "Here is the corrected text:" chatter
    prompt = (
        f"correct the grammar. do not change tone or actual meaning. return only the corrected text. Text:{text}"
    )
    
    try:
        corrected = _call_gemini_with_model(prompt)

        # Check if the text actually changed (ignoring case)
        if corrected.strip() and corrected.strip().lower() != text.strip().lower():
            _logging.info("Geminify applied corrections.")
            st.toast("Grammar fixed by Gemini!", icon="✨") # Visual Feedback
            return corrected.strip()
        else:
            _logging.info("Geminify no change.")
            st.toast("✅ Text looks good! No changes needed.", icon="👍")
            return text
            
    except Exception as e:
        _logging.error("Geminify Error: %s", e)
        st.toast(f"❌ Geminify failed: {str(e)[:50]}...", icon="uxorz")
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
            # We also relax safety settings for RAG to avoid blocking valid search results
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            
            _logging.info("Calling Gemini model.generate_content for RAG")
            resp = gemini.model.generate_content(prompt, safety_settings=safety_settings)
            
            text_out = getattr(resp, "text", "")
            
            _logging.info("RAG executed successfully using Gemini.")
            return {"answer": text_out, "source_documents": sources, "used_gemini": True}
        except Exception as e:
            _logging.error("RAG Gemini call failed: %s", e)
            return {"answer": None, "source_documents": sources, "used_gemini": False, "error": str(e)}

    _logging.warning("RAG fallback: Gemini is not configured.")
    return {"answer": None, "source_documents": sources, "used_gemini": False}
