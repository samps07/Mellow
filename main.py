# main.py
# --- Imports ---
import uvicorn  # Server
from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, Response, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.status import HTTP_303_SEE_OTHER
from pydantic import BaseModel, Field
from typing import Optional
from functools import lru_cache
import datetime
import html
import logging

# --- Hugging Face Model (Cached) ---
from transformers import pipeline

# --- Local Modules ---
import db
import embeddings
import moderation
import tagging
import rag
from utils import generate_token, hmac_hash
import gemini # New centralized Gemini module

# --- Initialization ---
db.init_db()  # Initialize the database

# Configure logging
logging.basicConfig(level=logging.INFO)

# --- FastAPI App & Templates ---
app = FastAPI(title="Mellow")
templates = Jinja2Templates(directory="templates")

# Mount a /static directory (though we'll use a CDN for major libraries)
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- Load ML Model (Cached) ---
# Use lru_cache as a replacement for st.cache_resource
@lru_cache(maxsize=1)
def load_sentiment():
    """Loads the sentiment analysis pipeline."""
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    logging.info(f"Loading sentiment model: {model_name}")
    try:
        sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)
        logging.info("Sentiment model loaded successfully.")
        return sentiment_pipeline
    except Exception as e:
        logging.error(f"Failed to load sentiment model: {e}", exc_info=True)
        return None

# Load the model at startup
sentiment_pipeline = load_sentiment()


# --- Helper Functions for Templates ---

def get_sentiment(content: str) -> str:
    """Helper to get sentiment label for a post."""
    if not sentiment_pipeline or not content:
        return "neutral"
    try:
        s = sentiment_pipeline(content)[0]
        raw_label = s.get("label", "").upper()
        score = float(s.get("score", 0.0))
        if score < 0.75: return "neutral"
        elif raw_label == "POSITIVE": return "positive"
        elif raw_label == "NEGATIVE": return "negative"
    except Exception as e:
        logging.warning(f"Sentiment analysis failed: {e}")
    return "neutral"

def format_timestamp(ts: int) -> str:
    """Helper to format timestamp."""
    try:
        dt = datetime.datetime.fromtimestamp(int(ts))
        return dt.strftime("%b %d, '%y %H:%M")
    except Exception:
        return "Invalid date"

def process_post_for_template(row: tuple) -> dict:
    """Converts a DB row tuple into a dict for the template."""
    pid, content, ts, title, likes, dislikes, reports = row
    tags = db.get_tags_for_post(pid)
    
    # Separate tags
    topic_tags = [t.split("topic:")[-1] for t in tags if t.startswith("topic:")]
    general_tags = [t for t in tags if not t.startswith("topic:")]
    
    return {
        "id": pid,
        "content": content,
        "title": title or f"Post #{pid}",
        "likes": likes,
        "dislikes": dislikes,
        "reports": reports,
        "sentiment": get_sentiment(content),
        "timestamp": format_timestamp(ts),
        "topic_tags": topic_tags,
        "general_tags": general_tags
    }

# --- FastAPI Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """
    Main page. Fetches recent posts and renders the index.
    """
    try:
        rows = db.list_recent(limit=50)
        posts = [process_post_for_template(row) for row in rows]
    except Exception as e:
        logging.error(f"Failed to fetch recent posts: {e}", exc_info=True)
        posts = []
        
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "posts": posts, "search_results": None}
    )


@app.post("/posts", response_class=HTMLResponse)
async def create_post(
    request: Request,
    title: Optional[str] = Form(None),
    body: str = Form(...),
    use_geminify: Optional[bool] = Form(False)
):
    """
    Handles new post creation.
    Called by HTMX form. Returns the new post card to prepend.
    """
    final_post_text = body
    
    # 1. Check length
    if not final_post_text or len(final_post_text.strip()) < 3:
        # We can't use st.error, so we return an error message via HTMX
        return HTMLResponse(
            '<div id="form-error" class="text-red-400 p-2 mb-4">⚠️ Post body is too short.</div>',
            status_code=400
        )

    # 2. Apply Geminify (if checked)
    if use_geminify and gemini.IS_CONFIGURED:
        try:
            fixed_text = rag.fix_grammar(final_post_text)
            final_post_text = fixed_text  # Use the fixed text
        except Exception as gemini_e:
            logging.error(f"Geminify failed: {gemini_e}", exc_info=True)
            # Fail gracefully, just use original text

    # 3. Moderation
    allowed, reason = moderation.simple_moderation(final_post_text)
    if not allowed:
        return HTMLResponse(
            f'<div id="form-error" class="text-red-400 p-2 mb-4">⚠️ Post blocked: {reason}</div>',
            status_code=400
        )
        
    # 4. Create Post
    try:
        token = generate_token()
        token_hash = hmac_hash(token)
        
        # Use final_post_text for DB
        post_id = db.insert_post(
            content=final_post_text, 
            token_hash=token_hash, 
            title=title or ""
        )
        logging.info(f"New post #{post_id} created.")

        # Combine title and text for embedding/tagging
        full_text_content = f"{(title or '').strip()}\n\n{final_post_text.strip()}"

        # 5. Embeddings and Tagging
        embeddings.add_to_index(post_id, full_text_content)
        tagging.generate_tags_for_post(post_id, full_text_content)

        # 6. Fetch the newly created post to render it
        row = db.get_post(post_id)
        if not row:
            raise HTTPException(status_code=404, detail="Post created but not found")
            
        post = process_post_for_template(row)
        
        # Render the partial template for the new post
        response = templates.TemplateResponse(
            "_post_card.html",
            {"request": request, "post": post}
        )
        # Send an HTMX event to the browser to show the deletion token
        response.headers["HX-Trigger"] = (
            f'{{"showToken": {{"token": "{token}", "post_id": {post_id}}}}}'
        )
        return response

    except Exception as post_e:
        logging.error(f"Error processing post creation: {post_e}", exc_info=True)
        return HTMLResponse(
            f'<div id="form-error" class="text-red-400 p-2 mb-4">An error occurred: {post_e}</div>',
            status_code=500
        )


@app.post("/delete", response_class=HTMLResponse)
async def delete_post(
    del_id: int = Form(...),
    del_token: str = Form(...)
):
    """
    Handles post deletion.
    On success, redirects back to the main page to refresh the list.
    """
    if not del_token or del_id <= 0:
        # This error should be handled client-side, but good to check
        raise HTTPException(status_code=400, detail="Invalid ID or token")
        
    try:
        ok = db.delete_post_with_token(int(del_id), del_token)
        if ok:
            embeddings.remove_from_index(int(del_id))
            logging.info(f"Post #{del_id} deleted.")
            # Redirect to main page, which forces a refresh
            return RedirectResponse("/", status_code=HTTP_303_SEE_OTHER)
        else:
            # Return an error message to the delete form
            return HTMLResponse(
                '<div id="delete-error" class="text-red-400 p-2 mb-2">❌ Invalid Post ID or Token.</div>',
                status_code=400
            )
    except Exception as del_e:
         logging.error(f"Error during post deletion: {del_e}", exc_info=True)
         return HTMLResponse(
            f'<div id="delete-error" class="text-red-400 p-2 mb-2">An error occurred: {del_e}</div>',
            status_code=500
        )


@app.post("/search", response_class=HTMLResponse)
async def search_posts(request: Request, query: str = Form(...)):
    """
    Handles search.
    Returns a partial HTML snippet of the search results.
    """
    if not query or not query.strip():
        return HTMLResponse("") # Return empty, clearing the results

    rag_answer = None
    rag_sources = []
    rag_used_gemini = False
    rag_error = None
    semantic_results = []

    # 1. Attempt RAG Search
    try:
        rag_result = rag.ask(query, k=5)
        rag_answer = rag_result.get("answer")
        rag_sources = rag_result.get("source_documents", [])
        rag_used_gemini = rag_result.get("used_gemini", False)
        rag_error = rag_result.get("error")
    except Exception as e:
        logging.error(f"Error calling RAG search: {e}", exc_info=True)
        rag_error = str(e)

    # 2. Fallback/Display Semantic Search Results
    try:
        # Use RAG sources if available, otherwise perform a new search
        if rag_sources and not rag_error:
            search_hits = [(s['post_id'], s['score']) for s in rag_sources]
        else:
            search_hits = embeddings.semantic_search(query, k=10)
        
        # Fetch post details for semantic results
        for pid, score in sorted(search_hits, key=lambda x: x[1], reverse=True):
            row = db.get_post(pid)
            if not row: continue
            semantic_results.append({
                "id": row[0],
                "title": html.escape(row[3] or f"Post #{row[0]}"),
                "content_snippet": html.escape(row[1] or "")[:150] + "...",
                "score": score
            })

    except Exception as search_e:
        logging.error(f"Error during semantic search: {search_e}", exc_info=True)
        rag_error = rag_error or str(search_e) # Add this error too

    # Render the search results partial
    return templates.TemplateResponse(
        "_search_results.html",
        {
            "request": request,
            "query": query,
            "rag_answer": rag_answer,
            "rag_used_gemini": rag_used_gemini,
            "rag_error": rag_error,
            "semantic_results": semantic_results
        }
    )


# --- Post Action Endpoints (Like, Dislike, Report) ---

async def handle_post_action(request: Request, post_id: int, action: str):
    """
    Generic handler for like/dislike/report.
    It updates the DB and returns a *new* post card to HTMX.
    """
    try:
        if action == "like":
            db.increment_like(post_id)
        elif action == "dislike":
            db.increment_dislike(post_id)
        elif action == "report":
            db.report_post(post_id)
        else:
            raise HTTPException(status_code=400, detail="Invalid action")

        # Fetch the updated post data
        row = db.get_post(post_id)
        if not row:
            # Post might have been deleted, return no content
            return Response(content="", status_code=200)
            
        post = process_post_for_template(row)
        
        # Return the re-rendered post card
        return templates.TemplateResponse(
            "_post_card.html",
            {"request": request, "post": post}
        )
    except Exception as e:
        logging.error(f"Error handling post action: {e}", exc_info=True)
        # Don't crash, just return an error (or 204 No Content)
        return Response(content=f"Error: {e}", status_code=500)


@app.post("/posts/{post_id}/like", response_class=HTMLResponse)
async def like_post(request: Request, post_id: int):
    return await handle_post_action(request, post_id, "like")

@app.post("/posts/{post_id}/dislike", response_class=HTMLResponse)
async def dislike_post(request: Request, post_id: int):
    return await handle_post_action(request, post_id, "dislike")

@app.post("/posts/{post_id}/report", response_class=HTMLResponse)
async def report_post(request: Request, post_id: int):
    return await handle_post_action(request, post_id, "report")


# --- Main entry point to run the app ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)