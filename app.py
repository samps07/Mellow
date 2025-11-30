# app.py
# --- Imports ---
import streamlit as st # Streamlit library for building the web app
from dotenv import load_dotenv # Loads environment variables from a .env file
from transformers import pipeline # Hugging Face library for ML pipelines (sentiment analysis)
import datetime # Standard library for handling dates and times
import html # Standard library for escaping HTML characters
import logging # Standard library for logging information and errors

# --- Local Modules ---
# Import custom modules for different functionalities
import db # Handles database interactions (posts, tags, etc.)
import embeddings # Manages text embeddings and semantic search
import moderation # Handles content moderation (profanity check, Gemini safety)
import tagging # Generates tags (keywords, topics) for posts
import rag # Handles Retrieval-Augmented Generation (RAG) search using Gemini
from utils import generate_token, hmac_hash # Utility functions for token generation and hashing

# --- Initialization ---
load_dotenv() # Load environment variables from .env file (must be called early)
db.init_db() # Initialize the database (create tables if they don't exist)
st.set_page_config(page_title="Mellow", layout="wide", initial_sidebar_state="collapsed") # Configure Streamlit page

# Configure logging
logging.basicConfig(level=logging.INFO) # Set logging level to INFO

# --- Load ML Model (Cached) ---
# Use st.cache_resource to load the sentiment analysis model once and reuse it across sessions
@st.cache_resource
def load_sentiment():
    """Loads the sentiment analysis pipeline using Hugging Face Transformers."""
    # Using a standard DistilBERT model fine-tuned for sentiment (POSITIVE/NEGATIVE)
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    logging.info(f"Loading sentiment model: {model_name}")
    try:
        # Initialize the pipeline
        sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)
        logging.info("Sentiment model loaded successfully.")
        return sentiment_pipeline
    except Exception as e:
        # Log error and display a message in the app if model loading fails
        logging.error(f"Failed to load sentiment model: {e}", exc_info=True)
        st.error(f"Error loading sentiment analysis model: {e}")
        return None # Return None if loading fails

# Attempt to load the model; will be None if loading failed
sentiment_pipeline = load_sentiment()

# --- Initialize Session State Variables ---
# Use session state to keep track of UI state across reruns
st.session_state.setdefault("show_post_form", False) # Tracks visibility of the create post form
st.session_state.setdefault("show_delete_form", False) # Tracks visibility of the delete post form
st.session_state.setdefault("last_post_id", None) # Stores the ID of the last successfully created post
st.session_state.setdefault("last_token", None) # Stores the deletion token of the last post

# --- Utility Function to Clear Last Post Info ---
def clear_last_post_info():
    """Removes the last post ID and token from session state."""
    st.session_state.last_post_id = None
    st.session_state.last_token = None

# ---------- CSS Styling ----------
# Apply custom CSS for styling elements like tags, buttons, layout, etc.
st.markdown(
    """
    <style>
    /* Adjust container padding for better spacing */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; padding-left: 1.5rem; padding-right: 1.5rem; }

    /* Custom styles for tags */
    .tag { display:inline-block; padding: 4px 8px; border-radius:12px; background:#f0f2f6; margin-right:4px; margin-bottom:4px; font-size: 0.75rem; color:#333; line-height: 1.2; }
    .topic { display:inline-block; padding: 4px 8px; border-radius:12px; background:#e6f0ff; margin-right:4px; margin-bottom:4px; font-size: 0.75rem; color:#234; line-height: 1.2; }
    .sent-pos { display:inline-block; padding: 4px 8px; border-radius:12px; background: #dff7e9; margin-right:4px; margin-bottom:4px; font-size: 0.75rem; color: #0d6b3a; border: 1px solid #b3efd1; line-height: 1.2; }
    .sent-neg { display:inline-block; padding: 4px 8px; border-radius:12px; background: #fff0f0; margin-right:4px; margin-bottom:4px; font-size: 0.75rem; color: #c92a2a; border: 1px solid #ffdcdc; line-height: 1.2; }

    /* Compact emoji action buttons */
    .stButton>button {
        padding: 4px !important; /* Minimal padding */
        font-size: 1rem !important;
        border-radius: 8px !important;
        width: 34px !important; /* Slightly smaller */
        height: 34px !important;
        border: none;
        background-color: transparent;
        transition: background-color 0.2s ease; /* Smooth hover effect */
    }
    .stButton>button:hover {
        background-color: rgba(240, 242, 246, 0.5); /* Lighter hover */
    }
    /* Style for the like/dislike/report counts */
    .small-count { text-align:center; font-size: 0.7rem; color:#999; margin-top:-6px; } /* Adjusted color and margin */

    /* Style overrides for specific Streamlit components */
    div[data-testid="stForm"] { border: 1px solid #444; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;} /* Add border and bottom margin to forms */
    div[data-testid="stExpander"] details { border: none; box-shadow: none; background-color: #222; border-radius: 4px;} /* Cleaner expander styling */
    div[data-testid="stExpander"] summary { font-size: 0.9rem; color: #ccc;}

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Header ----------
# Display the application title and subtitle using Markdown
st.markdown('<h1 style="font-size:48px; text-align:center; color:white; margin-bottom:0; font-weight: 600;">Mellow</h1>', unsafe_allow_html=True)
st.markdown('<p style="font-size:12px; text-align:center; color:#bbb; margin-top:0; margin-bottom: 2rem;">A Privacy-Focused, AI-Enhanced Discussion Board</p>', unsafe_allow_html=True)

# ---------- Top Row Buttons (Create/Delete Toggle) ----------
# Use columns for layout control, pushing buttons towards center/edges
left_col_top, _, right_col_top = st.columns([1, 6, 1], gap="small")
with left_col_top:
    # Button to toggle the create post form visibility
    if st.button("Create Post ➕", key="toggle_post", help="Write a new anonymous post"):
        clear_last_post_info() # Clear previous post token message when opening form
        st.session_state.show_post_form = not st.session_state.show_post_form
        st.session_state.show_delete_form = False # Ensure delete form is hidden

with right_col_top:
    # Button to toggle the delete post form visibility
    if st.button("Delete Post 🗑️", key="toggle_delete", help="Remove one of your posts"):
        clear_last_post_info() # Clear previous post token message when opening form
        st.session_state.show_delete_form = not st.session_state.show_delete_form
        st.session_state.show_post_form = False # Ensure create form is hidden


# ---------- Display Deletion Token (Persists until acknowledged) ----------
# This block is *outside* the form logic, displayed if token info is in session state
if st.session_state.last_post_id is not None and st.session_state.last_token is not None:
    st.success("✅ Post created anonymously!") # Confirmation
    st.warning("🔒 **IMPORTANT: Save your deletion token below. It is shown only ONCE!**") # Warning
    # Display the Post ID and the generated token
    st.code(f"Post ID: {st.session_state.last_post_id}\nDeletion token: {st.session_state.last_token}", language=None)
    # Button to explicitly acknowledge and clear the token message from session state
    if st.button("I have saved my token", key="ack_token", help="Click after saving the token"):
        clear_last_post_info() # Remove token info from state
        st.rerun() # Rerun to remove the message from the UI

# ---------- Create Post Form ----------
# Display the form only if the corresponding session state flag is True
if st.session_state.show_post_form:
    with st.form("create_post_form"):
        st.subheader("✍️ Create New Post") # Form title
        # Input field for the post title (optional)
        post_title = st.text_input("Title (optional)", placeholder="Keep it short and relevant")
        # Text area for the main post content
        post_text_input = st.text_area("Body", height=150, placeholder="Share your tip, question, or experience...")

        # Toggle switch to enable/disable Gemini grammar correction
        use_geminify = st.toggle("✨ Auto-fix Grammar (Geminify)", value=False, help="Use AI to automatically correct spelling and grammar errors before posting.")

        # Submit button for the form
        submitted = st.form_submit_button("Post Anonymously")

        # --- Process Form Submission ---
        if submitted:
            # Get the raw text from the text area
            final_post_text = post_text_input

            # Apply Gemini grammar fix if toggled and text exists
            if use_geminify and final_post_text and final_post_text.strip():
                with st.spinner("✨ Applying grammar fixes..."):
                    try:
                        # Call the grammar correction function from rag.py
                        fixed_text = rag.fix_grammar(final_post_text)
                    except Exception as gemini_e:
                        # Log error but don't prevent posting if Geminify fails
                        logging.error(f"Geminify failed during post creation: {gemini_e}", exc_info=True)
                        st.warning("Could not apply grammar fixes due to an error. Posting original text.")

            # Check content length *after* potential Geminify
            if not final_post_text or len(final_post_text.strip()) < 3:
                 st.error("⚠️ Post body is too short.")
            else:
                # Check moderation status
                allowed, reason = moderation.simple_moderation(final_post_text)
                if not allowed:
                    st.error(f"⚠️ Post blocked: {reason}") # Show error if blocked by moderation
                else:
                    # Proceed if content is allowed and long enough
                    try:
                        # Generate a unique token and its hash for deletion
                        token = generate_token()
                        token_hash = hmac_hash(token)

                        # Insert the post into the database
                        post_id = db.insert_post(content=final_post_text, token_hash=token_hash, title=post_title or "")
                        logging.info(f"New post #{post_id} created successfully.")

                        # Combine title and text for embedding and tagging
                        full_text_content = f"{(post_title or '').strip()}\n\n{final_post_text.strip()}"

                        # --- Efficient Single-Item Processing (Post-DB Insert) ---
                        # Add embedding for the new post only
                        embeddings.add_to_index(post_id, full_text_content)
                        # Generate tags for the new post only
                        tagging.generate_tags_for_post(post_id, full_text_content)

                        # Store post ID and token in session state to display persistently
                        st.session_state.last_post_id = post_id
                        st.session_state.last_token = token

                        # Hide the form after successful submission
                        st.session_state.show_post_form = False
                        st.rerun() # Rerun to display the token message outside the form

                    except Exception as post_e:
                        # Handle potential errors during DB insertion or indexing
                        logging.error(f"Error processing post creation after submission: {post_e}", exc_info=True)
                        st.error(f"An error occurred while saving the post: {post_e}")

# ---------- Delete Post Form ----------
# Display the form only if the corresponding session state flag is True
if st.session_state.show_delete_form:
    with st.form("global_delete_form"):
        st.subheader("🗑️ Delete Your Post") # Form title
        # Input fields for post ID and the corresponding deletion token
        del_id = st.number_input("Post ID to Delete", min_value=1, step=1, help="The ID number of the post you want to delete.")
        # Use type="password" to obscure the token input
        del_token = st.text_input("Deletion Token", type="password", placeholder="Paste the token saved during posting", help="The unique token provided when you created the post.")

        # Submit button for deletion
        do_delete = st.form_submit_button("Confirm Deletion")

        # --- Process Deletion Request ---
        if do_delete:
            if not del_token:
                st.error("⚠️ Deletion token is required.") # Validate token presence
            elif del_id <= 0:
                 st.error("⚠️ Please enter a valid Post ID.") # Validate ID
            else:
                try:
                    # Attempt to delete the post using ID and token via db module
                    ok = db.delete_post_with_token(int(del_id), del_token)
                    if ok:
                        # --- Efficient Single-Item Processing ---
                        # Remove the specific post's embedding from the index
                        embeddings.remove_from_index(int(del_id))
                        # (Tags are automatically deleted via database CASCADE constraint)

                        st.success(f"✅ Post #{del_id} deleted successfully.") # Show success message
                        st.session_state.show_delete_form = True # Hide the form
                        st.rerun() # Rerun to update the post list
                    else:
                        st.error("❌ Invalid Post ID or Deletion Token.") # Show error if deletion fails
                except Exception as del_e:
                     # Handle potential errors during deletion
                     logging.error(f"Error during post deletion: {del_e}", exc_info=True)
                     st.error(f"An error occurred while deleting the post: {del_e}")

# ---------- Search Form ----------
# Always display the search form, slightly separated
st.markdown("---")
with st.form("search_form"):
    # Input field for search query
    query = st.text_input("Search Posts", key="search_input", placeholder="Search posts by keyword or topic...", label_visibility="collapsed")
    # Submit button for search
    submitted_search = st.form_submit_button("Search 🔍")

# ---------- Search Results Processing ----------
# Execute search if form submitted and query is not empty/whitespace
if submitted_search and query and query.strip():
    clear_last_post_info() # Clear any lingering post token info when searching
    st.markdown("---") # Separator before results
    st.subheader(f"Search Results for: \"{html.escape(query)}\"") # Display the escaped search query

    rag_answer = None
    rag_sources = []
    rag_used_gemini = False
    rag_error = None

    # --- Attempt RAG Search ---
    try:
        rag_result = rag.ask(query, k=5) # Perform RAG query (k=5 sources max)
        rag_answer = rag_result.get("answer")
        rag_sources = rag_result.get("source_documents", []) # Sources used/retrieved by RAG
        rag_used_gemini = rag_result.get("used_gemini", False)
        rag_error = rag_result.get("error") # Capture specific RAG errors

        # Display generated answer if RAG worked and provided one
        if rag_used_gemini and rag_answer and "I don't know" not in rag_answer:
            st.success("🤖 AI-Generated Answer (based on posts):") # Indicate AI involvement
            st.markdown(rag_answer) # Display the answer
            # Optionally show sources used for the RAG answer in an expander
            if rag_sources:
                with st.expander("Show Sources Consulted by AI"):
                    for s in sorted(rag_sources, key=lambda x: x['score'], reverse=True): # Sort sources by score
                        pid = s.get("post_id")
                        txt = s.get("text", "")
                        score = s.get("score", 0.0)
                        st.caption(f"Post #{pid} (Similarity: {score:.2f}): {txt[:150]}...")
            st.markdown("---") # Separator

        # Handle cases where RAG failed or didn't find an answer
        elif rag_error:
             logging.error(f"RAG ask function failed: {rag_error}")
             st.warning(f"Could not generate an answer due to an error:{rag_error}. Showing relevant posts instead.")
        elif not rag_used_gemini and rag.GEMINI_API_KEY: # Only show if key exists but RAG didn't run (e.g., client init failed)
             st.info("AI answer generation unavailable. Showing relevant posts.")
        elif "I don't know" in (rag_answer or ""):
             st.info("AI could not determine an answer from the posts. Showing most relevant posts:")
        # If no key, don't show any RAG message, just proceed to semantic search

    except Exception as e:
        # Catch errors during the rag.ask call itself
        logging.error(f"Error calling RAG search: {e}", exc_info=True)
        st.error(f"An error occurred during the AI search phase: {e}")

    # --- Fallback/Display Semantic Search Results ---
    # Display semantic results if RAG wasn't used, failed, or didn't know
    # Also useful to show even if RAG worked, as "Related Posts"
    if not (rag_used_gemini and rag_answer and "I don't know" not in (rag_answer or "")):
         st.markdown("#### Most Relevant Posts (Semantic Search):") # Subheading for semantic results
    else:
         st.markdown("#### Related Posts (Semantic Search):") # Different heading if RAG worked

    try:
        # Perform semantic search (or use sources from RAG if available and error-free)
        # Use RAG sources if available and relevant, otherwise perform a new search
        semantic_results = embeddings.semantic_search(query, k=10) if not rag_sources or rag_error else [(s['post_id'], s['score']) for s in rag_sources]

        if not semantic_results:
            st.write("No relevant posts found.")
        else:
            # Display each semantic search result card
            # Sort results by score before displaying
            for pid, score in sorted(semantic_results, key=lambda x: x[1], reverse=True):
                row = db.get_post(pid) # Fetch post details
                if not row: continue # Skip if post deleted

                _id, content, ts, title, likes, dislikes, reports = row
                esc_title = html.escape(title or f"Post #{_id}")
                esc_content = html.escape(content or "")

                # Display post preview in a container
                with st.container(border=True):
                    st.markdown(f"**{esc_title}**") # Display title
                    st.caption(f"Post #{_id} | Relevance Score: {score:.3f}") # Show ID and score
                    # Show a snippet of the content using st.markdown for better formatting potential
                    st.markdown(f"<div style='max-height: 100px; overflow: hidden; text-overflow: ellipsis; font-size: 0.9em; color: #eee;'>{esc_content}</div>", unsafe_allow_html=True)

    except Exception as search_e:
        logging.error(f"Error during semantic search display: {search_e}", exc_info=True)
        st.error(f"An error occurred while retrieving search results: {search_e}")

    st.markdown("---") # Separator after all search results


# ---------- Recent Posts Display ----------
st.subheader("💬 Recent Posts") # Section header with emoji
try:
    # Fetch recent posts from the database (up to 50)
    rows = db.list_recent(limit=50)
except Exception as db_e:
    logging.error(f"Failed to fetch recent posts: {db_e}", exc_info=True)
    st.error(f"Could not load recent posts: {db_e}")
    rows = [] # Ensure rows is an empty list on error

# Display message if no posts exist
if not rows:
    st.info("No posts have been created yet. Be the first! ☝️")
else:
    # Loop through each recent post and display it as a card
    for row in rows:
        # --- Extract Data from DB Row ---
        pid, content, ts, title, likes, dislikes, reports = row

        # --- Process Data (Sentiment, Tags, Timestamp) ---
        label = "neutral" # Default sentiment
        # Only run sentiment if the pipeline loaded successfully
        if sentiment_pipeline:
            try:
                # Get sentiment result from the cached pipeline
                s = sentiment_pipeline(content)[0]
                raw_label = s.get("label", "").upper() # POSITIVE or NEGATIVE
                score = float(s.get("score", 0.0))
                # Heuristic: Classify as neutral if confidence score is below threshold
                # This helps filter out weak positive/negative classifications
                if score < 0.75: label = "neutral" # Tune threshold (0.6 - 0.8 is common)
                elif raw_label == "POSITIVE": label = "positive"
                elif raw_label == "NEGATIVE": label = "negative"
                # else remains 'neutral'
            except Exception as sent_e:
                logging.warning(f"Sentiment analysis failed for post #{pid}: {sent_e}") # Log errors but don't crash

        # Get tags associated with the post from the database
        tags = db.get_tags_for_post(pid)

        # Format timestamp for display
        try:
            dt = datetime.datetime.fromtimestamp(int(ts))
            ts_str = dt.strftime("%b %d, '%y %H:%M") # Shorter format e.g., Oct 26, '25 13:47
        except Exception:
            ts_str = "Invalid date" # Fallback

        # Escape HTML characters for safe rendering
        esc_title = html.escape(title or "")
        esc_content = html.escape(content or "")

        # --- Build HTML Snippets for Tags ---
        topic_html = "" # For 'topic:' tags
        tags_html = "" # For other tags
        for t in tags:
            if t.startswith("topic:"):
                topic_text = html.escape(t.split("topic:")[-1])
                topic_html += f'<span class="topic">{topic_text}</span>'
            else:
                tag_text = html.escape(t)
                tags_html += f'<span class="tag">{tag_text}</span>'

        # Generate HTML snippet for the sentiment tag
        if label == "positive": sent_html = '<span class="sent-pos">Positive</span>'
        elif label == "negative": sent_html = '<span class="sent-neg">Negative</span>'
        else: sent_html = '<span class="tag">Neutral</span>'

        # --- Render Post Card using st.container ---
        with st.container(border=True): # Adds a visual border around the post
            # Row 1: Header (Post ID, Title, Sentiment, Topic)
            col1_head, col2_head = st.columns([7, 3]) # Asymmetric columns
            with col1_head:
                st.markdown(f"**#{pid}**") # Display Post ID boldly
                if esc_title:
                    # Display Title, prevent large margins
                    st.markdown(f"<h6 style='margin-bottom: 0; margin-top: 0; line-height: 1.3; font-weight: 500;'>{esc_title}</h6>", unsafe_allow_html=True)
            with col2_head:
                # Display sentiment and topic tags, right-aligned
                st.markdown(f"<div style='text-align: right;'>{sent_html}{topic_html}</div>", unsafe_allow_html=True)

            # Row 2: Content
            # Use pre-wrap to preserve line breaks, add appropriate margins
            st.markdown(f"<p style='white-space: pre-wrap; margin-top: 8px; margin-bottom: 12px; font-size: 0.95em;'>{esc_content}</p>", unsafe_allow_html=True)

            # Row 3: General Tags (if they exist) - placed before the final row
            if tags_html:
                # Add bottom margin for spacing
                st.markdown(f"<div style='margin-bottom: 10px;'>{tags_html}</div>", unsafe_allow_html=True)

            # Row 4: Timestamp and Action Buttons (Combined on one line)
            ts_col, action_col_container = st.columns([4, 1]) # Timestamp takes more space
            with ts_col:
                # Display formatted timestamp, smaller and grayed out
                st.markdown(f"<span style='font-size: 0.8em; color: #888;'>{ts_str}</span>", unsafe_allow_html=True)
            with action_col_container:
                # Use sub-columns for compact action buttons, adjust spacing with first ratio
                # Ratios: [Space, Like, Dislike, Report]
                _, like_col, dislike_col, report_col = st.columns([1, 1, 1, 1], gap="small")

                # Like Button & Count
                with like_col:
                    if st.button("👍", key=f"like_{pid}", help="Like this post"):
                        db.increment_like(pid)
                        st.rerun() # Update UI immediately
                    st.markdown(f"<div class='small-count'>{likes}</div>", unsafe_allow_html=True)

                # Dislike Button & Count
                with dislike_col:
                    if st.button("👎", key=f"dislike_{pid}", help="Dislike this post"):
                        db.increment_dislike(pid)
                        st.rerun() # Update UI immediately
                    st.markdown(f"<div class='small-count'>{dislikes}</div>", unsafe_allow_html=True)

                # Report Button & Count
                with report_col:
                    if st.button("⚠️", key=f"report_{pid}", help="Report this post (for review)"):
                        db.report_post(pid)
                        st.rerun() # Update UI immediately
                    st.markdown(f"<div class='small-count'>{reports}</div>", unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown("---") # Final separator line
# Simple footer caption
st.caption("Mellow © 2025 | Anonymous discussions, AI-enhanced.")
