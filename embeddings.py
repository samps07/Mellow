# embeddings.py
import numpy as np
from sentence_transformers import SentenceTransformer
import db

# --- LIGHTWEIGHT EMBEDDING MODEL ---
MODEL_NAME = "all-MiniLM-L6-v2"
# Use cache_folder to prevent redownloading on every simple rerun, 
# though Streamlit cloud might clear this occasionally.
embed_model = SentenceTransformer(MODEL_NAME)

def add_to_index(post_id: int, text: str, metadata=None):
    """
    Generate embedding and save to Postgres.
    """
    # Convert to standard list of floats for Postgres compatibility
    emb = embed_model.encode([text], convert_to_numpy=True)[0].tolist()
    db.save_embedding(post_id, emb)

def remove_from_index(post_id: int):
    """Remove from Postgres."""
    db.delete_embedding(post_id)

def semantic_search(query: str, k: int = 5):
    """
    Fetch all embeddings from DB, calculate cosine similarity in memory.
    """
    if not query or not query.strip():
        return []

    # 1. Encode query
    q_emb = embed_model.encode([query], convert_to_numpy=True)[0]
    
    # 2. Fetch all embeddings from DB
    # Returns list of (post_id, embedding_list)
    rows = db.get_all_embeddings()
    if not rows:
        return []
    
    ids = []
    matrix = []
    
    for pid, emb_list in rows:
        ids.append(pid)
        matrix.append(emb_list)
    
    if not matrix:
        return []

    # 3. Calculate Similarity (Vectorized)
    # Convert to numpy array
    doc_embs = np.array(matrix, dtype="float32")
    
    # Normalize query
    norm_q = np.linalg.norm(q_emb)
    if norm_q > 0:
        q_emb = q_emb / norm_q
        
    # Normalize docs
    norms_doc = np.linalg.norm(doc_embs, axis=1, keepdims=True)
    norms_doc[norms_doc == 0] = 1.0
    doc_embs = doc_embs / norms_doc
    
    # Dot product
    sims = doc_embs @ q_emb
    
    # 4. Sort and return top K
    # argsort gives indices of sorted, we take last k and reverse
    top_indices = np.argsort(sims)[-k:][::-1]
    
    results = []
    for idx in top_indices:
        score = float(sims[idx])
        pid = ids[idx]
        results.append((pid, score))
        
    return results

def rebuild_index_from_posts(posts):
    """
    For the updated DB logic, this just loops and ensures embeddings exist.
    """
    for pid, content, ts in posts:
        add_to_index(pid, content)