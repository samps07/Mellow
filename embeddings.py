# embeddings.py
"""
Robust embeddings manager:
- Prefer Chroma (new client style). If Chroma cannot be initialized (legacy config/migration),
  fall back to the original numpy-based index used previously.
- Exposes:
    embed_model (SentenceTransformer instance)
    add_to_index(post_id, text, metadata=None)
    semantic_search(query, k=5) -> list[(post_id:int, score:float)]
    rebuild_index_from_posts(posts)
    remove_from_index(post_id)
"""

import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# --- LIGHTWEIGHT EMBEDDING MODEL ---
# Swapped L12 (130MB) for L6 (70MB) to save RAM
MODEL_NAME = "all-MiniLM-L6-v2"
embed_model = SentenceTransformer(MODEL_NAME)

# Files for local numpy fallback
_EMB_FILE = "embeddings.npy"
_ID_FILE = "id_mapping.pkl"

# Try to use Chroma (best-effort). If Chroma cannot be initialized (legacy config/migration),
# fall back to numpy index.
_USE_CHROMA = False
_chroma_client = None
_chroma_collection = None

try:
    import chromadb
    from chromadb.config import Settings
    # Try the simple client creation first (no legacy Settings keys)
    try:
        # New recommended simple client construction:
        _chroma_client = chromadb.Client()
        # create/get collection
        _chroma_collection = _chroma_client.get_or_create_collection(name="posts")
        _USE_CHROMA = True
    except Exception:
        # If this fails, try a safer Settings attempt without deprecated keys
        try:
            persist_dir = "chroma_db"
            _chroma_client = chromadb.Client(Settings(persist_directory=persist_dir))
            _chroma_collection = _chroma_client.get_or_create_collection(name="posts")
            _USE_CHROMA = True
        except Exception:
            # Any failure -> fall back to numpy approach
            _USE_CHROMA = False
            _chroma_client = None
            _chroma_collection = None
except Exception:
    _USE_CHROMA = False
    _chroma_client = None
    _chroma_collection = None

# --------- Numpy fallback helpers ---------
def _load_numpy_index():
    if not os.path.exists(_EMB_FILE) or not os.path.exists(_ID_FILE):
        return None, []
    embs = np.load(_EMB_FILE)
    ids = pickle.load(open(_ID_FILE, "rb"))
    return embs, ids

def _save_numpy_index(embs: np.ndarray, ids: list):
    np.save(_EMB_FILE, embs)
    pickle.dump(ids, open(_ID_FILE, "wb"))

def _normalize(arr: np.ndarray):
    if arr is None or len(arr) == 0:
        return arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms

# --------- Public API ---------
def add_to_index(post_id: int, text: str, metadata: dict = None):
    """
    Add a post to Chroma (preferred) or to local numpy index (fallback).
    metadata is stored in Chroma only.
    """
    if metadata is None:
        metadata = {}
    if _USE_CHROMA and _chroma_collection is not None:
        try:
            emb = embed_model.encode([text], convert_to_numpy=True)[0].astype("float32").tolist()
            _chroma_collection.add(
                ids=[str(post_id)],
                documents=[text],
                metadatas=[{**metadata, "post_id": post_id}],
                embeddings=[emb]
            )
            # try to persist if client supports
            try:
                _chroma_client.persist()
            except Exception:
                pass
            return
        except Exception:
            # on any chroma error, fallback to numpy below
            pass

    # Fallback numpy append
    embs, ids = _load_numpy_index()
    new = embed_model.encode([text], convert_to_numpy=True)[0].astype("float32")
    if embs is None:
        embs = np.array([new], dtype="float32")
        ids = [post_id]
    else:
        embs = np.vstack([embs, new])
        ids.append(post_id)
    embs = _normalize(embs)
    _save_numpy_index(embs, ids)

def semantic_search(query: str, k: int = 5):
    """
    Return list of (post_id, score) sorted descending by similarity.
    If Chroma available, use it. Otherwise use the numpy fallback.
    Score is between 0..1 (heuristic for Chroma distances -> similarity).
    """
    if not query or not query.strip():
        return []

    # Try Chroma path
    if _USE_CHROMA and _chroma_collection is not None:
        try:
            q_emb = embed_model.encode([query], convert_to_numpy=True)[0].astype("float32").tolist()
            res = _chroma_collection.query(
                query_embeddings=[q_emb],
                n_results=k,
                include=["ids", "distances", "metadatas", "documents"]
            )
            if not res or "ids" not in res:
                return []
            ids = res.get("ids", [[]])[0]
            distances = res.get("distances", [[]])[0] if "distances" in res else []
            out = []
            for i, sid in enumerate(ids):
                try:
                    pid = int(sid)
                except Exception:
                    continue
                score = 0.0
                if distances:
                    d = float(distances[i])
                    score = max(0.0, 1.0 - d)
                out.append((pid, float(score)))
            return out
        except Exception:
            # on any chroma query failure, fallback to numpy
            pass

    # Numpy fallback
    embs, ids = _load_numpy_index()
    if embs is None or len(ids) == 0:
        return []
    q = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    sims = (embs @ q[0]).astype("float32")
    topk = np.argsort(-sims)[:k]
    results = []
    for idx in topk:
        results.append((ids[int(idx)], float(sims[int(idx)])))
    return results

def rebuild_index_from_posts(posts):
    """
    posts: list of (id, full_text, ts) - NOTE: full_text must be provided
    If Chroma available attempt to rebuild collection; otherwise overwrite numpy files.
    """
    global _chroma_collection  # must be declared before any use in function
    if _USE_CHROMA and _chroma_client is not None and _chroma_collection is not None:
        try:
            # delete and recreate collection to ensure clean rebuild (best-effort)
            try:
                _chroma_client.delete_collection(name="posts")
            except Exception:
                pass
            _chroma_collection = _chroma_client.get_or_create_collection(name="posts")
            if not posts:
                return
            ids, docs, metas, embs = [], [], [], []
            for pid, content, ts in posts:
                ids.append(str(pid))
                docs.append(content)
                metas.append({"post_id": pid, "created_at": ts})
                emb = embed_model.encode([content], convert_to_numpy=True)[0].astype("float32").tolist()
                embs.append(emb)
            if ids:
                _chroma_collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
                try:
                    _chroma_client.persist()
                except Exception:
                    pass
                return
        except Exception:
            # fallback to numpy rebuild
            pass

    # numpy rebuild path
    if not posts:
        # remove files if present
        try:
            if os.path.exists(_EMB_FILE):
                os.remove(_EMB_FILE)
            if os.path.exists(_ID_FILE):
                os.remove(_ID_FILE)
        except Exception:
            pass
        return

    ids = []
    embs_list = []
    for pid, content, _ in posts:
        ids.append(pid)
        emb = embed_model.encode([content], convert_to_numpy=True)[0]
        embs_list.append(emb)
    embs = np.array(embs_list, dtype="float32")
    embs = _normalize(embs)
    _save_numpy_index(embs, ids)

def remove_from_index(post_id: int):
    """
    Remove single post from Chroma (if used) or from numpy index (fallback).
    """
    if _USE_CHROMA and _chroma_collection is not None:
        try:
            _chroma_collection.delete(ids=[str(post_id)])
            try:
                _chroma_client.persist()
            except Exception:
                pass
            return
        except Exception:
            pass

    # numpy path: rebuild without the id
    embs, ids = _load_numpy_index()
    if embs is None or not ids:
        return
    try:
        idx = ids.index(post_id)
    except ValueError:
        return
    ids.pop(idx)
    embs = np.delete(embs, idx, axis=0)
    if len(ids) == 0:
        # remove files
        try:
            if os.path.exists(_EMB_FILE):
                os.remove(_EMB_FILE)
            if os.path.exists(_ID_FILE):
                os.remove(_ID_FILE)
        except Exception:
            pass
        return
    embs = _normalize(embs)
    _save_numpy_index(embs, ids)