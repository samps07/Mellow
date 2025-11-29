# tagging.py
"""
Improved tagging:
- Primary: zero-shot NLI classifier for topic labels (configurable model)
- Fallback: embedding-based topic assignment (previous approach)
- Keyword extraction: KeyBERT if available, else TF-IDF fallback
- Writes tags into DB via db.set_tags_for_post(...)

Refactored to support single-post tagging to avoid full re-computation.
"""

from typing import List
import numpy as np
import logging
import re
import os
import db
import embeddings  # uses embed_model
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO)

# Try KeyBERT
try:
    from keybert import KeyBERT
    KW_AVAILABLE = True
    # Note: this automatically uses the lightweight 'all-MiniLM-L6-v2' from embeddings.py
    _kw_model = KeyBERT(embeddings.embed_model)
except Exception:
    KW_AVAILABLE = False
    _kw_model = None

# --- LIGHTWEIGHT NLI MODEL ---
# Using a tiny 17MB model for background tagging.
NLI_MODEL = os.getenv("NLI_MODEL", "m-polignano-uniba/mnli_in_tiny_bert")
try:
    from transformers import pipeline
    _nli_pipeline = pipeline("zero-shot-classification", model=NLI_MODEL)
    ZS_AVAILABLE = True
except Exception:
    _nli_pipeline = None
    ZS_AVAILABLE = False

# Topic labels (customize)
TOPIC_LABELS = [
    "politics", "login", "payment", "account", "installation", "bug", "crash",
    "performance", "privacy", "security", "feature", "ui", "settings",
    "shopping", "education", "health", "travel", "game", "social", "reporting",
    "deployment", "integration", "billing", "subscription", "refund", "network"
]

# Extra stopwords and token regex
_EXTRA_STOPWORDS = {
    "actually","really","please","thanks","thank","also","just","still",
    "one","like","would","could","get","got","say","says","well","hi","hello"
}
_TOKEN_RE = re.compile(r"[A-Za-z0-9\-\_']+")

def _clean_phrase(p: str) -> str:
    p = p.strip().lower()
    p = re.sub(r"\s+", " ", p)
    return p

# ---- Keyword extraction ----
def extract_keywords_keybert(doc: str, top_n: int = 6):
    if not KW_AVAILABLE or not doc or not doc.strip():
        return []
    try:
        kws = _kw_model.extract_keywords(doc, keyphrase_ngram_range=(1,3), stop_words='english',
                                         use_mmr=True, diversity=0.6, top_n=top_n)
    except Exception:
        return []
    phrases = []
    for p, _ in kws:
        p = _clean_phrase(p)
        if len(p) >= 1 and re.search(r"[A-Za-z0-9]", p):
            phrases.append(p)
    return phrases

def extract_keywords_tfidf_fallback(doc: str, top_n: int = 6):
    if not doc or not doc.strip():
        return []
    vec = TfidfVectorizer(max_df=0.85, stop_words='english', ngram_range=(1,3), max_features=1000)
    try:
        X = vec.fit_transform([doc])
    except Exception:
        return []
    feature_names = np.array(vec.get_feature_names_out())
    row = X.getrow(0).toarray().ravel()
    if row.sum() == 0:
        return []
    top_idx = row.argsort()[-(top_n*4):][::-1]
    candidates = []
    for j in top_idx:
        token = feature_names[j]
        token_clean = _clean_phrase(token)
        if len(token_clean) < 1:
            continue
        if token_clean in _EXTRA_STOPWORDS:
            continue
        if len(token_clean) == 1 and not token_clean.isalpha():
            continue
        if not _TOKEN_RE.search(token_clean):
            continue
        candidates.append(token_clean)
        if len(candidates) >= top_n:
            break
    return candidates

def extract_keywords_for_doc(doc: str, top_n: int = 6) -> List[str]:
    if not doc or not doc.strip():
        return []
    if KW_AVAILABLE:
        kws = extract_keywords_keybert(doc, top_n=top_n)
        if kws:
            return kws[:top_n]
    return extract_keywords_tfidf_fallback(doc, top_n=top_n)

# ---- Topic assignment ----
_label_emb_cache = None
def _get_label_embeddings():
    global _label_emb_cache
    if _label_emb_cache is None:
        _label_emb_cache = embeddings.embed_model.encode(TOPIC_LABELS, convert_to_numpy=True)
        norms = np.linalg.norm(_label_emb_cache, axis=1, keepdims=True)
        norms[norms==0] = 1.0
        _label_emb_cache = _label_emb_cache / norms
    return _label_emb_cache

def assign_topic_labels_with_nli(docs: List[str], threshold: float = 0.5) -> List[str]:
    """
    Use zero-shot classification to choose the best topic label.
    If best score < threshold, return 'misc'.
    """
    if not ZS_AVAILABLE or not docs:
        return ["misc"] * len(docs)
    results = []
    for doc in docs:
        if not doc or not doc.strip():
            results.append("misc")
            continue
        try:
            res = _nli_pipeline(doc, TOPIC_LABELS, hypothesis_template="This text is about {}.")
            best_label = res['labels'][0]
            best_score = float(res['scores'][0])
            if best_score >= threshold:
                results.append(best_label)
            else:
                results.append("misc")
        except Exception:
            results.append("misc")
    return results

def assign_topic_labels_with_embeddings(docs: List[str], threshold: float = 0.45) -> List[str]:
    if not docs:
        return []
    label_embs = _get_label_embeddings()
    doc_embs = embeddings.embed_model.encode(docs, convert_to_numpy=True)
    den = np.linalg.norm(doc_embs, axis=1, keepdims=True)
    den[den==0] = 1.0
    doc_norm = doc_embs / den
    sims = doc_norm @ label_embs.T
    best_idx = np.argmax(sims, axis=1)
    best_scores = sims[np.arange(sims.shape[0]), best_idx]
    topics = []
    for idx, score in zip(best_idx, best_scores):
        if float(score) >= threshold:
            topics.append(TOPIC_LABELS[int(idx)])
        else:
            topics.append("misc")
    return topics

def assign_topic_labels(docs: List[str], nli_threshold: float = 0.5, emb_threshold: float = 0.45) -> List[str]:
    """
    Try NLI first (configurable). If NLI not confident, fallback to embeddings.
    """
    if ZS_AVAILABLE:
        nli_res = assign_topic_labels_with_nli(docs, threshold=nli_threshold)
        # For items marked 'misc' by NLI, try embedding fallback
        to_fallback_idx = [i for i, v in enumerate(nli_res) if v == "misc"]
        if not to_fallback_idx:
            return [r.lower() for r in nli_res]
        fallback_docs = [docs[i] for i in to_fallback_idx]
        emb_res = assign_topic_labels_with_embeddings(fallback_docs, threshold=emb_threshold)
        final = []
        fi = 0
        for i in range(len(docs)):
            if nli_res[i] != "misc":
                final.append(nli_res[i].lower())
            else:
                final.append(emb_res[fi].lower() if emb_res and emb_res[fi] else "misc")
                fi += 1
        return final
    return [t.lower() for t in assign_topic_labels_with_embeddings(docs, threshold=emb_threshold)]

# ---- NEW: Generate tags for a single post ----
def generate_tags_for_post(post_id: int, doc: str, top_n_keywords: int = 5, topic_threshold: float = 0.5):
    """
    Efficiently generates and saves tags for ONE new post.
    Called by app.py after a new post is inserted.
    """
    if not doc or not doc.strip():
        return

    try:
        # 1. Get keywords
        kws = extract_keywords_for_doc(doc, top_n=top_n_keywords)
        
        # 2. Get topic
        # assign_topic_labels expects a list, so wrap doc in []
        topic = assign_topic_labels([doc], nli_threshold=topic_threshold, emb_threshold=topic_threshold)[0]
        
        # 3. Combine and save
        tag_set = []
        if topic:
            tag_set.append(f"topic:{topic.lower()}")
        for k in kws:
            t = k.strip().lower()
            if t:
                tag_set.append(t)
        
        # dedupe preserving order
        seen = set()
        final_tags = []
        for t in tag_set:
            if t and t not in seen:
                final_tags.append(t)
                seen.add(t)
        
        db.set_tags_for_post(post_id, final_tags)
        logging.info(f"Generated tags for new post #{post_id}: {final_tags}")
        
    except Exception as e:
        logging.error(f"Failed to generate tags for post #{post_id}: {e}")

# ---- Recompute tags for all posts (for manual execution) ----
def recompute_all_tags(top_n_keywords: int = 5, topic_threshold: float = 0.5):
    """
    Script-level function to re-process ALL posts.
    Do NOT call this from the web app on every change.
    """
    posts = db.get_all_posts()
    if not posts:
        return
    
    ids = []
    docs = []
    for pid, content, _ in posts:
        row = db.get_post(pid) 
        if not row:
            continue
        
        _id, post_content, _ts, post_title, *_ = row
        full_text = f"{(post_title or '').strip()}\n\n{(post_content or '').strip()}"
        ids.append(_id)
        docs.append(full_text)
        
    keywords_list = [extract_keywords_for_doc(d, top_n=top_n_keywords) for d in docs]
    topics = assign_topic_labels(docs, nli_threshold=topic_threshold, emb_threshold=topic_threshold)
    
    for post_id, kws, topic in zip(ids, keywords_list, topics):
        tag_set = []
        if topic:
            tag_set.append(f"topic:{topic.lower()}")
        for k in kws:
            t = k.strip().lower()
            if t:
                tag_set.append(t)
        
        seen = set()
        final_tags = []
        for t in tag_set:
            if t and t not in seen:
                final_tags.append(t)
                seen.add(t)
        db.set_tags_for_post(post_id, final_tags)
    logging.info("Recomputed tags for %d posts", len(posts))


if __name__ == "__main__":
    logging.info("Running manual re-computation of all tags...")
    recompute_all_tags()
    logging.info("Manual re-computation complete.")