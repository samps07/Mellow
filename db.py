# db.py
import os
import time
import psycopg2
from typing import List, Optional, Tuple, Any
import streamlit as st

# Function to get connection
def _connect():
    # Try getting URL from Streamlit secrets first, then environment variables
    db_url = None
    if hasattr(st, "secrets") and "DB_URL" in st.secrets:
        db_url = st.secrets["DB_URL"]
    elif "DB_URL" in os.environ:
        db_url = os.environ["DB_URL"]
    
    if not db_url:
        raise ValueError("DB_URL not found in secrets or environment variables.")

    return psycopg2.connect(db_url)

def init_db():
    """Create tables in PostgreSQL if they don't exist."""
    conn = _connect()
    cur = conn.cursor()
    
    # Create Posts Table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            created_at BIGINT NOT NULL,
            token_hash TEXT NOT NULL,
            title TEXT DEFAULT '',
            likes INTEGER DEFAULT 0,
            dislikes INTEGER DEFAULT 0,
            reports INTEGER DEFAULT 0
        );
    """)
    
    # Create Tags Table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS post_tags (
            post_id INTEGER NOT NULL,
            tag TEXT NOT NULL,
            PRIMARY KEY (post_id, tag),
            CONSTRAINT fk_post
                FOREIGN KEY(post_id) 
                REFERENCES posts(id) 
                ON DELETE CASCADE
        );
    """)

    # Create Embeddings Table (storing as array of floats)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS post_embeddings (
            post_id INTEGER PRIMARY KEY,
            embedding FLOAT[] NOT NULL,
            CONSTRAINT fk_post_emb
                FOREIGN KEY(post_id) 
                REFERENCES posts(id) 
                ON DELETE CASCADE
        );
    """)
    
    conn.commit()
    conn.close()

def insert_post(content: str, token_hash: str, title: str = "") -> int:
    conn = _connect()
    cur = conn.cursor()
    ts = int(time.time())
    # Postgres uses %s placeholder and RETURNING id to get the ID
    cur.execute(
        "INSERT INTO posts (content, created_at, token_hash, title) VALUES (%s, %s, %s, %s) RETURNING id",
        (content, ts, token_hash, title)
    )
    post_id = cur.fetchone()[0]
    conn.commit()
    conn.close()
    return post_id

def get_post(post_id: int):
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT id, content, created_at, title, likes, dislikes, reports FROM posts WHERE id = %s", (post_id,))
    row = cur.fetchone()
    conn.close()
    return row

def list_recent(limit: int = 30):
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, content, created_at, title, likes, dislikes, reports FROM posts ORDER BY created_at DESC LIMIT %s",
        (limit,)
    )
    rows = cur.fetchall()
    conn.close()
    return rows

def get_token_hash(post_id: int) -> Optional[str]:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT token_hash FROM posts WHERE id = %s", (post_id,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None

def delete_post_with_token(post_id: int, provided_token: str) -> bool:
    import utils # local import to avoid circular dependency
    if not provided_token: 
        return False
    
    stored = get_token_hash(post_id)
    if not stored:
        return False
        
    provided_hash = utils.hmac_hash(provided_token.strip())
    
    import hmac
    if hmac.compare_digest(stored, provided_hash):
        conn = _connect()
        cur = conn.cursor()
        cur.execute("DELETE FROM posts WHERE id = %s", (post_id,))
        conn.commit()
        conn.close()
        return True
    return False

def get_all_posts():
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT id, content, created_at FROM posts ORDER BY created_at ASC")
    rows = cur.fetchall()
    conn.close()
    return rows

# ---------- Tag helpers ----------

def set_tags_for_post(post_id: int, tags: List[str]):
    conn = _connect()
    cur = conn.cursor()
    cur.execute("DELETE FROM post_tags WHERE post_id = %s", (post_id,))
    for t in tags:
        cur.execute(
            "INSERT INTO post_tags (post_id, tag) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (post_id, t)
        )
    conn.commit()
    conn.close()

def get_tags_for_post(post_id: int) -> List[str]:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT tag FROM post_tags WHERE post_id = %s ORDER BY tag ASC", (post_id,))
    rows = cur.fetchall()
    conn.close()
    return [r[0] for r in rows]

# ---------- Like/Dislike/Report helpers ----------

def increment_like(post_id: int):
    conn = _connect()
    cur = conn.cursor()
    cur.execute("UPDATE posts SET likes = COALESCE(likes,0) + 1 WHERE id = %s", (post_id,))
    conn.commit()
    conn.close()

def increment_dislike(post_id: int):
    conn = _connect()
    cur = conn.cursor()
    cur.execute("UPDATE posts SET dislikes = COALESCE(dislikes,0) + 1 WHERE id = %s", (post_id,))
    conn.commit()
    conn.close()

def report_post(post_id: int):
    conn = _connect()
    cur = conn.cursor()
    cur.execute("UPDATE posts SET reports = COALESCE(reports,0) + 1 WHERE id = %s", (post_id,))
    conn.commit()
    conn.close()

# ---------- Embedding helpers (New for Postgres) ----------

def save_embedding(post_id: int, embedding: List[float]):
    """Saves the embedding list to Postgres."""
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO post_embeddings (post_id, embedding) VALUES (%s, %s) ON CONFLICT (post_id) DO UPDATE SET embedding = EXCLUDED.embedding",
        (post_id, embedding)
    )
    conn.commit()
    conn.close()

def get_all_embeddings() -> List[Tuple[int, List[float]]]:
    """Fetches all embeddings to perform cosine similarity in Python."""
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT post_id, embedding FROM post_embeddings")
    rows = cur.fetchall()
    conn.close()
    return rows

def delete_embedding(post_id: int):
    # Cascading delete in SQL handles this usually, but safe to have.
    conn = _connect()
    cur = conn.cursor()
    cur.execute("DELETE FROM post_embeddings WHERE post_id = %s", (post_id,))
    conn.commit()
    conn.close()