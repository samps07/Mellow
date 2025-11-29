# db.py
import sqlite3
import time
from typing import List, Optional, Tuple
import hmac as _hmac  # for safe compare
import utils  # local utils (hmac_hash, etc.)

DB_PATH = "posts.db"


def _connect():
    """
    Return a sqlite3.Connection with foreign keys enabled.
    Use this helper everywhere so behavior is consistent.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    """
    Create tables if not exists. If schema older, attempt safe ALTERs to add new columns.
    New columns added: title TEXT, likes INTEGER DEFAULT 0, dislikes INTEGER DEFAULT 0, reports INTEGER DEFAULT 0
    """
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            token_hash TEXT NOT NULL
        )
    """)
    # ensure tag table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS post_tags (
            post_id INTEGER NOT NULL,
            tag TEXT NOT NULL,
            PRIMARY KEY (post_id, tag),
            FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE
        )
    """)
    conn.commit()

    # now ensure additional columns exist (safe ALTER)
    cur.execute("PRAGMA table_info(posts)")
    cols = [r[1] for r in cur.fetchall()]  # second column is name
    # add title
    if "title" not in cols:
        try:
            cur.execute("ALTER TABLE posts ADD COLUMN title TEXT DEFAULT ''")
        except Exception:
            pass
    # add likes
    if "likes" not in cols:
        try:
            cur.execute("ALTER TABLE posts ADD COLUMN likes INTEGER DEFAULT 0")
        except Exception:
            pass
    # add dislikes
    if "dislikes" not in cols:
        try:
            cur.execute("ALTER TABLE posts ADD COLUMN dislikes INTEGER DEFAULT 0")
        except Exception:
            pass
    # add reports
    if "reports" not in cols:
        try:
            cur.execute("ALTER TABLE posts ADD COLUMN reports INTEGER DEFAULT 0")
        except Exception:
            pass

    conn.commit()
    conn.close()


def insert_post(content: str, token_hash: str, title: str = "") -> int:
    """
    Insert a new post. title optional. Returns post_id.
    """
    conn = _connect()
    cur = conn.cursor()
    ts = int(time.time())
    cur.execute(
        "INSERT INTO posts (content, created_at, token_hash, title, likes, dislikes, reports) VALUES (?, ?, ?, ?, 0, 0, 0)",
        (content, ts, token_hash, title),
    )
    post_id = cur.lastrowid
    conn.commit()
    conn.close()
    return post_id


def get_post(post_id: int) -> Optional[Tuple[int, str, int, str, int, int, int]]:
    """
    Return (id, content, created_at, title, likes, dislikes, reports)
    """
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT id, content, created_at, title, likes, dislikes, reports FROM posts WHERE id = ?", (post_id,))
    row = cur.fetchone()
    conn.close()
    return row


def list_recent(limit: int = 30) -> List[Tuple[int, str, int, str, int, int, int]]:
    """
    Return list of recent posts as (id, content, created_at, title, likes, dislikes, reports)
    """
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, content, created_at, title, likes, dislikes, reports FROM posts ORDER BY created_at DESC LIMIT ?",
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def get_token_hash(post_id: int) -> Optional[str]:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT token_hash FROM posts WHERE id = ?", (post_id,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def delete_post_by_hash(post_id: int, provided_token_hash: str) -> bool:
    """
    Backwards-compatible delete: accepts already-hashed token string.
    """
    stored = get_token_hash(post_id)
    if not stored:
        return False
    # use constant-time compare
    if _hmac.compare_digest(stored, provided_token_hash):
        conn = _connect()
        cur = conn.cursor()
        cur.execute("DELETE FROM posts WHERE id = ?", (post_id,))
        conn.commit()
        conn.close()
        # cleanup_tags_for_post will be a no-op if foreign keys cascade is active,
        # but we keep it to be safe on DBs without pragma enabled elsewhere.
        cleanup_tags_for_post(post_id)
        return True
    return False


def delete_post_with_token(post_id: int, provided_token: str) -> bool:
    """
    New recommended API: pass raw token (user input). We hash it and compare safely.
    Trims whitespace to be forgiving about copy/paste.
    """
    if not provided_token:
        return False
    provided_token = provided_token.strip()
    provided_hash = utils.hmac_hash(provided_token)
    return delete_post_by_hash(post_id, provided_hash)


def get_all_posts() -> List[Tuple[int, str, int]]:
    """Return all posts (id, content, created_at) - used for rebuilding index."""
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, content, created_at FROM posts ORDER BY created_at ASC"
    )
    rows = cur.fetchall()
    conn.close()
    return rows


# ---------- Tag helpers ----------


def set_tags_for_post(post_id: int, tags: List[str]):
    """
    Replace tags for a post (idempotent).
    """
    conn = _connect()
    cur = conn.cursor()
    # delete existing
    cur.execute("DELETE FROM post_tags WHERE post_id = ?", (post_id,))
    # insert new
    for t in tags:
        cur.execute(
            "INSERT OR IGNORE INTO post_tags (post_id, tag) VALUES (?, ?)",
            (post_id, t),
        )
    conn.commit()
    conn.close()


def get_tags_for_post(post_id: int) -> List[str]:
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT tag FROM post_tags WHERE post_id = ? ORDER BY tag ASC", (post_id,)
    )
    rows = cur.fetchall()
    conn.close()
    return [r[0] for r in rows]


def cleanup_tags_for_post(post_id: int):
    conn = _connect()
    cur = conn.cursor()
    cur.execute("DELETE FROM post_tags WHERE post_id = ?", (post_id,))
    conn.commit()
    conn.close()


# ---------- Like/Dislike/Report helpers ----------
def increment_like(post_id: int):
    conn = _connect()
    cur = conn.cursor()
    cur.execute("UPDATE posts SET likes = COALESCE(likes,0) + 1 WHERE id = ?", (post_id,))
    conn.commit()
    conn.close()

def increment_dislike(post_id: int):
    conn = _connect()
    cur = conn.cursor()
    cur.execute("UPDATE posts SET dislikes = COALESCE(dislikes,0) + 1 WHERE id = ?", (post_id,))
    conn.commit()
    conn.close()

def report_post(post_id: int):
    conn = _connect()
    cur = conn.cursor()
    cur.execute("UPDATE posts SET reports = COALESCE(reports,0) + 1 WHERE id = ?", (post_id,))
    conn.commit()
    conn.close()


# ---------- Simple DB inspector / CRUD helper ----------
def _human_size(path: str) -> str:
    s = __import__("os").path.getsize(path)
    for unit in ["B", "KB", "MB", "GB"]:
        if s < 1024:
            return f"{s:.1f}{unit}"
        s /= 1024
    return f"{s:.1f}TB"


if __name__ == "__main__":
    # quick DB inspector for debugging
    import os

    print("DB absolute path:", os.path.abspath(DB_PATH))
    try:
        print("DB size:", _human_size(DB_PATH))
    except OSError:
        print("DB file not found yet.")
    print()

    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, content, created_at, token_hash FROM posts ORDER BY created_at ASC"
    )
    rows = cur.fetchall()
    print(f"Total rows: {len(rows)}")
    print("-" * 60)
    for pid, content, ts, token_hash in rows:
        print(f"ID: {pid}")
        print("CONTENT:", content)
        print("CREATED_AT:", ts)
        print("TOKEN_HASH:", token_hash)
        print("-" * 60)
    conn.close()
    post_id = int(input("Enter Post id to delete: ").strip())
    stored_hash = input("Enter stored hash (pass hashed token): ").strip()
    ok = delete_post_by_hash(post_id, stored_hash)
    print("Deleted:", ok)
