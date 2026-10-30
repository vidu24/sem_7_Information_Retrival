import sqlite3
import json
from datetime import datetime, timezone
from .config import DB_PATH

def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with connect() as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS articles(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            text TEXT NOT NULL,
            source TEXT DEFAULT '',
            url TEXT DEFAULT '',
            label TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            metadata TEXT DEFAULT '{}'
        )""")
        con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_article_url ON articles(url) WHERE url <> ''")

def add_article(title, text, source="", url="", label="", metadata=None):
    init_db()
    with connect() as con:
        if url:
            row = con.execute("SELECT id FROM articles WHERE url=?", (url,)).fetchone()
            if row:
                return row["id"], False
        cur = con.execute(
            "INSERT INTO articles(title,text,source,url,label,created_at,metadata) VALUES(?,?,?,?,?,?,?)",
            (title.strip() or "Untitled article", text.strip(), source, url, label,
             datetime.now(timezone.utc).isoformat(), json.dumps(metadata or {}))
        )
        return cur.lastrowid, True

def get_articles():
    init_db()
    with connect() as con:
        return [dict(r) for r in con.execute("SELECT * FROM articles ORDER BY id DESC").fetchall()]

def count_articles():
    init_db()
    with connect() as con:
        return con.execute("SELECT COUNT(*) FROM articles").fetchone()[0]

def clear_articles():
    with connect() as con:
        con.execute("DELETE FROM articles")
