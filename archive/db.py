import sqlite3
import json
from pathlib import Path
from datetime import datetime

DB_PATH = Path("sleep.db")

# ------------------ Connection ------------------

def get_conn():
    conn = sqlite3.connect(
        DB_PATH,
        detect_types=sqlite3.PARSE_DECLTYPES,
        check_same_thread=False
    )
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS classifications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        night_id TEXT,
        ts DATETIME,
        label TEXT,
        probabilities TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS sleep_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        night_id TEXT,
        ts DATETIME,
        type TEXT,
        payload TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS alarms (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts DATETIME,
        target_ts DATETIME,
        source TEXT,
        active INTEGER,
        fired INTEGER,
        reason TEXT,
        meta TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT
    );
    """)

    conn.commit()
    conn.close()

# ------------------ Writes ------------------

def insert_classification(night_id, ts, label, probabilities):
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO classifications (night_id, ts, label, probabilities)
        VALUES (?, ?, ?, ?)
        """,
        (night_id, ts, label, json.dumps(probabilities))
    )
    conn.commit()
    conn.close()


def insert_event(night_id, event_type, payload=None, ts=None):
    if ts is None:
        ts = datetime.now()

    conn = get_conn()
    conn.execute(
        """
        INSERT INTO sleep_events (night_id, ts, type, payload)
        VALUES (?, ?, ?, ?)
        """,
        (night_id, ts, event_type, json.dumps(payload) if payload else None)
    )
    conn.commit()
    conn.close()


def insert_alarm(target_ts, source, reason, meta):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("UPDATE alarms SET active=0 WHERE active=1")

    cur.execute(
        """
        INSERT INTO alarms (ts, target_ts, source, active, fired, reason, meta)
        VALUES (?, ?, ?, 1, 0, ?, ?)
        """,
        (datetime.now(), target_ts, source, reason, json.dumps(meta))
    )

    alarm_id = cur.lastrowid
    conn.commit()
    conn.close()
    return alarm_id

# ------------------ Reads ------------------

def get_active_alarm():
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM alarms WHERE active=1 LIMIT 1"
    ).fetchone()
    conn.close()
    return row


def get_setting(key, default=None):
    conn = get_conn()
    row = conn.execute(
        "SELECT value FROM settings WHERE key=?",
        (key,)
    ).fetchone()
    conn.close()
    return row["value"] if row else default


def set_setting(key, value):
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO settings (key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (key, value)
    )
    conn.commit()
    conn.close()
