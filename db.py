#!/usr/bin/python3

import sqlite3
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# register adapters for pandas and numpy types
sqlite3.register_adapter(pd.Timestamp, lambda x: x.isoformat())
sqlite3.register_adapter(np.int64, int)
sqlite3.register_adapter(np.float64, float)

DB_PATH = Path("sleep.db")

#Create connection
def get_conn():
    # create database connection with thread safety disabled
    conn = sqlite3.connect(
        DB_PATH,
        detect_types=sqlite3.PARSE_DECLTYPES,
        check_same_thread=False
    )
    # enable row access by name
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()

    # optimize journal and synchronization settings
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")

    # create classifications table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS classifications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        night_id TEXT,
        ts DATETIME,
        label TEXT,
        probabilities TEXT
    );
    """)

    # create sleep events table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sleep_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        night_id TEXT,
        ts DATETIME,
        type TEXT,
        payload TEXT
    );
    """)

    # create alarms table
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

    # create settings table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT
    );
    """)

    # save changes and close connection
    conn.commit()
    conn.close()

#Write functions
def insert_classification(night_id, ts, label, probabilities):
    conn = get_conn()
    
    # insert new classification record
    conn.execute(
        """
        INSERT INTO classifications (night_id, ts, label, probabilities)
        VALUES (?, ?, ?, ?)
        """,
        (night_id, ts, label, json.dumps(probabilities))
    )
    
    # save changes and close connection
    conn.commit()
    conn.close()

def insert_event(night_id, event_type, payload=None, ts=None):
    # set timestamp if missing
    if ts is None:
        ts = datetime.now()

    conn = get_conn()
    
    # insert new sleep event
    conn.execute(
        """
        INSERT INTO sleep_events (night_id, ts, type, payload)
        VALUES (?, ?, ?, ?)
        """,
        (night_id, ts, event_type, json.dumps(payload) if payload else None)
    )
    
    # save changes and close connection
    conn.commit()
    conn.close()

def insert_alarm(target_ts, source, reason, meta):
    conn = get_conn()
    cur = conn.cursor()

    # deactivate any currently active alarms
    cur.execute("UPDATE alarms SET active=0 WHERE active=1")

    # insert new active alarm
    cur.execute(
        """
        INSERT INTO alarms (ts, target_ts, source, active, fired, reason, meta)
        VALUES (?, ?, ?, 1, 0, ?, ?)
        """,
        (datetime.now(), target_ts, source, reason, json.dumps(meta))
    )

    # get id of new alarm
    alarm_id = cur.lastrowid
    
    # save changes and close connection
    conn.commit()
    conn.close()
    return alarm_id

#Read functions
def get_active_alarm():
    conn = get_conn()
    
    # fetch the single active alarm
    row = conn.execute(
        "SELECT * FROM alarms WHERE active=1 LIMIT 1"
    ).fetchone()
    
    conn.close()
    return row

def get_setting(key, default=None):
    conn = get_conn()
    
    # fetch value for specific setting key
    row = conn.execute(
        "SELECT value FROM settings WHERE key=?",
        (key,)
    ).fetchone()
    
    conn.close()
    return row["value"] if row else default

def set_setting(key, value):
    conn = get_conn()
    
    # insert or update setting key-value pair
    conn.execute(
        """
        INSERT INTO settings (key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (key, value)
    )
    
    # save changes and close connection
    conn.commit()
    conn.close()

def get_sleep_history(days=7):
    conn = get_conn()
    cutoff_date = datetime.now() - pd.Timedelta(days=days)
    
    query = """
    SELECT ts, type 
    FROM sleep_events 
    WHERE type IN ('sleep_onset', 'wake_up', 'alarm_target') 
    AND ts > ? 
    ORDER BY ts ASC
    """
    
    rows = conn.execute(query, (cutoff_date,)).fetchall()
    conn.close()
    
    return [{'ts': pd.to_datetime(r['ts']), 'type': r['type']} for r in rows]

def update_alarm_status(target_ts, active=1, source="auto"):
    #Updates the alarms table.

    conn = get_conn()
    conn.execute("UPDATE alarms SET active=0") # Disable old alarms
    
    if active:
        conn.execute(
            "INSERT INTO alarms (ts, target_ts, source, active, fired, reason, meta) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (datetime.now(), target_ts, source, 1, 0, "scheduled", "{}")
        )
    conn.commit()
    conn.close()

def get_recent_classifications(night_id, hours=4):
    """
    Fetches classification rows for the last N hours for smart alarm logic.
    """
    conn = get_conn()
    
    # Calculate lookback time
    lookback = datetime.now() - pd.Timedelta(hours=hours)
    
    query = """
    SELECT ts, probabilities 
    FROM classifications 
    WHERE night_id = ? 
    AND ts > ? 
    ORDER BY ts ASC
    """
    
    rows = conn.execute(query, (night_id, lookback)).fetchall()
    conn.close()
    
    return rows