"""SQLite storage for logged violations."""
from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS violations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detected_at TEXT NOT NULL,
    source TEXT,
    video_time_s REAL,
    track_id INTEGER,
    plate TEXT,
    plate_valid INTEGER,
    helmet_status TEXT,
    confidence REAL,
    snapshot_path TEXT,
    plate_path TEXT,
    video_path TEXT
)
"""

COLUMNS = ["detected_at", "source", "video_time_s", "track_id", "plate", "plate_valid",
           "helmet_status", "confidence", "snapshot_path", "plate_path", "video_path"]


def connect(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(SCHEMA)
    # migrate databases created before video_path existed
    existing = {r[1] for r in conn.execute("PRAGMA table_info(violations)")}
    if "video_path" not in existing:
        conn.execute("ALTER TABLE violations ADD COLUMN video_path TEXT")
    conn.commit()
    return conn


def insert(conn, row: dict):
    values = [row.get(c) for c in COLUMNS]
    # numpy numbers would be stored as BLOBs; turn them into plain Python numbers
    values = [v.item() if hasattr(v, "item") else v for v in values]
    conn.execute(f"INSERT INTO violations ({','.join(COLUMNS)}) VALUES ({','.join('?' * len(COLUMNS))})", values)
    conn.commit()


def export_csv(db_path, out_path):
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT id," + ",".join(COLUMNS) + " FROM violations ORDER BY id").fetchall()
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id"] + COLUMNS)
        w.writerows(rows)
    conn.close()