"""Initialize the SQLite prediction/feedback registry without fake labels."""
from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path


def database_path() -> Path:
    return Path(os.environ.get("FEEDBACK_DB", "data/feedbacks.db"))


def initialize(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS predictions (
                request_id TEXT PRIMARY KEY,
                input_json TEXT NOT NULL,
                prediction INTEGER NOT NULL CHECK (prediction BETWEEN 0 AND 2),
                probabilities_json TEXT NOT NULL,
                model_version TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS feedbacks (
                request_id TEXT PRIMARY KEY,
                prediction INTEGER NOT NULL CHECK (prediction BETWEEN 0 AND 2),
                true_label INTEGER NOT NULL CHECK (true_label BETWEEN 0 AND 2),
                comments TEXT,
                created_at TEXT NOT NULL,
                used_for_training INTEGER NOT NULL DEFAULT 0 CHECK (used_for_training IN (0, 1)),
                FOREIGN KEY(request_id) REFERENCES predictions(request_id)
            )"""
        )
    print(f"Feedback database initialized: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=None)
    args = parser.parse_args()
    initialize(args.path or database_path())
