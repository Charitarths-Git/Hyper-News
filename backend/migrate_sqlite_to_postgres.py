"""
One-time helper to migrate legacy SQLite profile state into PostgreSQL.

This migrates the durable user/profile layers and recent-state windows. Event
tables can continue to accumulate from the new stack after cutover.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3

from db import DATABASE_DIALECT, init_db, save_reading_vector, save_recent_state, save_user


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate HyperNews SQLite profile state into PostgreSQL.")
    parser.add_argument(
        "--sqlite-path",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "hypernews.db"),
        help="Path to the legacy SQLite file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if DATABASE_DIALECT != "postgresql":
        raise SystemExit("Set DATABASE_URL to a PostgreSQL connection before running this migration.")
    if not os.path.exists(args.sqlite_path):
        raise SystemExit(f"SQLite file not found: {args.sqlite_path}")

    init_db()

    counts = {
        "users": 0,
        "reading_history_vectors": 0,
        "user_recent_state": 0,
    }

    source = sqlite3.connect(args.sqlite_path)
    source.row_factory = sqlite3.Row
    try:
        for row in source.execute("SELECT user_id, interests, reading_history, avg_dwell_time, total_reads FROM users"):
            save_user(
                row["user_id"],
                json.loads(row["interests"] or "{}"),
                json.loads(row["reading_history"] or "[]"),
                int(row["total_reads"] or 0),
                avg_dwell_time=float(row["avg_dwell_time"] or 0.0),
            )
            counts["users"] += 1

        for row in source.execute("SELECT user_id, article_id, feedback_weight, embedding_json FROM reading_history_vectors"):
            embedding = json.loads(row["embedding_json"] or "[]")
            if not embedding:
                continue
            save_reading_vector(
                row["user_id"],
                row["article_id"],
                embedding=embedding,
                feedback_weight=float(row["feedback_weight"] or 1.0),
            )
            counts["reading_history_vectors"] += 1

        for row in source.execute("SELECT * FROM user_recent_state"):
            save_recent_state(
                row["user_id"],
                recent_clicks=json.loads(row["recent_clicks_json"] or "[]"),
                recent_skips=json.loads(row["recent_skips_json"] or "[]"),
                recent_negative_actions=json.loads(row["recent_negative_actions_json"] or "[]"),
                recent_queries=json.loads(row["recent_queries_json"] or "[]"),
                recent_entities=json.loads(row["recent_entities_json"] or "[]"),
                recent_sources=json.loads(row["recent_sources_json"] or "[]"),
            )
            counts["user_recent_state"] += 1
    finally:
        source.close()

    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
