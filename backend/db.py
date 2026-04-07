"""
Database persistence layer with PostgreSQL wiring and SQLite fallback.

This module intentionally avoids SQLAlchemy because that import path stalls on
this machine. The active backend is selected from `DATABASE_URL`:

- `postgresql://...` -> psycopg2-backed PostgreSQL
- `sqlite:///...`    -> sqlite3 fallback
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import parse_qsl, unquote, urlparse

import numpy as np


_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_SQLITE_PATH = os.path.join(_PROJECT_ROOT, "data", "hypernews.db")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/hypernews.db").strip()


def _normalize_database_url(database_url: str) -> str:
    raw = str(database_url or "").strip()
    if not raw:
        return f"sqlite:///{_DEFAULT_SQLITE_PATH}"
    if raw.startswith("postgres://"):
        return "postgresql://" + raw[len("postgres://"):]
    if raw.startswith("sqlite:///"):
        suffix = raw[len("sqlite:///"):].strip()
        if not suffix:
            return f"sqlite:///{_DEFAULT_SQLITE_PATH}"
        if os.path.isabs(suffix):
            return f"sqlite:///{suffix}"
        normalized = suffix[2:] if suffix.startswith(("./", ".\\")) else suffix
        return f"sqlite:///{os.path.normpath(os.path.join(_PROJECT_ROOT, normalized))}"
    return raw


DATABASE_URL = _normalize_database_url(DATABASE_URL)
DATABASE_DIALECT = "postgresql" if DATABASE_URL.startswith("postgresql://") else "sqlite"
_POSTGRES_DRIVER = None
_POSTGRES_QUERY_OPTIONS = {
    "application_name",
    "channel_binding",
    "connect_timeout",
    "gssencmode",
    "keepalives",
    "keepalives_count",
    "keepalives_idle",
    "keepalives_interval",
    "options",
    "sslcert",
    "sslcrl",
    "sslkey",
    "sslmode",
    "sslpassword",
    "sslrootcert",
    "target_session_attrs",
}


def _sqlite_path() -> str:
    return DATABASE_URL[len("sqlite:///"):]


def _ensure_postgres_driver():
    global _POSTGRES_DRIVER
    if _POSTGRES_DRIVER is not None:
        return _POSTGRES_DRIVER
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
    except Exception as exc:
        raise RuntimeError(
            "psycopg2 is required for PostgreSQL DATABASE_URL values. "
            "Install dependencies from requirements.txt."
        ) from exc
    _POSTGRES_DRIVER = (psycopg2, RealDictCursor)
    return _POSTGRES_DRIVER


@contextmanager
def _connect():
    if DATABASE_DIALECT == "sqlite":
        path = _sqlite_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        connection = sqlite3.connect(path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()
        return

    psycopg2, RealDictCursor = _ensure_postgres_driver()
    parsed = urlparse(DATABASE_URL)
    connect_kwargs = dict(
        dbname=parsed.path.lstrip("/"),
        user=unquote(parsed.username or ""),
        password=unquote(parsed.password or ""),
        host=parsed.hostname or "localhost",
        port=parsed.port or 5432,
        cursor_factory=RealDictCursor,
    )
    for key, value in parse_qsl(parsed.query, keep_blank_values=False):
        normalized_key = str(key or "").strip()
        if normalized_key in _POSTGRES_QUERY_OPTIONS:
            connect_kwargs[normalized_key] = value
    connection = psycopg2.connect(**connect_kwargs)
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def _fetchall(cursor):
    rows = cursor.fetchall()
    return [dict(row) for row in rows]


def _fetchone(cursor):
    row = cursor.fetchone()
    if row is None:
        return None
    return dict(row)


def _placeholder(name: str) -> str:
    if DATABASE_DIALECT == "sqlite":
        return f":{name}"
    return f"%({name})s"


def _json_loads(value, fallback):
    try:
        return json.loads(value or json.dumps(fallback))
    except Exception:
        return fallback


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _table_columns(conn, table_name: str) -> set[str]:
    cursor = conn.cursor()
    try:
        if DATABASE_DIALECT == "sqlite":
            cursor.execute(f"PRAGMA table_info({table_name})")
            rows = cursor.fetchall()
            return {str(row["name"]) for row in rows}

        cursor.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = %(table_name)s
            """,
            {"table_name": str(table_name)},
        )
        rows = cursor.fetchall()
        return {str(row["column_name"]) for row in rows}
    finally:
        cursor.close()


def _ensure_columns(conn, table_name: str, columns: dict[str, str]) -> None:
    existing = _table_columns(conn, table_name)
    if not columns:
        return

    cursor = conn.cursor()
    try:
        for column_name, definition in columns.items():
            if column_name in existing:
                continue
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")
            existing.add(column_name)
    finally:
        cursor.close()


def get_database_backend_info() -> dict:
    return {
        "database_url": DATABASE_URL,
        "dialect": DATABASE_DIALECT,
    }


def init_db():
    user_pk = "TEXT PRIMARY KEY" if DATABASE_DIALECT == "sqlite" else "VARCHAR(255) PRIMARY KEY"
    id_pk = "INTEGER PRIMARY KEY AUTOINCREMENT" if DATABASE_DIALECT == "sqlite" else "BIGSERIAL PRIMARY KEY"
    timestamp_default = "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"

    statements = [
        f"""
        CREATE TABLE IF NOT EXISTS users (
            user_id {user_pk},
            interests TEXT DEFAULT '{{}}',
            reading_history TEXT DEFAULT '[]',
            avg_dwell_time REAL DEFAULT 0.0,
            total_reads INTEGER DEFAULT 0,
            created_at {timestamp_default},
            updated_at {timestamp_default}
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS reading_history_vectors (
            id {id_pk},
            user_id TEXT NOT NULL,
            article_id TEXT NOT NULL,
            read_at {timestamp_default},
            feedback_weight REAL DEFAULT 1.0,
            embedding_json TEXT,
            UNIQUE (user_id, article_id)
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS feedback_events (
            id {id_pk},
            user_id TEXT NOT NULL,
            session_id TEXT DEFAULT '',
            request_id TEXT DEFAULT '',
            impression_id TEXT DEFAULT '',
            article_id TEXT NOT NULL,
            action TEXT NOT NULL,
            position INTEGER DEFAULT -1,
            dwell_time REAL DEFAULT 0.0,
            query_text TEXT DEFAULT '',
            source_feedback TEXT DEFAULT '',
            created_at {timestamp_default}
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS recommendation_events (
            id {id_pk},
            user_id TEXT NOT NULL,
            session_id TEXT DEFAULT '',
            request_id TEXT DEFAULT '',
            surface TEXT DEFAULT 'feed',
            mood TEXT DEFAULT 'neutral',
            mode TEXT DEFAULT '',
            query_text TEXT DEFAULT '',
            candidate_sources_json TEXT DEFAULT '{{}}',
            impression_ids_json TEXT DEFAULT '[]',
            created_at {timestamp_default}
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS search_events (
            id {id_pk},
            user_id TEXT NOT NULL,
            session_id TEXT DEFAULT '',
            query_text TEXT NOT NULL,
            normalized_query TEXT DEFAULT '',
            intent_json TEXT DEFAULT '{{}}',
            created_at {timestamp_default}
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS user_recent_state (
            user_id {user_pk},
            recent_clicks_json TEXT DEFAULT '[]',
            recent_skips_json TEXT DEFAULT '[]',
            recent_negative_actions_json TEXT DEFAULT '[]',
            recent_queries_json TEXT DEFAULT '[]',
            recent_entities_json TEXT DEFAULT '[]',
            recent_sources_json TEXT DEFAULT '[]',
            updated_at {timestamp_default}
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS user_profile_snapshots (
            id {id_pk},
            user_id TEXT NOT NULL,
            window_name TEXT NOT NULL,
            profile_json TEXT DEFAULT '{{}}',
            created_at {timestamp_default}
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS auth_users (
            user_id TEXT PRIMARY KEY,
            email TEXT NOT NULL UNIQUE,
            display_name TEXT DEFAULT '',
            age_bucket TEXT DEFAULT '',
            gender TEXT DEFAULT '',
            occupation TEXT DEFAULT '',
            location_region TEXT DEFAULT '',
            location_country TEXT DEFAULT '',
            interest_text TEXT DEFAULT '',
            top_categories_json TEXT DEFAULT '[]',
            affect_consent INTEGER DEFAULT 0,
            bio_embedding_json TEXT DEFAULT '[]',
            bio_text_embedding_json TEXT DEFAULT '[]',
            bio_embedding_version TEXT DEFAULT '',
            onboarding_completed INTEGER DEFAULT 0,
            onboarding_completed_at TIMESTAMP NULL,
            password_hash TEXT NOT NULL,
            created_at {timestamp_default},
            updated_at {timestamp_default}
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS ranking_feature_events (
            id {id_pk},
            user_id TEXT NOT NULL,
            session_id TEXT DEFAULT '',
            request_id TEXT NOT NULL,
            impression_id TEXT DEFAULT '',
            article_id TEXT NOT NULL,
            surface TEXT DEFAULT 'feed',
            query_text TEXT DEFAULT '',
            candidate_source TEXT DEFAULT '',
            position INTEGER DEFAULT -1,
            label REAL,
            dwell_time REAL DEFAULT 0.0,
            features_json TEXT DEFAULT '{{}}',
            created_at {timestamp_default}
        )
        """,
    ]

    with _connect() as conn:
        cursor = conn.cursor()
        if DATABASE_DIALECT == "postgresql":
            try:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            except Exception as exc:
                print(f"pgvector extension unavailable ({exc}); continuing without DB-side vectors.")
        for statement in statements:
            cursor.execute(statement)
        cursor.close()
        _ensure_columns(
            conn,
            "auth_users",
            {
                "age_bucket": "TEXT DEFAULT ''",
                "gender": "TEXT DEFAULT ''",
                "occupation": "TEXT DEFAULT ''",
                "location_region": "TEXT DEFAULT ''",
                "location_country": "TEXT DEFAULT ''",
                "interest_text": "TEXT DEFAULT ''",
                "top_categories_json": "TEXT DEFAULT '[]'",
                "affect_consent": "INTEGER DEFAULT 0",
                "bio_embedding_json": "TEXT DEFAULT '[]'",
                "bio_text_embedding_json": "TEXT DEFAULT '[]'",
                "bio_embedding_version": "TEXT DEFAULT ''",
                "onboarding_completed": "INTEGER DEFAULT 0",
                "onboarding_completed_at": "TIMESTAMP NULL",
            },
        )


def save_user(
    user_id: str,
    interests: dict,
    reading_history: list,
    total_reads: int,
    avg_dwell_time: float = 0.0,
):
    user_param = _placeholder("user_id")
    interests_param = _placeholder("interests")
    history_param = _placeholder("reading_history")
    dwell_param = _placeholder("avg_dwell_time")
    reads_param = _placeholder("total_reads")
    sql = f"""
    INSERT INTO users (user_id, interests, reading_history, avg_dwell_time, total_reads)
    VALUES ({user_param}, {interests_param}, {history_param}, {dwell_param}, {reads_param})
    ON CONFLICT(user_id) DO UPDATE SET
        interests = EXCLUDED.interests,
        reading_history = EXCLUDED.reading_history,
        avg_dwell_time = EXCLUDED.avg_dwell_time,
        total_reads = EXCLUDED.total_reads,
        updated_at = CURRENT_TIMESTAMP
    """
    params = {
        "user_id": str(user_id),
        "interests": json.dumps(interests or {}),
        "reading_history": json.dumps(reading_history or []),
        "avg_dwell_time": float(avg_dwell_time or 0.0),
        "total_reads": int(total_reads),
    }
    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        cursor.close()


def load_user(user_id: str) -> Optional[dict]:
    sql = f"""
    SELECT user_id, interests, reading_history, avg_dwell_time, total_reads
    FROM users
    WHERE user_id = {_placeholder("user_id")}
    """
    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, {"user_id": str(user_id)})
        row = _fetchone(cursor)
        cursor.close()
    if row is None:
        return None
    return {
        "user_id": str(row["user_id"]),
        "interests": _json_loads(row["interests"], {}),
        "reading_history": _json_loads(row["reading_history"], []),
        "avg_dwell_time": float(row["avg_dwell_time"] or 0.0),
        "total_reads": int(row["total_reads"] or 0),
    }


def delete_user(user_id: str):
    user_key = str(user_id)
    with _connect() as conn:
        cursor = conn.cursor()
        for table_name in (
            "users",
            "reading_history_vectors",
            "feedback_events",
            "recommendation_events",
            "search_events",
            "user_recent_state",
            "user_profile_snapshots",
            "ranking_feature_events",
        ):
            cursor.execute(
                f"DELETE FROM {table_name} WHERE user_id = {_placeholder('user_id')}",
                {"user_id": user_key},
            )
        cursor.close()


def create_auth_user(
    user_id: str,
    email: str,
    display_name: str,
    password_hash: str,
    age_bucket: str = "",
    gender: str = "",
    occupation: str = "",
    location_region: str = "",
    location_country: str = "",
    interest_text: str = "",
    top_categories: list[str] | None = None,
    affect_consent: bool = False,
    bio_embedding: list[float] | None = None,
    bio_text_embedding: list[float] | None = None,
    bio_embedding_version: str = "",
    onboarding_completed: bool = False,
    onboarding_completed_at: str | None = None,
) -> dict:
    sql = f"""
    INSERT INTO auth_users (
        user_id,
        email,
        display_name,
        age_bucket,
        gender,
        occupation,
        location_region,
        location_country,
        interest_text,
        top_categories_json,
        affect_consent,
        bio_embedding_json,
        bio_text_embedding_json,
        bio_embedding_version,
        onboarding_completed,
        onboarding_completed_at,
        password_hash
    )
    VALUES (
        {_placeholder("user_id")},
        {_placeholder("email")},
        {_placeholder("display_name")},
        {_placeholder("age_bucket")},
        {_placeholder("gender")},
        {_placeholder("occupation")},
        {_placeholder("location_region")},
        {_placeholder("location_country")},
        {_placeholder("interest_text")},
        {_placeholder("top_categories_json")},
        {_placeholder("affect_consent")},
        {_placeholder("bio_embedding_json")},
        {_placeholder("bio_text_embedding_json")},
        {_placeholder("bio_embedding_version")},
        {_placeholder("onboarding_completed")},
        {_placeholder("onboarding_completed_at")},
        {_placeholder("password_hash")}
    )
    """
    completion_timestamp = (
        str(onboarding_completed_at or "").strip()
        or (_utc_timestamp() if onboarding_completed else None)
    )
    params = {
        "user_id": str(user_id),
        "email": str(email).strip().lower(),
        "display_name": str(display_name or "").strip(),
        "age_bucket": str(age_bucket or "").strip(),
        "gender": str(gender or "").strip(),
        "occupation": str(occupation or "").strip(),
        "location_region": str(location_region or "").strip(),
        "location_country": str(location_country or "").strip(),
        "interest_text": str(interest_text or "").strip(),
        "top_categories_json": json.dumps(list(top_categories or [])),
        "affect_consent": int(bool(affect_consent)),
        "bio_embedding_json": json.dumps(list(bio_embedding or [])),
        "bio_text_embedding_json": json.dumps(list(bio_text_embedding or [])),
        "bio_embedding_version": str(bio_embedding_version or "").strip(),
        "onboarding_completed": int(bool(onboarding_completed)),
        "onboarding_completed_at": completion_timestamp,
        "password_hash": str(password_hash),
    }
    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        cursor.close()
    return {
        "user_id": params["user_id"],
        "email": params["email"],
        "display_name": params["display_name"],
        "age_bucket": params["age_bucket"],
        "gender": params["gender"],
        "occupation": params["occupation"],
        "location_region": params["location_region"],
        "location_country": params["location_country"],
        "interest_text": params["interest_text"],
        "top_categories": list(top_categories or []),
        "affect_consent": bool(affect_consent),
        "bio_embedding_version": params["bio_embedding_version"],
        "onboarding_completed": bool(onboarding_completed),
        "onboarding_completed_at": completion_timestamp or "",
    }


def load_auth_user_by_email(email: str) -> Optional[dict]:
    sql = f"""
    SELECT
        user_id,
        email,
        display_name,
        age_bucket,
        gender,
        occupation,
        location_region,
        location_country,
        interest_text,
        top_categories_json,
        affect_consent,
        bio_embedding_json,
        bio_text_embedding_json,
        bio_embedding_version,
        onboarding_completed,
        onboarding_completed_at,
        password_hash,
        created_at,
        updated_at
    FROM auth_users
    WHERE email = {_placeholder("email")}
    """
    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, {"email": str(email).strip().lower()})
        row = _fetchone(cursor)
        cursor.close()
    return row


def load_auth_user_by_id(user_id: str) -> Optional[dict]:
    sql = f"""
    SELECT
        user_id,
        email,
        display_name,
        age_bucket,
        gender,
        occupation,
        location_region,
        location_country,
        interest_text,
        top_categories_json,
        affect_consent,
        bio_embedding_json,
        bio_text_embedding_json,
        bio_embedding_version,
        onboarding_completed,
        onboarding_completed_at,
        password_hash,
        created_at,
        updated_at
    FROM auth_users
    WHERE user_id = {_placeholder("user_id")}
    """
    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, {"user_id": str(user_id)})
        row = _fetchone(cursor)
        cursor.close()
    return row


def update_auth_user_profile(
    user_id: str,
    *,
    display_name: str,
    age_bucket: str = "",
    gender: str = "",
    occupation: str = "",
    location_region: str = "",
    location_country: str = "",
    interest_text: str = "",
    top_categories: list[str] | None = None,
    affect_consent: bool = False,
    bio_embedding: list[float] | None = None,
    bio_text_embedding: list[float] | None = None,
    bio_embedding_version: str = "",
) -> Optional[dict]:
    sql = f"""
    UPDATE auth_users
    SET
        display_name = {_placeholder("display_name")},
        age_bucket = {_placeholder("age_bucket")},
        gender = {_placeholder("gender")},
        occupation = {_placeholder("occupation")},
        location_region = {_placeholder("location_region")},
        location_country = {_placeholder("location_country")},
        interest_text = {_placeholder("interest_text")},
        top_categories_json = {_placeholder("top_categories_json")},
        affect_consent = {_placeholder("affect_consent")},
        bio_embedding_json = {_placeholder("bio_embedding_json")},
        bio_text_embedding_json = {_placeholder("bio_text_embedding_json")},
        bio_embedding_version = {_placeholder("bio_embedding_version")},
        updated_at = CURRENT_TIMESTAMP
    WHERE user_id = {_placeholder("user_id")}
    """
    params = {
        "user_id": str(user_id),
        "display_name": str(display_name or "").strip(),
        "age_bucket": str(age_bucket or "").strip(),
        "gender": str(gender or "").strip(),
        "occupation": str(occupation or "").strip(),
        "location_region": str(location_region or "").strip(),
        "location_country": str(location_country or "").strip(),
        "interest_text": str(interest_text or "").strip(),
        "top_categories_json": json.dumps(list(top_categories or [])),
        "affect_consent": int(bool(affect_consent)),
        "bio_embedding_json": json.dumps(list(bio_embedding or [])),
        "bio_text_embedding_json": json.dumps(list(bio_text_embedding or [])),
        "bio_embedding_version": str(bio_embedding_version or "").strip(),
    }
    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        cursor.close()
    return load_auth_user_by_id(user_id)


def complete_auth_user_onboarding(
    user_id: str,
    *,
    display_name: str,
    age_bucket: str = "",
    gender: str = "",
    occupation: str = "",
    location_region: str = "",
    location_country: str = "",
    interest_text: str = "",
    top_categories: list[str] | None = None,
    affect_consent: bool = False,
    bio_embedding: list[float] | None = None,
    bio_text_embedding: list[float] | None = None,
    bio_embedding_version: str = "",
) -> Optional[dict]:
    sql = f"""
    UPDATE auth_users
    SET
        display_name = {_placeholder("display_name")},
        age_bucket = {_placeholder("age_bucket")},
        gender = {_placeholder("gender")},
        occupation = {_placeholder("occupation")},
        location_region = {_placeholder("location_region")},
        location_country = {_placeholder("location_country")},
        interest_text = {_placeholder("interest_text")},
        top_categories_json = {_placeholder("top_categories_json")},
        affect_consent = {_placeholder("affect_consent")},
        bio_embedding_json = {_placeholder("bio_embedding_json")},
        bio_text_embedding_json = {_placeholder("bio_text_embedding_json")},
        bio_embedding_version = {_placeholder("bio_embedding_version")},
        onboarding_completed = 1,
        onboarding_completed_at = {_placeholder("onboarding_completed_at")},
        updated_at = CURRENT_TIMESTAMP
    WHERE user_id = {_placeholder("user_id")}
    """
    params = {
        "user_id": str(user_id),
        "display_name": str(display_name or "").strip(),
        "age_bucket": str(age_bucket or "").strip(),
        "gender": str(gender or "").strip(),
        "occupation": str(occupation or "").strip(),
        "location_region": str(location_region or "").strip(),
        "location_country": str(location_country or "").strip(),
        "interest_text": str(interest_text or "").strip(),
        "top_categories_json": json.dumps(list(top_categories or [])),
        "affect_consent": int(bool(affect_consent)),
        "bio_embedding_json": json.dumps(list(bio_embedding or [])),
        "bio_text_embedding_json": json.dumps(list(bio_text_embedding or [])),
        "bio_embedding_version": str(bio_embedding_version or "").strip(),
        "onboarding_completed_at": _utc_timestamp(),
    }
    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        cursor.close()
    return load_auth_user_by_id(user_id)


def save_reading_vector(
    user_id: str,
    article_id: str,
    embedding: np.ndarray,
    feedback_weight: float = 1.0,
):
    sql = f"""
    INSERT INTO reading_history_vectors (
        user_id, article_id, read_at, feedback_weight, embedding_json
    )
    VALUES (
        {_placeholder("user_id")},
        {_placeholder("article_id")},
        CURRENT_TIMESTAMP,
        {_placeholder("feedback_weight")},
        {_placeholder("embedding_json")}
    )
    ON CONFLICT(user_id, article_id) DO UPDATE SET
        read_at = CURRENT_TIMESTAMP,
        feedback_weight = EXCLUDED.feedback_weight,
        embedding_json = EXCLUDED.embedding_json
    """
    params = {
        "user_id": str(user_id),
        "article_id": str(article_id),
        "feedback_weight": float(feedback_weight),
        "embedding_json": json.dumps(np.asarray(embedding, dtype=np.float32).tolist()),
    }
    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        cursor.close()


def query_similar_history(
    user_id: str,
    query_embedding: np.ndarray,
    top_k: int = 20,
) -> list[dict]:
    query = np.asarray(query_embedding, dtype=np.float32)
    query_norm = float(np.linalg.norm(query))
    if query_norm == 0.0:
        return []

    sql = f"""
    SELECT article_id, feedback_weight, embedding_json
    FROM reading_history_vectors
    WHERE user_id = {_placeholder("user_id")}
    ORDER BY read_at DESC
    LIMIT 200
    """
    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, {"user_id": str(user_id)})
        rows = _fetchall(cursor)
        cursor.close()

    results = []
    for row in rows:
        try:
            stored = np.asarray(_json_loads(row["embedding_json"], []), dtype=np.float32)
            stored_norm = float(np.linalg.norm(stored))
            if stored_norm == 0.0:
                continue
            similarity = float(np.dot(query, stored) / ((query_norm * stored_norm) + 1e-9))
            results.append(
                {
                    "article_id": str(row["article_id"]),
                    "feedback_weight": float(row["feedback_weight"] or 0.0),
                    "similarity": similarity,
                }
            )
        except Exception:
            continue
    results.sort(key=lambda item: item["similarity"], reverse=True)
    return results[: max(int(top_k), 0)]


def log_feedback_event(
    user_id: str,
    article_id: str,
    action: str,
    *,
    session_id: str = "",
    request_id: str = "",
    impression_id: str = "",
    position: int = -1,
    dwell_time: float = 0.0,
    query_text: str = "",
    source_feedback: str = "",
):
    sql = f"""
    INSERT INTO feedback_events (
        user_id, session_id, request_id, impression_id, article_id, action,
        position, dwell_time, query_text, source_feedback
    )
    VALUES (
        {_placeholder("user_id")},
        {_placeholder("session_id")},
        {_placeholder("request_id")},
        {_placeholder("impression_id")},
        {_placeholder("article_id")},
        {_placeholder("action")},
        {_placeholder("position")},
        {_placeholder("dwell_time")},
        {_placeholder("query_text")},
        {_placeholder("source_feedback")}
    )
    """
    params = {
        "user_id": str(user_id),
        "session_id": str(session_id or ""),
        "request_id": str(request_id or ""),
        "impression_id": str(impression_id or ""),
        "article_id": str(article_id),
        "action": str(action),
        "position": int(position),
        "dwell_time": float(dwell_time or 0.0),
        "query_text": str(query_text or ""),
        "source_feedback": str(source_feedback or ""),
    }
    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        cursor.close()


def log_recommendation_event(
    user_id: str,
    *,
    session_id: str = "",
    request_id: str = "",
    surface: str = "feed",
    mood: str = "neutral",
    mode: str = "",
    query_text: str = "",
    candidate_sources: dict | None = None,
    impression_ids: list[str] | None = None,
):
    sql = f"""
    INSERT INTO recommendation_events (
        user_id, session_id, request_id, surface, mood, mode,
        query_text, candidate_sources_json, impression_ids_json
    )
    VALUES (
        {_placeholder("user_id")},
        {_placeholder("session_id")},
        {_placeholder("request_id")},
        {_placeholder("surface")},
        {_placeholder("mood")},
        {_placeholder("mode")},
        {_placeholder("query_text")},
        {_placeholder("candidate_sources_json")},
        {_placeholder("impression_ids_json")}
    )
    """
    params = {
        "user_id": str(user_id),
        "session_id": str(session_id or ""),
        "request_id": str(request_id or ""),
        "surface": str(surface or "feed"),
        "mood": str(mood or "neutral"),
        "mode": str(mode or ""),
        "query_text": str(query_text or ""),
        "candidate_sources_json": json.dumps(candidate_sources or {}),
        "impression_ids_json": json.dumps(impression_ids or []),
    }
    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        cursor.close()


def log_ranking_feature_rows(
    user_id: str,
    *,
    session_id: str = "",
    request_id: str,
    surface: str = "feed",
    query_text: str = "",
    rows: list[dict],
):
    if not request_id or not rows:
        return

    sql = f"""
    INSERT INTO ranking_feature_events (
        user_id, session_id, request_id, impression_id, article_id,
        surface, query_text, candidate_source, position, label, dwell_time, features_json
    )
    VALUES (
        {_placeholder("user_id")},
        {_placeholder("session_id")},
        {_placeholder("request_id")},
        {_placeholder("impression_id")},
        {_placeholder("article_id")},
        {_placeholder("surface")},
        {_placeholder("query_text")},
        {_placeholder("candidate_source")},
        {_placeholder("position")},
        {_placeholder("label")},
        {_placeholder("dwell_time")},
        {_placeholder("features_json")}
    )
    """
    with _connect() as conn:
        cursor = conn.cursor()
        for row in rows:
            params = {
                "user_id": str(user_id),
                "session_id": str(session_id or ""),
                "request_id": str(request_id),
                "impression_id": str(row.get("impression_id") or row.get("article_id") or ""),
                "article_id": str(row.get("article_id") or ""),
                "surface": str(surface or "feed"),
                "query_text": str(query_text or ""),
                "candidate_source": str(row.get("candidate_source") or ""),
                "position": int(row.get("position", -1)),
                "label": row.get("label"),
                "dwell_time": float(row.get("dwell_time", 0.0) or 0.0),
                "features_json": json.dumps(row.get("features") or {}),
            }
            cursor.execute(sql, params)
        cursor.close()


_ACTION_LABEL_MAP = {
    "click": 1.0,
    "read_full": 2.0,
    "save": 3.0,
    "more_like_this": 2.0,
    "skip": 0.0,
    "not_interested": -1.0,
    "less_from_source": -0.5,
}


def update_ranking_feature_label(
    user_id: str,
    *,
    request_id: str,
    article_id: str,
    action: str,
    dwell_time: float = 0.0,
):
    request_key = str(request_id or "").strip()
    article_key = str(article_id or "").strip()
    if not request_key or not article_key:
        return

    label_value = float(_ACTION_LABEL_MAP.get(str(action), 0.0))
    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"""
            UPDATE ranking_feature_events
            SET label = {_placeholder("label")},
                dwell_time = {_placeholder("dwell_time")}
            WHERE user_id = {_placeholder("user_id")}
              AND request_id = {_placeholder("request_id")}
              AND article_id = {_placeholder("article_id")}
            """,
            {
                "label": label_value,
                "dwell_time": float(dwell_time or 0.0),
                "user_id": str(user_id),
                "request_id": request_key,
                "article_id": article_key,
            },
        )
        if label_value > 0:
            cursor.execute(
                f"""
                UPDATE ranking_feature_events
                SET label = 0.0
                WHERE user_id = {_placeholder("user_id")}
                  AND request_id = {_placeholder("request_id")}
                  AND article_id <> {_placeholder("article_id")}
                  AND label IS NULL
                """,
                {
                    "user_id": str(user_id),
                    "request_id": request_key,
                    "article_id": article_key,
                },
            )
        cursor.close()


def fetch_ltr_training_rows(limit: int = 0) -> list[dict]:
    limit_clause = f"LIMIT {max(int(limit), 0)}" if int(limit or 0) > 0 else ""
    sql = f"""
    SELECT request_id, article_id, position, label, dwell_time, candidate_source, features_json, created_at
    FROM ranking_feature_events
    WHERE label IS NOT NULL
    ORDER BY id DESC
    {limit_clause}
    """
    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = _fetchall(cursor)
        cursor.close()
    return rows


def log_search_event(
    user_id: str,
    query_text: str,
    *,
    session_id: str = "",
    normalized_query: str = "",
    intent: dict | None = None,
):
    sql = f"""
    INSERT INTO search_events (
        user_id, session_id, query_text, normalized_query, intent_json
    )
    VALUES (
        {_placeholder("user_id")},
        {_placeholder("session_id")},
        {_placeholder("query_text")},
        {_placeholder("normalized_query")},
        {_placeholder("intent_json")}
    )
    """
    params = {
        "user_id": str(user_id),
        "session_id": str(session_id or ""),
        "query_text": str(query_text or ""),
        "normalized_query": str(normalized_query or ""),
        "intent_json": json.dumps(intent or {}),
    }
    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        cursor.close()


def save_recent_state(
    user_id: str,
    *,
    recent_clicks: list[str],
    recent_skips: list[str],
    recent_negative_actions: list[str],
    recent_queries: list[str],
    recent_entities: list[str],
    recent_sources: list[str],
):
    sql = f"""
    INSERT INTO user_recent_state (
        user_id, recent_clicks_json, recent_skips_json, recent_negative_actions_json,
        recent_queries_json, recent_entities_json, recent_sources_json
    )
    VALUES (
        {_placeholder("user_id")},
        {_placeholder("recent_clicks_json")},
        {_placeholder("recent_skips_json")},
        {_placeholder("recent_negative_actions_json")},
        {_placeholder("recent_queries_json")},
        {_placeholder("recent_entities_json")},
        {_placeholder("recent_sources_json")}
    )
    ON CONFLICT(user_id) DO UPDATE SET
        recent_clicks_json = EXCLUDED.recent_clicks_json,
        recent_skips_json = EXCLUDED.recent_skips_json,
        recent_negative_actions_json = EXCLUDED.recent_negative_actions_json,
        recent_queries_json = EXCLUDED.recent_queries_json,
        recent_entities_json = EXCLUDED.recent_entities_json,
        recent_sources_json = EXCLUDED.recent_sources_json,
        updated_at = CURRENT_TIMESTAMP
    """
    params = {
        "user_id": str(user_id),
        "recent_clicks_json": json.dumps(recent_clicks or []),
        "recent_skips_json": json.dumps(recent_skips or []),
        "recent_negative_actions_json": json.dumps(recent_negative_actions or []),
        "recent_queries_json": json.dumps(recent_queries or []),
        "recent_entities_json": json.dumps(recent_entities or []),
        "recent_sources_json": json.dumps(recent_sources or []),
    }
    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        cursor.close()


def load_recent_state(user_id: str) -> Optional[dict]:
    sql = f"""
    SELECT *
    FROM user_recent_state
    WHERE user_id = {_placeholder("user_id")}
    """
    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, {"user_id": str(user_id)})
        row = _fetchone(cursor)
        cursor.close()
    if row is None:
        return None
    return {
        "recent_clicks": _json_loads(row["recent_clicks_json"], []),
        "recent_skips": _json_loads(row["recent_skips_json"], []),
        "recent_negative_actions": _json_loads(row["recent_negative_actions_json"], []),
        "recent_queries": _json_loads(row["recent_queries_json"], []),
        "recent_entities": _json_loads(row["recent_entities_json"], []),
        "recent_sources": _json_loads(row["recent_sources_json"], []),
    }


def list_recent_feedback(user_id: str, limit: int = 25) -> list[dict]:
    sql = f"""
    SELECT article_id, action, position, dwell_time, query_text, source_feedback, created_at
    FROM feedback_events
    WHERE user_id = {_placeholder("user_id")}
    ORDER BY id DESC
    LIMIT {max(int(limit), 0)}
    """
    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, {"user_id": str(user_id)})
        rows = _fetchall(cursor)
        cursor.close()
    return rows


def list_recent_searches(user_id: str, limit: int = 25) -> list[dict]:
    sql = f"""
    SELECT query_text, normalized_query, created_at
    FROM search_events
    WHERE user_id = {_placeholder("user_id")}
    ORDER BY id DESC
    LIMIT {max(int(limit), 0)}
    """
    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, {"user_id": str(user_id)})
        rows = _fetchall(cursor)
        cursor.close()
    return rows
