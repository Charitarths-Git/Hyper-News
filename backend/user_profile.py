"""
user_profile.py — In-memory user model with two-tier session storage:

  RECENT (hot)  → Redis sorted set, keyed by timestamp (1-hour TTL).
                  Fast sliding-window access; evicts oldest automatically.
  LONG-TERM     → PostgreSQL + pgvector (or SQLite blob fallback) in db.py.
                  Semantic similarity queries over all past reads.

Falls back to an in-process TTL dict when Redis is unavailable.
"""
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

# ── Redis with graceful fallback ──────────────────────────────────────────────
try:
    import redis as _redis_lib
    _r = _redis_lib.Redis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379"),
        decode_responses=True,
        socket_connect_timeout=1,
    )
    _r.ping()
    _REDIS_OK = True
except Exception:
    _r = None
    _REDIS_OK = False

_mem_store: dict = {}   # {key: (value, expires_at)}

_RECENT_HISTORY_TTL = 3600          # 1 hour for recent-click sorted set
_RECENT_HISTORY_MAX = 100           # keep more raw history while recent state trims to 25
_SESSION_TTL        = 3600
_RECENT_STATE_LIMIT = 25
_RECENT_QUERY_LIMIT = 10


def _redis_set(key: str, value: str, ttl: int = _SESSION_TTL):
    if _REDIS_OK:
        _r.setex(key, ttl, value)
    else:
        _mem_store[key] = (value, time.time() + ttl)


def _redis_get(key: str) -> Optional[str]:
    if _REDIS_OK:
        return _r.get(key)
    entry = _mem_store.get(key)
    if entry and time.time() < entry[1]:
        return entry[0]
    return None


def _redis_delete(key: str):
    if _REDIS_OK:
        _r.delete(key)
    else:
        _mem_store.pop(key, None)


# ── Redis sorted-set helpers for recent history ───────────────────────────────

def _history_key(user_id: str) -> str:
    return f"history:{user_id}"


def push_recent_history(user_id: str, article_id: str, score: float = 1.0):
    """Add article_id to the user's recent-history sorted set.

    Score = unix_timestamp so ZRANGE … BYSCORE gives chronological order.
    We keep only the most recent _RECENT_HISTORY_MAX entries and reset TTL.
    """
    key = _history_key(user_id)
    now = time.time()
    if _REDIS_OK:
        pipe = _r.pipeline()
        pipe.zadd(key, {article_id: now})
        # trim to most-recent max items
        pipe.zremrangebyrank(key, 0, -((_RECENT_HISTORY_MAX + 1)))
        pipe.expire(key, _RECENT_HISTORY_TTL)
        pipe.execute()
    else:
        # fallback: store as a list inside _mem_store
        raw = _mem_store.get(key)
        lst: list = json.loads(raw[0]) if raw and time.time() < raw[1] else []
        if article_id not in lst:
            lst.append(article_id)
        if len(lst) > _RECENT_HISTORY_MAX:
            lst = lst[-_RECENT_HISTORY_MAX:]
        _mem_store[key] = (json.dumps(lst), time.time() + _RECENT_HISTORY_TTL)


def get_recent_history(user_id: str, limit: int = _RECENT_STATE_LIMIT) -> list[str]:
    """Return the most-recent `limit` article IDs from the hot Redis layer."""
    key = _history_key(user_id)
    if _REDIS_OK:
        # ZREVRANGE returns highest-score (most recent) first
        return list(reversed(_r.zrevrange(key, 0, limit - 1)))
    raw = _mem_store.get(key)
    if raw and time.time() < raw[1]:
        lst = json.loads(raw[0])
        return lst[-limit:][::-1]
    return []


def clear_recent_history(user_id: str):
    _redis_delete(_history_key(user_id))


# ── User Profile ──────────────────────────────────────────────────────────────
@dataclass
class UserProfile:
    user_id:         str
    display_name:    str              = ""
    email:           str              = ""
    interests:       Dict[str, float] = field(default_factory=dict)
    reading_history: List[str]        = field(default_factory=list)
    avg_dwell_time:  float            = 0.0
    mood:            str              = "neutral"
    time_of_day:     str              = "morning"
    recent_clicks:   List[str]        = field(default_factory=list)
    recent_skips:    List[str]        = field(default_factory=list)
    recent_negative_actions: List[str] = field(default_factory=list)
    session_topics:  List[str]        = field(default_factory=list)
    recent_queries:  List[str]        = field(default_factory=list)
    recent_entities: List[str]        = field(default_factory=list)
    recent_sources:  List[str]        = field(default_factory=list)
    age_bucket:      str              = ""
    gender:          str              = ""
    occupation:      str              = ""
    location_region: str              = ""
    location_country: str             = ""
    interest_text:   str              = ""
    top_categories:  List[str]        = field(default_factory=list)
    affect_consent:  bool             = False
    bio_embedding:   List[float]      = field(default_factory=list)
    bio_text_embedding: List[float]   = field(default_factory=list)
    bio_embedding_version: str        = ""
    onboarding_completed: bool        = False
    onboarding_completed_at: str      = ""

    # how many positive interactions has this user had total?
    # used to gate cold-start vs RL mode
    total_positive_interactions: int  = 0

    # incremented on every feedback event; used by EMA decay logic
    interest_update_count: int        = 0

    # ephemeral: last candidate pool served to this user (for category reward propagation)
    # not persisted to Redis or SQLite
    _last_candidate_pool: list        = field(default_factory=list, repr=False)


def _trim_recent(values: List[str], limit: int) -> List[str]:
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[-limit:]


def append_recent(values: List[str], item: str, limit: int = _RECENT_STATE_LIMIT, dedupe: bool = False) -> List[str]:
    normalized = str(item or "").strip()
    if not normalized:
        return _trim_recent(values, limit)
    if dedupe:
        values = [value for value in values if str(value).strip() != normalized]
    values.append(normalized)
    return _trim_recent(values, limit)


def extend_recent(values: List[str], items: List[str], limit: int = _RECENT_STATE_LIMIT, dedupe: bool = True) -> List[str]:
    current = list(values)
    for item in items:
        current = append_recent(current, item, limit=limit, dedupe=dedupe)
    return current


def update_user_state(
    user: UserProfile,
    *,
    query: str | None = None,
    entities: list[str] | None = None,
    source: str | None = None,
    article_id: str | None = None,
    category: str | None = None,
    negative: bool = False,
):
    if query:
        user.recent_queries = append_recent(user.recent_queries, query, limit=_RECENT_QUERY_LIMIT)
    if entities:
        user.recent_entities = extend_recent(user.recent_entities, list(entities), limit=_RECENT_STATE_LIMIT)
    if source:
        user.recent_sources = append_recent(user.recent_sources, source, limit=_RECENT_STATE_LIMIT, dedupe=True)
    if article_id:
        if negative:
            user.recent_negative_actions = append_recent(
                user.recent_negative_actions,
                article_id,
                limit=_RECENT_STATE_LIMIT,
                dedupe=True,
            )
        else:
            user.recent_clicks = append_recent(user.recent_clicks, article_id, limit=_RECENT_STATE_LIMIT, dedupe=True)
    if category:
        user.session_topics = append_recent(user.session_topics, category, limit=_RECENT_STATE_LIMIT)


# ── Context helpers ───────────────────────────────────────────────────────────
def get_time_of_day() -> str:
    hour = datetime.now().hour
    if 5  <= hour < 12: return "morning"
    elif 12 <= hour < 17: return "afternoon"
    elif 17 <= hour < 21: return "evening"
    else:                  return "night"


MOOD_WEIGHTS = {
    "stressed": {"entertainment": 1.5, "sports": 1.3, "politics": 0.4},
    "curious":  {"technology": 1.5, "science": 1.4, "business": 1.2},
    "tired":    {"entertainment": 1.4, "lifestyle": 1.3, "politics": 0.3},
    "happy":    {"sports": 1.2, "entertainment": 1.2, "technology": 1.2},
    "neutral":  {},
}

TIME_WEIGHTS = {
    "morning":   {"politics": 1.3, "business": 1.2, "technology": 1.1},
    "afternoon": {"sports": 1.2, "business": 1.2},
    "evening":   {"entertainment": 1.3, "lifestyle": 1.2},
    "night":     {"entertainment": 1.5, "lifestyle": 1.4, "politics": 0.5},
}


def compute_context_score(article_category: str, mood: str, time_of_day: str) -> float:
    category_key = str(article_category or "").strip().lower()
    score = 1.0
    score *= MOOD_WEIGHTS.get(mood, {}).get(category_key, 1.0)
    score *= TIME_WEIGHTS.get(time_of_day, {}).get(category_key, 1.0)
    return score


# ── Session store (Redis / fallback) ─────────────────────────────────────────
def update_user_session(user: UserProfile):
    """Persist short-term session data to Redis (or memory fallback) for 1 hour."""
    key = f"session:{user.user_id}"
    data = {
        "recent_clicks":               user.recent_clicks[-_RECENT_STATE_LIMIT:],
        "recent_skips":                user.recent_skips[-_RECENT_STATE_LIMIT:],
        "recent_negative_actions":     user.recent_negative_actions[-_RECENT_STATE_LIMIT:],
        "session_topics":              user.session_topics[-_RECENT_STATE_LIMIT:],
        "recent_queries":              user.recent_queries[-_RECENT_QUERY_LIMIT:],
        "recent_entities":             user.recent_entities[-_RECENT_STATE_LIMIT:],
        "recent_sources":              user.recent_sources[-_RECENT_STATE_LIMIT:],
        "mood":                        user.mood,
        "avg_dwell_time":              float(user.avg_dwell_time or 0.0),
        "total_positive_interactions": user.total_positive_interactions,
        "interest_update_count":       user.interest_update_count,
    }
    _redis_set(key, json.dumps(data), ttl=_SESSION_TTL)


def load_user_session(user_id: str) -> dict:
    """Load short-term session from Redis / fallback."""
    raw = _redis_get(f"session:{user_id}")
    if raw:
        return json.loads(raw)
    return {}


def clear_user_session(user_id: str):
    """Remove short-term session state and recent-history sorted set."""
    _redis_delete(f"session:{user_id}")
    clear_recent_history(user_id)
