"""
FastAPI backend for HyperNews.
Loads persisted data, serves recommendations, and records feedback.
"""

import os
import json
import re
import sys
import time
import uuid
from collections import Counter
from typing import Dict, Iterable, Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from dotenv import load_dotenv

load_dotenv(os.path.join(_BACKEND_DIR, "..", ".env"))

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from bandit import LinUCBBandit
from db import (
    _json_loads,
    complete_auth_user_onboarding,
    create_auth_user,
    delete_user,
    fetch_ltr_training_rows,
    get_database_backend_info,
    init_db,
    list_recent_feedback,
    list_recent_searches,
    load_auth_user_by_email,
    load_auth_user_by_id,
    load_recent_state,
    load_user,
    log_feedback_event,
    log_ranking_feature_rows,
    log_recommendation_event,
    log_search_event,
    save_recent_state,
    save_reading_vector,
    save_user,
    update_auth_user_profile,
    update_ranking_feature_label,
)
from graph import build_knowledge_graph, get_graph_stats
from hybrid_search import build_hybrid_candidates, prepare_search_cache, suggest_queries
from ltr import HybridLTRScorer
from mind_data import parse_entity_list
from rag_pipeline import (
    build_faiss_index,
    generate_personalized_summary,
    load_vector_index,
    save_vector_index,
)
from coldstart_hints import (
    extract_interest_terms,
    infer_interest_category_weights,
    location_terms,
)
from ranker import (
    _apply_interest_ema_decay,
    _build_news_id_to_idx,
    _graph_bonus_map,
    _normalize_key,
    build_candidate_pool,
    build_context_vector,
    build_long_term_memory_signal,
    build_query_candidate_pool,
    cold_start_recommendations,
    get_kg_related_ids,
    rank_articles,
    rank_search_articles,
)
from user_profile import (
    UserProfile,
    clear_user_session,
    get_time_of_day,
    load_user_session,
    push_recent_history,
    update_user_state,
    update_user_session,
)
from auth_utils import hash_password, verify_password
from service_backends import OpenSearchArticleStore, QdrantVectorStore

_BASE = os.path.join(_BACKEND_DIR, "..")
DATA_DIR = os.path.join(_BASE, "data")
GRAPH_DIR = os.path.join(_BASE, "graph")
DEFAULT_PARQUET = os.path.join(DATA_DIR, "articles.parquet")
DEFAULT_FAISS = os.path.join(DATA_DIR, "faiss_mind.index")
PARQUET_FILES = [
    DEFAULT_PARQUET,
    os.path.join(DATA_DIR, "news_processed.parquet"),
]
FAISS_FILES = [
    DEFAULT_FAISS,
    os.path.join(DATA_DIR, "news_faiss.index"),
]
EMB_FILE = os.path.join(DATA_DIR, "article_embeddings.npy")
BANDIT_FILE = os.path.join(_BASE, "models", "bandit_model.pkl")
GRAPH_ENABLED = os.getenv("HYPERNEWS_ENABLE_GRAPH", "1").strip().lower() not in {"0", "false", "no"}
GRAPH_ARTICLE_LIMIT = int(os.getenv("HYPERNEWS_GRAPH_ARTICLE_LIMIT", "0") or 0)
STARTUP_MAX_ARTICLES = int(os.getenv("HYPERNEWS_MAX_ARTICLES", "0") or 0)
QUERY_MODEL_ENABLED = os.getenv("HYPERNEWS_ENABLE_SENTENCE_TRANSFORMER", "0").strip().lower() in {"1", "true", "yes"}
QUERY_MODEL_NAME = os.getenv("HYPERNEWS_QUERY_MODEL_NAME", "BAAI/bge-small-en-v1.5")
RERANKER_ENABLED = os.getenv("HYPERNEWS_ENABLE_RERANKER", "0").strip().lower() in {"1", "true", "yes"}
RERANKER_MODEL_NAME = os.getenv("HYPERNEWS_RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L6-v2")
DEBUG_SCORE_BREAKDOWN = os.getenv("HYPERNEWS_DEBUG_SCORE_BREAKDOWN", "0").strip().lower() in {"1", "true", "yes"}
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_ENABLED = os.getenv("HYPERNEWS_ENABLE_QDRANT", "1").strip().lower() in {"1", "true", "yes"}
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
OPENSEARCH_ENABLED = os.getenv("HYPERNEWS_ENABLE_OPENSEARCH", "1").strip().lower() in {"1", "true", "yes"}

# ── Exploration / feedback policy ─────────────────────────────────────────────
# Stay in semi-cold-start until this many confirmed positive interactions.
# Must match _INTEREST_WARMUP_INTERACTIONS in ranker.py (both = 5).
_SEMI_COLD_START_THRESHOLD = 5

# Interest nudge: fraction of reward added to category weight per event.
# click: 0.5 * 0.25 = 0.125; save: 2.0 * 0.25 = 0.5. Prevents instant lock-in.
_INTEREST_NUDGE_FACTOR = 0.25

# Skip: relative decay + fixed penalty on the category weight.
_SKIP_DECAY_FACTOR = 0.08
_SKIP_FIXED_PENALTY = 0.03

# Reward map used for bandit training (not for interest accumulation).
_REWARD_MAP = {
    "click": 0.5,
    "read_full": 1.0,
    "skip": -0.3,
    "save": 2.0,
    "more_like_this": 1.3,
    "not_interested": -0.8,
    "less_from_source": -0.6,
}
_DEFAULT_DWELL_BASELINE = 30.0
_DWELL_EMA_ALPHA = 0.20

# Decay applied to sibling articles in the same category on positive feedback.
_CATEGORY_PROPAGATION_DECAY = 0.12
_KG_PROPAGATION_DECAY = 0.06
_CATEGORY_SIBLING_PROPAGATION_LIMIT = 4
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="HyperNews Recommendation API", version="2.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

USERS: Dict[str, UserProfile] = {}
BANDIT: Optional[LinUCBBandit] = None
DF: pd.DataFrame = pd.DataFrame()
EMBEDDINGS: np.ndarray = np.array([])
FAISS_INDEX: Optional[object] = None
KG_GRAPH: Optional[object] = None
MODEL: Optional[object] = None
MODEL_LOAD_ATTEMPTED = False
RERANKER: Optional[object] = None
RERANKER_LOAD_ATTEMPTED = False
LTR_SCORER: Optional[HybridLTRScorer] = None
VECTOR_BACKEND: Optional[QdrantVectorStore] = None
SEARCH_BACKEND: Optional[OpenSearchArticleStore] = None
BIO_ENCODER: Optional[object] = None
BIO_ENCODER_LOAD_ATTEMPTED = False


class RecommendRequest(BaseModel):
    user_id: str
    mood: str = "neutral"
    n: int = 10
    query: Optional[str] = None
    exclude_ids: list[str] = Field(default_factory=list)
    session_id: str = ""
    request_id: str = ""
    surface: str = "feed"
    explore_focus: float = 55.0


class FeedbackRequest(BaseModel):
    user_id: str
    article_id: str
    action: str
    dwell_time: float = 0.0
    session_id: str = ""
    request_id: str = ""
    impression_id: str = ""
    position: int = -1
    query_text: str = ""
    source_feedback: str = ""


class RegisterRequest(BaseModel):
    email: str
    password: str
    display_name: str = ""
    age_bucket: str = ""
    gender: str = ""
    occupation: str = ""
    location_region: str = ""
    location_country: str = ""
    interest_text: str = ""
    top_categories: list[str] = Field(default_factory=list)
    affect_consent: bool = False


class ProfileUpdateRequest(BaseModel):
    display_name: str = ""
    age_bucket: str = ""
    gender: str = ""
    occupation: str = ""
    location_region: str = ""
    location_country: str = ""
    interest_text: str = ""
    top_categories: list[str] = Field(default_factory=list)
    affect_consent: bool = False


class OnboardingRequest(BaseModel):
    age_bucket: str = ""
    gender: str = ""
    occupation: str = ""
    location_region: str = ""
    location_country: str = ""
    interest_text: str = ""
    top_categories: list[str] = Field(default_factory=list)
    affect_consent: bool = False


class LoginRequest(BaseModel):
    email: str
    password: str


_SAFE_COLS = {
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "popularity",
    "score",
    "source",
    "candidate_source",
    "ltr_score",
    "bandit_score",
}
_SAFE_LIST_COLS = {"reasons", "matched_entities", "matched_categories"}
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_BIO_EMBEDDING_VERSION = "bio-v1"
_BIO_AGE_BUCKETS = {"", "<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+", "unknown"}
_BIO_GENDER_CATS = {"", "male", "female", "nonbinary", "prefer_not", "unknown"}
_BIO_OCCUPATION_CATS = {
    "",
    "student",
    "engineer",
    "teacher",
    "doctor",
    "lawyer",
    "journalist",
    "artist",
    "finance",
    "government",
    "retail",
    "other",
    "unknown",
}
_BIO_LOCATION_CATS = {
    "",
    "north_america",
    "europe",
    "asia",
    "latin_america",
    "africa",
    "oceania",
    "unknown",
}
_BIO_CATEGORY_LIMIT = 6
_BIO_EMBEDDING_MATCH_LIMIT = 120
BIO_ENCODER_FILE = os.path.join(_BASE, "models", "bio_encoder.pt")


def _to_python(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, float) and value != value:
        return None
    return value


def _log_structured(event: str, **payload):
    entry = {"event": event, **payload}
    try:
        print(json.dumps(entry, default=_to_python))
    except Exception:
        print(entry)


def _normalize_surface(value: str) -> str:
    surface = str(value or "feed").strip().lower()
    return surface if surface in {"feed", "search"} else "feed"


def _normalize_query_text(value: str | None) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


def sanitize_article(article: dict) -> dict:
    sanitized = {}
    for key, value in article.items():
        if key in _SAFE_COLS:
            if isinstance(value, (list, dict)):
                continue
            sanitized[key] = _to_python(value)
        elif key in _SAFE_LIST_COLS and isinstance(value, list):
            sanitized[key] = [str(item) for item in value if str(item).strip()]
        elif key == "score_breakdown" and DEBUG_SCORE_BREAKDOWN and isinstance(value, dict):
            sanitized[key] = {
                str(inner_key): _to_python(inner_value)
                for inner_key, inner_value in value.items()
            }
    return sanitized


def _normalize_article_ids(values: Iterable[str] | None) -> set[str]:
    return {
        str(value).strip()
        for value in (values or [])
        if str(value).strip()
    }


def _candidate_source_distribution(items: Iterable[dict]) -> Counter[str]:
    distribution: Counter[str] = Counter()
    for article in items:
        source = str(article.get("candidate_source", "") or "ranker").strip()
        parts = [part.strip() for part in source.split("+") if part.strip()] or [source]
        distribution.update(parts)
    return distribution


def _normalize_email(email: str) -> str:
    return str(email or "").strip().lower()


def _normalize_bio_choice(value: str, allowed: set[str]) -> str:
    normalized = _normalize_key(value)
    return normalized if normalized in allowed else ""


def _normalize_category_list(values) -> list[str]:
    if isinstance(values, str):
        raw_values = re.split(r"[,|;/]+", values)
    elif isinstance(values, list):
        raw_values = values
    else:
        raw_values = []

    cleaned: list[str] = []
    for value in raw_values:
        normalized = _normalize_key(value)
        if normalized and normalized not in cleaned:
            cleaned.append(normalized)
        if len(cleaned) >= _BIO_CATEGORY_LIMIT:
            break
    return cleaned


def _float_list(value) -> list[float]:
    if value is None:
        return []
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return []
    return arr.astype(np.float32).tolist()


def _interest_terms(value: str) -> list[str]:
    return extract_interest_terms(value, limit=10)


def _expand_top_categories(top_categories: list[str], interest_text: str) -> list[str]:
    return list(_normalize_category_list(top_categories))[:_BIO_CATEGORY_LIMIT]


def _build_bio_text_embedding(
    top_categories: list[str],
    interest_text: str,
    location_region: str = "",
    location_country: str = "",
) -> np.ndarray | None:
    if len(DF) == 0 or len(EMBEDDINGS) == 0:
        return None

    matched_indices: list[int] = []
    category_keys = list(
        dict.fromkeys(
            list(top_categories or [])
            + list(infer_interest_category_weights(interest_text, limit=_BIO_CATEGORY_LIMIT).keys())
        )
    )[:_BIO_CATEGORY_LIMIT]
    if category_keys:
        category_series = DF["category"].fillna("").astype(str).str.lower()
        subcategory_series = (
            DF["subcategory"].fillna("").astype(str).str.lower()
            if "subcategory" in DF.columns
            else pd.Series("", index=DF.index)
        )
        category_mask = category_series.isin(category_keys) | subcategory_series.isin(category_keys)
        category_rows = DF.loc[category_mask]
        if "popularity" in category_rows.columns:
            category_rows = category_rows.sort_values("popularity", ascending=False)
        matched_indices.extend(category_rows.index.tolist())

    if len(matched_indices) < 24:
        terms = list(
            dict.fromkeys(
                _interest_terms(interest_text)
                + location_terms(location_region, location_country)
            )
        )
        if terms:
            pattern = "|".join(re.escape(term) for term in terms[:8])
            title_series = DF["title"].fillna("").astype(str).str.lower() if "title" in DF.columns else pd.Series("", index=DF.index)
            abstract_series = DF["abstract"].fillna("").astype(str).str.lower() if "abstract" in DF.columns else pd.Series("", index=DF.index)
            text_mask = title_series.str.contains(pattern, regex=True) | abstract_series.str.contains(pattern, regex=True)
            text_rows = DF.loc[text_mask]
            if "popularity" in text_rows.columns:
                text_rows = text_rows.sort_values("popularity", ascending=False)
            matched_indices.extend(text_rows.index.tolist())

    deduped_indices: list[int] = []
    seen: set[int] = set()
    for idx in matched_indices:
        if idx in seen or idx >= len(EMBEDDINGS):
            continue
        seen.add(idx)
        deduped_indices.append(int(idx))
        if len(deduped_indices) >= _BIO_EMBEDDING_MATCH_LIMIT:
            break

    if not deduped_indices:
        return None

    vector = np.asarray(EMBEDDINGS[deduped_indices], dtype=np.float32).mean(axis=0)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return None
    return (vector / norm).astype(np.float32)


def _load_bio_encoder_model():
    global BIO_ENCODER, BIO_ENCODER_LOAD_ATTEMPTED
    if BIO_ENCODER_LOAD_ATTEMPTED:
        return BIO_ENCODER

    BIO_ENCODER_LOAD_ATTEMPTED = True
    if not os.path.exists(BIO_ENCODER_FILE):
        return None

    try:
        from models.bio_encoder import load_bio_encoder
    except Exception as exc:
        print(f"Bio encoder import unavailable ({exc}); storing text-only cold-start profile.")
        return None

    try:
        BIO_ENCODER = load_bio_encoder(BIO_ENCODER_FILE, device="cpu")
    except Exception as exc:
        print(f"Bio encoder weights unavailable ({exc}); storing text-only cold-start profile.")
        BIO_ENCODER = None
    return BIO_ENCODER


def _build_bio_profile(
    *,
    age_bucket: str = "",
    gender: str = "",
    occupation: str = "",
    location_region: str = "",
    location_country: str = "",
    interest_text: str = "",
    top_categories: list[str] | None = None,
    affect_consent: bool = False,
) -> dict:
    normalized_age_bucket = _normalize_bio_choice(age_bucket, _BIO_AGE_BUCKETS)
    normalized_gender = _normalize_bio_choice(gender, _BIO_GENDER_CATS)
    normalized_occupation = _normalize_bio_choice(occupation, _BIO_OCCUPATION_CATS)
    normalized_location = _normalize_bio_choice(location_region, _BIO_LOCATION_CATS)
    normalized_location_country = str(location_country or "").strip()
    normalized_interest_text = str(interest_text or "").strip()
    normalized_top_categories = _expand_top_categories(list(top_categories or []), normalized_interest_text)

    bio_text_embedding = _build_bio_text_embedding(
        normalized_top_categories,
        normalized_interest_text,
        normalized_location,
        normalized_location_country,
    )
    bio_embedding: list[float] = []

    model = _load_bio_encoder_model()
    if model is not None:
        try:
            from models.bio_encoder import encode_bio_fields

            bio_embedding = _float_list(
                encode_bio_fields(
                    age_bucket=normalized_age_bucket or "unknown",
                    gender=normalized_gender or "unknown",
                    occupation=normalized_occupation or "unknown",
                    location_region=normalized_location or "unknown",
                    interest_text_emb=bio_text_embedding,
                    model=model,
                    device="cpu",
                )
            )
        except Exception as exc:
            print(f"Bio encoder inference failed ({exc}); using text-only cold-start profile.")

    return {
        "age_bucket": normalized_age_bucket,
        "gender": normalized_gender,
        "occupation": normalized_occupation,
        "location_region": normalized_location,
        "location_country": normalized_location_country,
        "interest_text": normalized_interest_text,
        "top_categories": normalized_top_categories,
        "affect_consent": bool(affect_consent),
        "bio_embedding": bio_embedding,
        "bio_text_embedding": _float_list(bio_text_embedding),
        "bio_embedding_version": _BIO_EMBEDDING_VERSION if (bio_embedding or bio_text_embedding is not None) else "",
    }


def _apply_auth_profile(user: UserProfile, record: dict | None) -> None:
    if not record:
        return

    user.display_name = str(record.get("display_name") or "")
    user.email = str(record.get("email") or "")
    user.age_bucket = str(record.get("age_bucket") or "")
    user.gender = str(record.get("gender") or "")
    user.occupation = str(record.get("occupation") or "")
    user.location_region = str(record.get("location_region") or "")
    user.location_country = str(record.get("location_country") or "")
    user.interest_text = str(record.get("interest_text") or "")
    user.top_categories = _normalize_category_list(_json_loads(record.get("top_categories_json"), []))
    user.affect_consent = bool(record.get("affect_consent"))
    user.bio_embedding = _float_list(_json_loads(record.get("bio_embedding_json"), []))
    user.bio_text_embedding = _float_list(_json_loads(record.get("bio_text_embedding_json"), []))
    user.bio_embedding_version = str(record.get("bio_embedding_version") or "")
    user.onboarding_completed = bool(record.get("onboarding_completed"))
    user.onboarding_completed_at = str(record.get("onboarding_completed_at") or "")


def _has_onboarding_answers(
    *,
    age_bucket: str = "",
    gender: str = "",
    occupation: str = "",
    location_region: str = "",
    location_country: str = "",
    interest_text: str = "",
    top_categories: list[str] | None = None,
    affect_consent: bool = False,
) -> bool:
    return any(
        [
            str(age_bucket or "").strip(),
            str(gender or "").strip(),
            str(occupation or "").strip(),
            str(location_region or "").strip(),
            str(location_country or "").strip(),
            str(interest_text or "").strip(),
            list(top_categories or []),
            bool(affect_consent),
        ]
    )


def _validate_registration_payload(email: str, password: str) -> Optional[str]:
    normalized_email = _normalize_email(email)
    if not normalized_email or not _EMAIL_RE.match(normalized_email):
        return "A valid email address is required."
    if len(str(password or "")) < 8:
        return "Password must be at least 8 characters."
    return None


def _auth_payload(record: dict) -> dict:
    return {
        "user_id": str(record.get("user_id") or ""),
        "email": str(record.get("email") or ""),
        "display_name": str(record.get("display_name") or ""),
        "age_bucket": str(record.get("age_bucket") or ""),
        "gender": str(record.get("gender") or ""),
        "occupation": str(record.get("occupation") or ""),
        "location_region": str(record.get("location_region") or ""),
        "location_country": str(record.get("location_country") or ""),
        "interest_text": str(record.get("interest_text") or ""),
        "top_categories": _normalize_category_list(_json_loads(record.get("top_categories_json"), [])),
        "affect_consent": bool(record.get("affect_consent")),
        "bio_embedding_version": str(record.get("bio_embedding_version") or ""),
        "onboarding_completed": bool(record.get("onboarding_completed")),
        "onboarding_completed_at": str(record.get("onboarding_completed_at") or ""),
    }


def _select_visible_sibling_ids(
    current_article_id: str,
    category: str,
    subcategory: str,
    served_articles: list[dict],
    limit: int = _CATEGORY_SIBLING_PROPAGATION_LIMIT,
) -> list[str]:
    normalized_category = _normalize_key(category)
    normalized_subcategory = _normalize_key(subcategory)
    same_subcategory: list[str] = []
    same_category: list[str] = []

    for article in served_articles:
        news_id = str(article.get("news_id") or "").strip()
        if not news_id or news_id == current_article_id:
            continue
        article_category = _normalize_key(article.get("category", ""))
        if article_category != normalized_category:
            continue
        article_subcategory = _normalize_key(article.get("subcategory", ""))
        if normalized_subcategory and article_subcategory == normalized_subcategory:
            same_subcategory.append(news_id)
        else:
            same_category.append(news_id)

    ordered = same_subcategory + same_category
    return ordered[:limit]


def _resolve_existing_path(candidates: Iterable[str]) -> Optional[str]:
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _ensure_article_frame(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy().reset_index(drop=True)
    if "subcategory" not in prepared.columns:
        prepared["subcategory"] = ""
    for column in ("entities", "title_entities", "abstract_entities"):
        if column in prepared.columns:
            prepared[column] = prepared[column].apply(parse_entity_list)
    for column in ("entity_ids", "entity_labels"):
        if column in prepared.columns:
            prepared[column] = prepared[column].apply(
                lambda value: value if isinstance(value, list) else ([] if value is None else [value] if isinstance(value, str) and value.strip() else [])
            )
    if "text" not in prepared.columns:
        title = prepared["title"].fillna("").astype(str) if "title" in prepared.columns else ""
        abstract = prepared["abstract"].fillna("").astype(str) if "abstract" in prepared.columns else ""
        prepared["text"] = title + ". " + abstract
    return prepared


def _normalize_embedding_matrix(embeddings: np.ndarray) -> np.ndarray:
    arr = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _clip_dwell_time_seconds(dwell_time: float) -> float:
    try:
        value = float(dwell_time)
    except (TypeError, ValueError):
        value = 0.0
    return float(np.clip(value, 0.0, 300.0))


def _load_sentence_transformer_model(model_name: str = QUERY_MODEL_NAME):
    if not QUERY_MODEL_ENABLED:
        print(
            "SentenceTransformer query model disabled. "
            "Using lexical RAG retrieval fallback."
        )
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        print(
            f"SentenceTransformer unavailable ({exc}). "
            "Query retrieval will fall back to the standard candidate pool."
        )
        return None
    return SentenceTransformer(model_name)


def _load_cross_encoder_model(model_name: str = RERANKER_MODEL_NAME):
    if not RERANKER_ENABLED:
        print("CrossEncoder reranker disabled. Using deterministic fallback reranking.")
        return None
    try:
        from sentence_transformers import CrossEncoder
    except Exception as exc:
        print(f"CrossEncoder unavailable ({exc}). Using deterministic fallback reranking.")
        return None
    try:
        return CrossEncoder(model_name)
    except Exception as exc:
        print(f"CrossEncoder init failed ({exc}). Using deterministic fallback reranking.")
        return None


def _hydrate_user_recent_state(user: UserProfile):
    state = load_recent_state(user.user_id)
    if not state:
        return
    user.recent_clicks = state.get("recent_clicks", user.recent_clicks)
    user.recent_skips = state.get("recent_skips", user.recent_skips)
    user.recent_negative_actions = state.get("recent_negative_actions", getattr(user, "recent_negative_actions", []))
    user.recent_queries = state.get("recent_queries", getattr(user, "recent_queries", []))
    user.recent_entities = state.get("recent_entities", getattr(user, "recent_entities", []))
    user.recent_sources = state.get("recent_sources", getattr(user, "recent_sources", []))


def _persist_user_recent_state(user: UserProfile):
    save_recent_state(
        user.user_id,
        recent_clicks=list(user.recent_clicks[-25:]),
        recent_skips=list(user.recent_skips[-25:]),
        recent_negative_actions=list(getattr(user, "recent_negative_actions", [])[-25:]),
        recent_queries=list(getattr(user, "recent_queries", [])[-10:]),
        recent_entities=list(getattr(user, "recent_entities", [])[-25:]),
        recent_sources=list(getattr(user, "recent_sources", [])[-25:]),
    )


def _profile_payload(user: UserProfile) -> dict:
    recent_feedback = list_recent_feedback(user.user_id, limit=25)
    recent_searches = list_recent_searches(user.user_id, limit=25)
    return {
        "user_id": user.user_id,
        "display_name": user.display_name,
        "email": user.email,
        "mood": user.mood,
        "time_of_day": user.time_of_day,
        "interests": user.interests,
        "avg_dwell_time": float(user.avg_dwell_time or 0.0),
        "articles_read": len(user.reading_history),
        "recent_clicks": user.recent_clicks[-25:],
        "recent_skips": user.recent_skips[-25:],
        "recent_negative_actions": getattr(user, "recent_negative_actions", [])[-25:],
        "session_topics": user.session_topics[-25:],
        "recent_queries": getattr(user, "recent_queries", [])[-10:],
        "recent_entities": getattr(user, "recent_entities", [])[-25:],
        "recent_sources": getattr(user, "recent_sources", [])[-25:],
        "recent_searches": recent_searches,
        "recent_feedback": recent_feedback,
        "total_positive_interactions": user.total_positive_interactions,
        "age_bucket": user.age_bucket,
        "gender": user.gender,
        "occupation": user.occupation,
        "location_region": user.location_region,
        "location_country": getattr(user, "location_country", ""),
        "interest_text": user.interest_text,
        "top_categories": list(user.top_categories),
        "affect_consent": bool(user.affect_consent),
        "bio_embedding_version": user.bio_embedding_version,
        "has_bio_embedding": bool(user.bio_embedding),
        "has_bio_text_embedding": bool(user.bio_text_embedding),
        "onboarding_completed": bool(user.onboarding_completed),
        "onboarding_completed_at": str(user.onboarding_completed_at or ""),
    }


def _dwell_reward_multiplier(action: str, dwell_time: float, avg_dwell_time: float) -> float:
    dwell = _clip_dwell_time_seconds(dwell_time)
    if dwell <= 0.0:
        return 1.0

    baseline = max(float(avg_dwell_time or 0.0), _DEFAULT_DWELL_BASELINE)
    ratio = dwell / baseline

    if action in ("click", "read_full", "save"):
        return float(np.clip(0.45 + (0.65 * ratio), 0.25, 1.35))
    if action == "skip" and (dwell < 5.0 or ratio < 0.35):
        return 1.15
    return 1.0


def _compute_feedback_reward(action: str, dwell_time: float, avg_dwell_time: float) -> float:
    base_reward = float(_REWARD_MAP.get(action, 0.0))
    dwell = _clip_dwell_time_seconds(dwell_time)
    if dwell <= 0.0:
        return base_reward
    return float(base_reward * _dwell_reward_multiplier(action, dwell, avg_dwell_time))


def _update_avg_dwell_time(user: UserProfile, dwell_time: float) -> None:
    dwell = _clip_dwell_time_seconds(dwell_time)
    if dwell <= 0.0:
        return
    if float(user.avg_dwell_time or 0.0) <= 0.0:
        user.avg_dwell_time = dwell
        return
    user.avg_dwell_time = (
        ((1.0 - _DWELL_EMA_ALPHA) * float(user.avg_dwell_time))
        + (_DWELL_EMA_ALPHA * dwell)
    )


def _redis_status():
    from user_profile import _REDIS_OK
    return _REDIS_OK


def _graph_frame(df: pd.DataFrame) -> pd.DataFrame:
    if not GRAPH_ARTICLE_LIMIT or len(df) <= GRAPH_ARTICLE_LIMIT:
        return df
    if "popularity" in df.columns:
        return df.sort_values("popularity", ascending=False).head(GRAPH_ARTICLE_LIMIT).reset_index(drop=True)
    return df.head(GRAPH_ARTICLE_LIMIT).reset_index(drop=True)


def _graph_cache_path_for_frame(graph_df: pd.DataFrame) -> str | None:
    if not STARTUP_MAX_ARTICLES and not GRAPH_ARTICLE_LIMIT:
        return None
    os.makedirs(GRAPH_DIR, exist_ok=True)
    return os.path.join(GRAPH_DIR, f"knowledge_graph.startup_{len(graph_df)}.pkl")


def _balanced_startup_indices(df: pd.DataFrame, limit: int) -> np.ndarray:
    if limit <= 0 or len(df) <= limit:
        return df.index.to_numpy()

    if "category" not in df.columns:
        return df.head(limit).index.to_numpy()

    working = df.copy()
    if "popularity" in working.columns:
        working = working.sort_values("popularity", ascending=False)

    grouped: dict[str, list[int]] = {}
    category_priority: list[tuple[str, float]] = []
    for category, group in working.groupby(working["category"].fillna("").astype(str).str.lower(), sort=False):
        rows = list(group.index)
        if not rows:
            continue
        grouped[category] = rows
        top_popularity = float(group["popularity"].iloc[0]) if "popularity" in group.columns else 0.0
        category_priority.append((category, top_popularity))

    ordered_categories = [category for category, _ in sorted(category_priority, key=lambda item: item[1], reverse=True)]
    if not ordered_categories:
        return working.head(limit).index.to_numpy()

    selected: list[int] = []
    cursors = {category: 0 for category in ordered_categories}

    while len(selected) < limit:
        progress = False
        for category in ordered_categories:
            rows = grouped[category]
            cursor = cursors[category]
            if cursor >= len(rows):
                continue
            selected.append(rows[cursor])
            cursors[category] += 1
            progress = True
            if len(selected) >= limit:
                break
        if not progress:
            break

    return np.asarray(selected, dtype=np.int64)


def _limit_loaded_assets(df: pd.DataFrame, embeddings: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    if not STARTUP_MAX_ARTICLES or len(df) <= STARTUP_MAX_ARTICLES:
        return df.reset_index(drop=True), embeddings

    selected_idx = _balanced_startup_indices(df, STARTUP_MAX_ARTICLES)

    limited_df = df.loc[selected_idx].reset_index(drop=True)
    limited_embeddings = np.asarray(embeddings[selected_idx], dtype=np.float32)
    category_mix = (
        limited_df["category"].fillna("").astype(str).str.lower().value_counts().head(6).to_dict()
        if "category" in limited_df.columns
        else {}
    )
    print(
        f"Trimmed startup assets to {len(limited_df):,} diverse articles via HYPERNEWS_MAX_ARTICLES. "
        f"Top categories: {category_mix}"
    )
    return limited_df, limited_embeddings


def get_or_create_user(user_id: str) -> UserProfile:
    if user_id not in USERS:
        stored = load_user(user_id)
        if stored:
            USERS[user_id] = UserProfile(
                user_id=user_id,
                interests=stored["interests"],
                reading_history=stored["reading_history"],
                avg_dwell_time=float(stored.get("avg_dwell_time", 0.0) or 0.0),
            )
        else:
            USERS[user_id] = UserProfile(user_id=user_id)

        auth_user = load_auth_user_by_id(user_id)
        _apply_auth_profile(USERS[user_id], auth_user)

        session = load_user_session(user_id)
        if session:
            USERS[user_id].recent_clicks  = session.get("recent_clicks", [])
            USERS[user_id].recent_skips   = session.get("recent_skips", [])
            USERS[user_id].recent_negative_actions = session.get("recent_negative_actions", [])
            USERS[user_id].session_topics = session.get("session_topics", [])
            USERS[user_id].recent_queries = session.get("recent_queries", [])
            USERS[user_id].recent_entities = session.get("recent_entities", [])
            USERS[user_id].recent_sources = session.get("recent_sources", [])
            USERS[user_id].mood           = session.get("mood", "neutral")
            USERS[user_id].avg_dwell_time = float(
                session.get("avg_dwell_time", USERS[user_id].avg_dwell_time) or 0.0
            )
            USERS[user_id].total_positive_interactions = session.get("total_positive_interactions", 0)
            USERS[user_id].interest_update_count       = session.get("interest_update_count", 0)

        _hydrate_user_recent_state(USERS[user_id])

    return USERS[user_id]


@app.on_event("startup")
async def startup_event():
    global BANDIT, DF, EMBEDDINGS, FAISS_INDEX, KG_GRAPH, MODEL, RERANKER, RERANKER_LOAD_ATTEMPTED, LTR_SCORER, VECTOR_BACKEND, SEARCH_BACKEND

    init_db()
    os.makedirs(os.path.join(_BASE, "models"), exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    parquet_path = _resolve_existing_path(PARQUET_FILES)
    faiss_path = _resolve_existing_path(FAISS_FILES)

    if not parquet_path or not os.path.exists(EMB_FILE):
        print(
            "Data files not found. Expected one of:\n"
            f"  parquet: {PARQUET_FILES}\n"
            f"  embeddings: {EMB_FILE}\n"
            f"  faiss: {FAISS_FILES}"
        )
        return

    try:
        DF = _ensure_article_frame(pd.read_parquet(parquet_path))
        EMBEDDINGS = _normalize_embedding_matrix(np.load(EMB_FILE).astype("float32"))
        if len(DF) != len(EMBEDDINGS):
            raise ValueError(
                f"Dataset and embedding count mismatch: {len(DF)} rows vs {len(EMBEDDINGS)} vectors"
            )
        DF, EMBEDDINGS = _limit_loaded_assets(DF, EMBEDDINGS)

        if faiss_path:
            FAISS_INDEX = load_vector_index(faiss_path)
            if FAISS_INDEX is not None:
                if FAISS_INDEX.ntotal != len(DF):
                    print(
                        f"Vector index size mismatch for {os.path.basename(faiss_path)} "
                        f"({FAISS_INDEX.ntotal} vs {len(DF)}). Rebuilding index."
                    )
                    FAISS_INDEX = None
                else:
                    print(
                        f"Loaded vector index from {os.path.basename(faiss_path)} "
                        f"({FAISS_INDEX.ntotal} vectors)"
                    )

        if FAISS_INDEX is None:
            FAISS_INDEX = build_faiss_index(EMBEDDINGS)
            if STARTUP_MAX_ARTICLES:
                print(f"Built in-memory vector index ({FAISS_INDEX.ntotal} vectors) for capped startup mode")
            else:
                if save_vector_index(FAISS_INDEX, DEFAULT_FAISS):
                    print(f"Built and saved vector index ({FAISS_INDEX.ntotal} vectors)")
                else:
                    print(f"Built in-memory vector index ({FAISS_INDEX.ntotal} vectors)")

        # context_dim = 384 embedding + 7 context scalars = 391
        # The 7th scalar is kg_score (knowledge graph affinity), so the bandit
        # learns jointly from RL rewards and graph structure.
        # Note: any existing bandit_model.pkl at a different dim will be recreated fresh.
        BANDIT = LinUCBBandit.load_or_create(
            BANDIT_FILE,
            context_dim=EMBEDDINGS.shape[1] + 7,
            alpha=0.15,
            epsilon=0.02,
        )

        if GRAPH_ENABLED:
            graph_df = _graph_frame(DF)
            KG_GRAPH = build_knowledge_graph(
                graph_df,
                cache_path=_graph_cache_path_for_frame(graph_df),
            )
        else:
            KG_GRAPH = None
        VECTOR_BACKEND = QdrantVectorStore(url=QDRANT_URL, enabled=QDRANT_ENABLED)
        SEARCH_BACKEND = OpenSearchArticleStore(url=OPENSEARCH_URL, enabled=OPENSEARCH_ENABLED)
        if VECTOR_BACKEND.available:
            vector_sync = VECTOR_BACKEND.sync_article_embeddings(DF, EMBEDDINGS)
            print(f"Qdrant sync: {vector_sync}")
        else:
            print(f"Qdrant unavailable; using FAISS fallback. {VECTOR_BACKEND.last_error if VECTOR_BACKEND else ''}")
        if SEARCH_BACKEND.available:
            search_sync = SEARCH_BACKEND.sync_articles(DF)
            print(f"OpenSearch sync: {search_sync}")
        else:
            print(f"OpenSearch unavailable; using lexical fallback. {SEARCH_BACKEND.last_error if SEARCH_BACKEND else ''}")
        prepare_search_cache(DF)
        MODEL = None
        RERANKER = None
        RERANKER_LOAD_ATTEMPTED = False
        LTR_SCORER = HybridLTRScorer()

        # Load Transformer encoder if saved weights exist
        try:
            from transformer_encoder import load_encoder, _MODEL_SAVE_PATH as _ENC_PATH

            enc_path = os.path.join(_BASE, _ENC_PATH)
            if os.path.exists(enc_path):
                if load_encoder(enc_path):
                    print(f"Transformer encoder loaded from {enc_path}")
                else:
                    print("Transformer encoder unavailable; using weighted-mean profile fallback")
            else:
                print("Transformer encoder: no saved weights found, falling back to weighted-mean profile")
        except Exception as exc:
            print(f"Transformer encoder unavailable ({exc}); using weighted-mean profile fallback")

        print(
            f"Backend ready: {len(DF):,} articles | "
            f"KG nodes: {KG_GRAPH.number_of_nodes() if KG_GRAPH else 0} | "
            f"Embeddings: {EMBEDDINGS.shape} | "
            f"Bandit context_dim: {BANDIT.context_dim} | "
            f"Reranker: {'on' if RERANKER is not None else 'fallback'} | "
            f"LTR: {'loaded' if LTR_SCORER and LTR_SCORER.model_loaded else 'fallback'} | "
            f"DB: {get_database_backend_info()['dialect']} | "
            f"Qdrant: {'on' if VECTOR_BACKEND and VECTOR_BACKEND.available else 'fallback'} | "
            f"OpenSearch: {'on' if SEARCH_BACKEND and SEARCH_BACKEND.available else 'fallback'}"
        )
    except Exception as exc:
        print(f"Startup failed while loading recommendation assets: {exc}")
        DF = pd.DataFrame()
        EMBEDDINGS = np.array([])
        FAISS_INDEX = None
        KG_GRAPH = None
        BANDIT = None
        MODEL = None
        RERANKER = None
        RERANKER_LOAD_ATTEMPTED = False
        LTR_SCORER = None
        VECTOR_BACKEND = None
        SEARCH_BACKEND = None


@app.on_event("shutdown")
async def shutdown_event():
    if BANDIT:
        BANDIT.save(BANDIT_FILE)
        print("Bandit state saved to disk.")
    try:
        from transformer_encoder import save_encoder, _MODEL_SAVE_PATH as _ENC_PATH
        if save_encoder(os.path.join(_BASE, _ENC_PATH)):
            print("Transformer encoder saved.")
    except ImportError:
        pass


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "articles_loaded": len(DF),
        "kg_nodes": KG_GRAPH.number_of_nodes() if KG_GRAPH else 0,
        "users_active": len(USERS),
        "database": get_database_backend_info(),
        "groq_enabled": bool(os.getenv("GROQ_API_KEY")),
        "redis_enabled": _redis_status(),
        "graph_enabled": GRAPH_ENABLED,
        "graph_article_limit": GRAPH_ARTICLE_LIMIT,
        "startup_max_articles": STARTUP_MAX_ARTICLES,
        "bandit_context_dim": BANDIT.context_dim if BANDIT else 0,
        "query_model_name": QUERY_MODEL_NAME,
        "reranker_configured": RERANKER_ENABLED,
        "reranker_enabled": bool(RERANKER is not None),
        "ltr_model_loaded": bool(LTR_SCORER and LTR_SCORER.model_loaded),
        "qdrant": VECTOR_BACKEND.status() if VECTOR_BACKEND else {},
        "opensearch": SEARCH_BACKEND.status() if SEARCH_BACKEND else {},
    }


@app.post("/auth/register")
async def auth_register(req: RegisterRequest):
    error = _validate_registration_payload(req.email, req.password)
    if error:
        return {"error": error}

    existing = load_auth_user_by_email(req.email)
    if existing:
        return {"error": "An account with that email already exists."}

    user_id = f"user_{uuid.uuid4().hex[:12]}"
    display_name = str(req.display_name or "").strip() or _normalize_email(req.email).split("@", 1)[0]
    onboarding_completed = _has_onboarding_answers(
        age_bucket=req.age_bucket,
        gender=req.gender,
        occupation=req.occupation,
        location_region=req.location_region,
        location_country=req.location_country,
        interest_text=req.interest_text,
        top_categories=req.top_categories,
        affect_consent=req.affect_consent,
    )
    bio_profile = _build_bio_profile(
        age_bucket=req.age_bucket,
        gender=req.gender,
        occupation=req.occupation,
        location_region=req.location_region,
        location_country=req.location_country,
        interest_text=req.interest_text,
        top_categories=req.top_categories,
        affect_consent=req.affect_consent,
    )
    auth_user = create_auth_user(
        user_id=user_id,
        email=_normalize_email(req.email),
        display_name=display_name,
        age_bucket=bio_profile["age_bucket"],
        gender=bio_profile["gender"],
        occupation=bio_profile["occupation"],
        location_region=bio_profile["location_region"],
        location_country=bio_profile["location_country"],
        interest_text=bio_profile["interest_text"],
        top_categories=bio_profile["top_categories"],
        affect_consent=bio_profile["affect_consent"],
        bio_embedding=bio_profile["bio_embedding"],
        bio_text_embedding=bio_profile["bio_text_embedding"],
        bio_embedding_version=bio_profile["bio_embedding_version"],
        onboarding_completed=onboarding_completed,
        password_hash=hash_password(req.password),
    )
    save_user(user_id, interests={}, reading_history=[], total_reads=0, avg_dwell_time=0.0)
    created_user = load_auth_user_by_id(user_id) or auth_user
    return {"status": "registered", "user": _auth_payload(created_user)}


@app.post("/auth/validate")
async def auth_validate(req: LoginRequest):
    auth_user = load_auth_user_by_email(req.email)
    if not auth_user or not verify_password(req.password, str(auth_user.get("password_hash") or "")):
        return {"error": "Invalid email or password."}
    return {"status": "ok", "user": _auth_payload(auth_user)}


@app.get("/auth/user/{user_id}")
async def auth_user(user_id: str):
    auth_user = load_auth_user_by_id(user_id)
    if not auth_user:
        return {"error": "User not found."}
    return {"user": _auth_payload(auth_user)}


@app.get("/articles")
async def get_articles(limit: int = 20):
    if len(DF) == 0:
        return {"articles": []}
    records = DF.sample(min(limit, len(DF))).to_dict("records")
    return {"articles": [sanitize_article(article) for article in records]}


@app.get("/graph")
async def graph_info():
    if KG_GRAPH is None:
        return {
            "error": "Graph not loaded",
            "graph_enabled": GRAPH_ENABLED,
            "graph_article_limit": GRAPH_ARTICLE_LIMIT,
            "startup_max_articles": STARTUP_MAX_ARTICLES,
        }
    return get_graph_stats(KG_GRAPH)


@app.get("/search/suggest")
async def search_suggest(q: str = Query(default=""), user_id: str = Query(default=""), limit: int = 10):
    recent_search_rows = list_recent_searches(user_id, limit=25) if user_id else []
    recent_queries = [
        str(row.get("normalized_query") or row.get("query_text") or "").strip()
        for row in recent_search_rows
        if str(row.get("normalized_query") or row.get("query_text") or "").strip()
    ]
    return {
        "query": q,
        "suggestions": suggest_queries(
            q,
            DF,
            recent_queries=recent_queries,
            limit=max(int(limit), 0),
            search_backend=SEARCH_BACKEND,
        ),
    }


@app.get("/admin/ranking-health")
async def admin_ranking_health():
    return {
        "bandit": {
            "enabled": BANDIT is not None,
            "context_dim": BANDIT.context_dim if BANDIT else 0,
            "alpha": getattr(BANDIT, "alpha", 0.0),
            "epsilon": getattr(BANDIT, "epsilon", 0.0),
            "total_updates": getattr(BANDIT, "total_updates", 0),
        },
        "retrieval": {
            "query_model": QUERY_MODEL_NAME,
            "query_model_loaded": MODEL is not None,
            "reranker_configured": RERANKER_ENABLED,
            "reranker_loaded": RERANKER is not None,
            "faiss_index_loaded": FAISS_INDEX is not None,
            "qdrant": VECTOR_BACKEND.status() if VECTOR_BACKEND else {},
            "opensearch": SEARCH_BACKEND.status() if SEARCH_BACKEND else {},
        },
        "ltr": {
            "loaded": bool(LTR_SCORER and LTR_SCORER.model_loaded),
            "model_path": getattr(LTR_SCORER, "model_path", ""),
            "training_rows_available": len(fetch_ltr_training_rows(limit=5000)),
        },
        "database": get_database_backend_info(),
        "state_windows": {
            "recent_clicks": 25,
            "recent_negative_actions": 25,
            "recent_queries": 10,
        },
    }


@app.get("/profile/{user_id}")
async def get_profile(user_id: str):
    user = get_or_create_user(user_id)
    return _profile_payload(user)


@app.get("/me/profile")
async def get_me_profile(user_id: str = Query(default="")):
    if not user_id:
        return {"error": "user_id query parameter is required in dev mode"}
    user = get_or_create_user(user_id)
    return _profile_payload(user)


@app.post("/me/profile")
async def update_me_profile(req: ProfileUpdateRequest, user_id: str = Query(default="")):
    if not user_id:
        return {"error": "user_id query parameter is required in dev mode"}

    auth_user = load_auth_user_by_id(user_id)
    if not auth_user:
        return {"error": "User not found."}

    display_name = str(req.display_name or auth_user.get("display_name") or "").strip()
    if not display_name:
        display_name = _normalize_email(str(auth_user.get("email") or "")).split("@", 1)[0]

    bio_profile = _build_bio_profile(
        age_bucket=req.age_bucket,
        gender=req.gender,
        occupation=req.occupation,
        location_region=req.location_region,
        location_country=req.location_country,
        interest_text=req.interest_text,
        top_categories=req.top_categories,
        affect_consent=req.affect_consent,
    )
    updated = update_auth_user_profile(
        user_id=user_id,
        display_name=display_name,
        age_bucket=bio_profile["age_bucket"],
        gender=bio_profile["gender"],
        occupation=bio_profile["occupation"],
        location_region=bio_profile["location_region"],
        location_country=bio_profile["location_country"],
        interest_text=bio_profile["interest_text"],
        top_categories=bio_profile["top_categories"],
        affect_consent=bio_profile["affect_consent"],
        bio_embedding=bio_profile["bio_embedding"],
        bio_text_embedding=bio_profile["bio_text_embedding"],
        bio_embedding_version=bio_profile["bio_embedding_version"],
    )
    user = get_or_create_user(user_id)
    _apply_auth_profile(user, updated)
    return {
        "status": "updated",
        "user": _auth_payload(updated or {}),
        "profile": _profile_payload(user),
    }


@app.post("/me/onboarding")
async def complete_me_onboarding(req: OnboardingRequest, user_id: str = Query(default="")):
    if not user_id:
        return {"error": "user_id query parameter is required in dev mode"}

    auth_user = load_auth_user_by_id(user_id)
    if not auth_user:
        return {"error": "User not found."}

    display_name = str(auth_user.get("display_name") or "").strip()
    if not display_name:
        display_name = _normalize_email(str(auth_user.get("email") or "")).split("@", 1)[0]

    bio_profile = _build_bio_profile(
        age_bucket=req.age_bucket,
        gender=req.gender,
        occupation=req.occupation,
        location_region=req.location_region,
        location_country=req.location_country,
        interest_text=req.interest_text,
        top_categories=req.top_categories,
        affect_consent=req.affect_consent,
    )
    updated = complete_auth_user_onboarding(
        user_id=user_id,
        display_name=display_name,
        age_bucket=bio_profile["age_bucket"],
        gender=bio_profile["gender"],
        occupation=bio_profile["occupation"],
        location_region=bio_profile["location_region"],
        location_country=bio_profile["location_country"],
        interest_text=bio_profile["interest_text"],
        top_categories=bio_profile["top_categories"],
        affect_consent=bio_profile["affect_consent"],
        bio_embedding=bio_profile["bio_embedding"],
        bio_text_embedding=bio_profile["bio_text_embedding"],
        bio_embedding_version=bio_profile["bio_embedding_version"],
    )
    user = get_or_create_user(user_id)
    _apply_auth_profile(user, updated)
    return {
        "status": "completed",
        "user": _auth_payload(updated or {}),
        "profile": _profile_payload(user),
    }


@app.get("/me/history")
async def get_me_history(user_id: str = Query(default=""), limit: int = 25):
    if not user_id:
        return {"error": "user_id query parameter is required in dev mode"}
    user = get_or_create_user(user_id)
    return {
        "user_id": user.user_id,
        "recent_clicks": user.recent_clicks[-max(int(limit), 0):],
        "recent_negative_actions": getattr(user, "recent_negative_actions", [])[-max(int(limit), 0):],
        "reading_history": user.reading_history[-max(int(limit), 0):],
        "feedback": list_recent_feedback(user.user_id, limit=max(int(limit), 0)),
    }


@app.get("/me/searches")
async def get_me_searches(user_id: str = Query(default=""), limit: int = 25):
    if not user_id:
        return {"error": "user_id query parameter is required in dev mode"}
    return {
        "user_id": user_id,
        "searches": list_recent_searches(user_id, limit=max(int(limit), 0)),
    }


@app.post("/recommend")
async def recommend(req: RecommendRequest):
    global MODEL, MODEL_LOAD_ATTEMPTED, RERANKER, RERANKER_LOAD_ATTEMPTED, LTR_SCORER

    if len(DF) == 0:
        return {"error": "Data not loaded. Run generate_data.py first."}

    request_started_at = time.perf_counter()
    stage_timings = {
        "retrieval_ms": 0.0,
        "ranking_ms": 0.0,
        "explanation_ms": 0.0,
    }
    search_stack_used = False
    req.surface = _normalize_surface(req.surface)
    req.query = _normalize_query_text(req.query)
    if req.surface == "feed":
        req.query = None
    elif req.query is None:
        req.surface = "feed"

    user = get_or_create_user(req.user_id)
    user.mood = req.mood
    user.time_of_day = get_time_of_day()
    excluded_ids = _normalize_article_ids(req.exclude_ids)
    request_id = str(req.request_id or uuid.uuid4().hex[:12])
    session_id = str(req.session_id or req.user_id)
    query_intent: dict | None = None
    candidate_source_distribution: Counter[str] = Counter()

    n_positive = user.total_positive_interactions
    news_id_to_idx = _build_news_id_to_idx(DF)
    memory_records: list[dict] = []
    memory_bonus_map: dict[str, float] = {}
    use_cold_start = (
        req.surface == "feed"
        and req.query is None
        and
        n_positive < _SEMI_COLD_START_THRESHOLD
    )
    if use_cold_start:
        retrieval_started_at = time.perf_counter()
        articles = cold_start_recommendations(
            user,
            DF,
            EMBEDDINGS,
            req.n,
            excluded_ids=excluded_ids,
        )
        stage_timings["retrieval_ms"] = round((time.perf_counter() - retrieval_started_at) * 1000.0, 2)
        candidate_source_distribution = Counter(
            {"cold_start": len(articles)}
        )
        mode = "cold_start"
    elif req.surface == "search" and req.query:
        search_stack_used = True
        if MODEL is None and not MODEL_LOAD_ATTEMPTED:
            MODEL = _load_sentence_transformer_model(QUERY_MODEL_NAME)
            MODEL_LOAD_ATTEMPTED = True
        if RERANKER_ENABLED and RERANKER is None and not RERANKER_LOAD_ATTEMPTED:
            RERANKER = _load_cross_encoder_model(RERANKER_MODEL_NAME)
            RERANKER_LOAD_ATTEMPTED = True

        retrieval_started_at = time.perf_counter()
        raw, diagnostics = build_hybrid_candidates(
            user,
            req.query,
            DF,
            FAISS_INDEX,
            MODEL,
            KG_GRAPH,
            news_id_to_idx,
            reranker=RERANKER,
            search_backend=SEARCH_BACKEND,
            vector_backend=VECTOR_BACKEND,
            limit=min(max(req.n * 8 + (len(excluded_ids) * 2), 48), len(DF)),
            rerank_k=min(max(req.n * 4, 24), 48),
        )
        stage_timings["retrieval_ms"] = round((time.perf_counter() - retrieval_started_at) * 1000.0, 2)
        blocked_ids = (
            set(user.reading_history)
            | set(getattr(user, "recent_negative_actions", [])[-25:])
            | set(getattr(user, "recent_skips", [])[-25:])
            | excluded_ids
        )
        raw = [article for article in raw if str(article.get("news_id", "")) not in blocked_ids]
        query_intent = diagnostics.get("query_intent", {})
        log_search_event(
            req.user_id,
            req.query,
            session_id=session_id,
            normalized_query=str(query_intent.get("normalized_query", "")),
            intent=query_intent,
        )
        candidate_source_distribution = _candidate_source_distribution(raw)

        ranking_started_at = time.perf_counter()
        articles = rank_search_articles(
            user,
            raw,
            EMBEDDINGS,
            DF,
            query_intent=query_intent,
            n=req.n,
        )
        stage_timings["ranking_ms"] = round((time.perf_counter() - ranking_started_at) * 1000.0, 2)
        mode = "rag"
    else:
        assert req.surface == "feed"
        assert req.query is None
        assert not search_stack_used

        retrieval_started_at = time.perf_counter()
        memory_records, memory_bonus_map = build_long_term_memory_signal(
            user,
            DF,
            EMBEDDINGS,
            news_id_to_idx,
            faiss_index=FAISS_INDEX,
            graph=KG_GRAPH,
            max_candidates=max(req.n * 4 + len(excluded_ids), 24),
        )
        candidates = build_candidate_pool(
            user,
            DF,
            EMBEDDINGS,
            FAISS_INDEX,
            graph=KG_GRAPH,
            max_candidates=min(max(req.n * 25 + (len(excluded_ids) * 3), 120), len(DF)),
            memory_records=memory_records,
            excluded_ids=excluded_ids,
        )
        stage_timings["retrieval_ms"] = round((time.perf_counter() - retrieval_started_at) * 1000.0, 2)
        candidate_source_distribution = _candidate_source_distribution(candidates)
        ranking_started_at = time.perf_counter()
        articles = rank_articles(
            user,
            candidates,
            EMBEDDINGS,
            BANDIT,
            DF,
            KG_GRAPH,
            n=req.n,
            memory_bonus_map=memory_bonus_map,
            ltr_scorer=LTR_SCORER,
            query_intent=query_intent,
            explore_focus=req.explore_focus,
        )
        stage_timings["ranking_ms"] = round((time.perf_counter() - ranking_started_at) * 1000.0, 2)
        if not articles:
            articles = cold_start_recommendations(user, DF, req.n, excluded_ids=excluded_ids)
            candidate_source_distribution = Counter({"cold_start": len(articles)})
        mode = "rl"

    if req.surface == "feed":
        assert not search_stack_used
        assert query_intent is None

    user._last_candidate_pool = articles

    explanation_started_at = time.perf_counter()
    explanation = generate_personalized_summary(
        {
            "mood": user.mood,
            "time_of_day": user.time_of_day,
            "mode": mode,
            "query": req.query,
            "recent_topics": user.session_topics[-5:],
            "top_categories": list(user.top_categories),
            "interest_text": user.interest_text,
            "location_region": user.location_region,
            "location_country": getattr(user, "location_country", ""),
            "profile_hint_categories": list(
                infer_interest_category_weights(user.interest_text, limit=4).keys()
            ),
        },
        articles,
    )
    stage_timings["explanation_ms"] = round((time.perf_counter() - explanation_started_at) * 1000.0, 2)

    update_user_session(user)
    _persist_user_recent_state(user)
    log_recommendation_event(
        req.user_id,
        session_id=session_id,
        request_id=request_id,
        surface=req.surface,
        mood=req.mood,
        mode=mode,
        query_text=str(req.query or ""),
        candidate_sources=candidate_source_distribution or Counter(str(article.get("candidate_source", "ranker")) for article in articles),
        impression_ids=[str(article.get("news_id", "")) for article in articles],
    )
    log_ranking_feature_rows(
        req.user_id,
        session_id=session_id,
        request_id=request_id,
        surface=req.surface,
        query_text=str(req.query or ""),
        rows=[
            {
                "impression_id": str(article.get("news_id", "")),
                "article_id": str(article.get("news_id", "")),
                "candidate_source": str(article.get("candidate_source", "")),
                "position": position,
                "features": dict(article.get("score_breakdown", {})) if isinstance(article.get("score_breakdown", {}), dict) else {},
            }
            for position, article in enumerate(articles)
        ],
    )
    total_latency_ms = round((time.perf_counter() - request_started_at) * 1000.0, 2)
    if req.surface == "search":
        _log_structured(
            "recommend_latency",
            surface=req.surface,
            mode=mode,
            search_latency_ms=total_latency_ms,
            candidate_source_distribution=dict(candidate_source_distribution),
            **stage_timings,
        )
    else:
        _log_structured(
            "recommend_latency",
            surface=req.surface,
            mode=mode,
            feed_latency_ms=total_latency_ms,
            candidate_source_distribution=dict(candidate_source_distribution),
            **stage_timings,
        )

    return {
        "articles": [sanitize_article(article) for article in articles],
        "explanation": explanation,
        "user_id": req.user_id,
        "mode": mode,
        "request_id": request_id,
        "query_intent": query_intent or {},
    }


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    user = get_or_create_user(req.user_id)
    dwell_time = _clip_dwell_time_seconds(req.dwell_time)
    reward = _compute_feedback_reward(req.action, dwell_time, user.avg_dwell_time)

    try:
        match = DF[DF["news_id"] == req.article_id]
        if len(match) == 0:
            return {"status": "article_not_found"}

        idx = int(match.index[0])
        article_embedding = EMBEDDINGS[idx]

        # Build kg_score for this article so the context vector fed to the bandit
        # includes graph affinity — RL and KG learn together.
        news_id_to_idx_fb = _build_news_id_to_idx(DF)
        graph_bonus_fb = _graph_bonus_map(user, KG_GRAPH)
        article_kg_score = float(graph_bonus_fb.get(req.article_id, 0.0))

        context_vector = build_context_vector(user, article_embedding, kg_score=article_kg_score)

        if BANDIT:
            # KG-augmented reward: if the article has strong graph affinity we reinforce
            # that signal by slightly boosting the reward the bandit trains on.
            # This teaches the bandit that "KG-connected + clicked" is especially good.
            kg_reward_boost = 0.15 * article_kg_score if reward > 0 else 0.0
            augmented_reward = float(np.clip(reward + kg_reward_boost, -1.0, 2.5))
            BANDIT.update(req.article_id, context_vector, augmented_reward)

            # Update category-level stats for all actions (incl. skip)
            category_str = str(match.iloc[0]["category"])
            subcategory_str = str(match.iloc[0].get("subcategory", "") or "")
            BANDIT._update_category_stats(category_str, BANDIT._normalize_reward(augmented_reward))

            # Propagate positive reward to category siblings from last pool
            if reward > 0:
                last_pool = list(getattr(user, "_last_candidate_pool", []))
                sibling_ids = _select_visible_sibling_ids(
                    req.article_id,
                    category_str,
                    subcategory_str,
                    last_pool,
                )
                if sibling_ids:
                    BANDIT.propagate_category_reward(
                        category_str,
                        sibling_ids,
                        reward,
                        decay=_CATEGORY_PROPAGATION_DECAY,
                    )

                # KG reward propagation: also update bandit stats for KG-linked articles.
                # This is the key RL+KG joint learning: clicking article A tells the bandit
                # that graph-neighbours of A are also likely to be interesting.
                if KG_GRAPH is not None:
                    kg_linked_ids = get_kg_related_ids(
                        req.article_id, KG_GRAPH, news_id_to_idx_fb, limit=20
                    )
                    if kg_linked_ids:
                        BANDIT.propagate_category_reward(
                            category_str,
                            kg_linked_ids,
                            reward,
                            decay=_KG_PROPAGATION_DECAY,
                        )

        row = match.iloc[0]
        category = row["category"]
        source = str(row.get("source", "") or "")
        entity_labels = row.get("entity_labels", [])
        if not isinstance(entity_labels, list):
            entity_labels = []

        if req.action in ("click", "read_full", "save", "more_like_this"):
            if req.article_id not in user.reading_history:
                user.reading_history.append(req.article_id)
            update_user_state(
                user,
                article_id=req.article_id,
                category=str(category),
                entities=entity_labels[:5],
                source=source,
                negative=False,
            )
            user.recent_skips = [nid for nid in user.recent_skips if nid != req.article_id]
            user.recent_negative_actions = [nid for nid in getattr(user, "recent_negative_actions", []) if nid != req.article_id]

            nudge = abs(reward) * _INTEREST_NUDGE_FACTOR
            user.interests[category] = user.interests.get(category, 0.0) + nudge
            user.total_positive_interactions += 1
            _update_avg_dwell_time(user, dwell_time)

            push_recent_history(req.user_id, req.article_id)

            try:
                save_reading_vector(req.user_id, req.article_id, article_embedding, feedback_weight=nudge)
            except Exception as vec_err:
                print(f"Vector history write failed (non-fatal): {vec_err}")
            if VECTOR_BACKEND is not None:
                VECTOR_BACKEND.upsert_user_memory(
                    user_id=req.user_id,
                    article_id=req.article_id,
                    vector=article_embedding,
                    category=str(category),
                    source=source,
                    feedback_weight=float(max(nudge, 0.25)),
                )

        elif req.action in {"skip", "not_interested", "less_from_source"}:
            user.recent_skips.append(req.article_id)
            user.recent_skips = user.recent_skips[-25:]
            update_user_state(
                user,
                article_id=req.article_id,
                category=str(category),
                entities=entity_labels[:5],
                source=source,
                negative=True,
            )

            current = user.interests.get(category, 0.0)
            if current > 0:
                user.interests[category] = max(0.0, current * (1.0 - _SKIP_DECAY_FACTOR) - _SKIP_FIXED_PENALTY)
            if req.action == "less_from_source" and source:
                source_key = f"source::{source}"
                current_source = user.interests.get(source_key, 0.0)
                user.interests[source_key] = max(0.0, current_source * 0.75)

        _apply_interest_ema_decay(user)
        user.interest_update_count += 1

        save_user(
            user.user_id,
            user.interests,
            user.reading_history,
            len(user.reading_history),
            avg_dwell_time=user.avg_dwell_time,
        )
        update_user_session(user)
        _persist_user_recent_state(user)
        log_feedback_event(
            req.user_id,
            req.article_id,
            req.action,
            session_id=req.session_id or req.user_id,
            request_id=req.request_id,
            impression_id=req.impression_id or req.article_id,
            position=req.position,
            dwell_time=dwell_time,
            query_text=req.query_text,
            source_feedback=req.source_feedback,
        )
        update_ranking_feature_label(
            req.user_id,
            request_id=req.request_id,
            article_id=req.article_id,
            action=req.action,
            dwell_time=dwell_time,
        )
    except Exception as exc:
        print(f"Error in /feedback: {exc}")

    return {"status": "updated", "reward": reward}


@app.post("/reset/{user_id}")
async def reset_user(user_id: str):
    USERS.pop(user_id, None)
    clear_user_session(user_id)
    delete_user(user_id)
    return {"status": "reset", "user_id": user_id}
