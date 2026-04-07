import numpy as np
import pandas as pd
from collections import Counter

try:
    from backend.bio_categories import BIO_CATEGORY_ORDER
    from backend.coldstart_hints import (
        extract_interest_terms,
        humanize_location,
        infer_interest_category_weights,
        infer_occupation_category_weights,
        location_terms,
    )
    from backend.db import query_similar_history
    from backend.graph import get_article_entities, get_related_articles
    from backend.mind_data import parse_entity_list
    from backend.rag_pipeline import encode_query_embedding, retrieve_articles
    from backend.user_profile import compute_context_score
except ImportError:
    from bio_categories import BIO_CATEGORY_ORDER
    from coldstart_hints import (
        extract_interest_terms,
        humanize_location,
        infer_interest_category_weights,
        infer_occupation_category_weights,
        location_terms,
    )
    from db import query_similar_history
    from graph import get_article_entities, get_related_articles
    from mind_data import parse_entity_list
    from rag_pipeline import encode_query_embedding, retrieve_articles
    from user_profile import compute_context_score

# ── Diversity / exploration policy ────────────────────────────────────────────
_DIVERSITY_MAX_FRACTION      = 0.35   # kept for reference (MMR replaces hard quota)
_CATEGORY_LOCK_THRESHOLD     = 4      # confirmed reads before category earns dominance override in MMR
_MMR_LAMBDA                  = 0.82   # relevance vs diversity tradeoff (λ close to 1 → more relevance)
_INTEREST_WARMUP_INTERACTIONS = 5     # positive interactions before interest_score reaches full weight
_INTEREST_DOMINANCE_THRESHOLD = 0.50  # concentration above which entropy penalty fires
_INTEREST_EMA_ALPHA           = 0.05  # per-feedback EMA decay rate on all interest weights
_CAT_BALANCE_MIN_INTERACTIONS = 3     # interactions before category-balanced FAISS kicks in
_MEMORY_BONUS_WEIGHT          = 0.10
_DISLIKED_CATEGORY_THRESHOLD  = 2.25  # requires repeated recent skips before hiding a category
_SKIP_CATEGORY_PENALTY_WEIGHT = 0.12
_SKIP_ENTITY_PENALTY_WEIGHT   = 0.05


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr
    return arr / norm


def _normalize_key(value: str) -> str:
    return str(value or "").strip().lower()


def _vector_or_none(value) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return None
    return arr / norm


def _interest_terms_from_text(value: str) -> list[str]:
    return extract_interest_terms(value, limit=10)


def _decode_bio_category_prior(user) -> dict[str, float]:
    raw = np.asarray(getattr(user, "bio_embedding", []), dtype=np.float32).reshape(-1)
    if raw.size < len(BIO_CATEGORY_ORDER):
        return {}

    category_slice = raw[:len(BIO_CATEGORY_ORDER)].astype(np.float32)
    category_slice -= float(category_slice.min())
    peak = float(category_slice.max())
    if peak <= 0.0:
        return {}

    normalized = category_slice / peak
    return {
        str(category): float(score)
        for category, score in zip(BIO_CATEGORY_ORDER, normalized)
        if float(score) >= 0.08
    }


def _preferred_onboarding_categories(user, bio_category_prior: dict[str, float], limit: int = 5) -> list[str]:
    preferred: list[str] = []
    for value in getattr(user, "top_categories", []):
        category_key = _normalize_key(value)
        if category_key and category_key not in preferred:
            preferred.append(category_key)

    for category, _ in sorted(
        bio_category_prior.items(),
        key=lambda item: float(item[1]),
        reverse=True,
    ):
        category_key = _normalize_key(category)
        if category_key and category_key not in preferred:
            preferred.append(category_key)
        if len(preferred) >= limit:
            break

    return preferred[:limit]


def _selected_category_weights(user) -> dict[str, float]:
    weights: dict[str, float] = {}
    selected_categories = [
        _normalize_key(value)
        for value in getattr(user, "top_categories", [])
        if _normalize_key(value)
    ]
    for index, category_key in enumerate(selected_categories):
        weight = max(0.35, 1.0 - (index * 0.12))
        weights[category_key] = max(weights.get(category_key, 0.0), float(weight))
    return weights


def _cold_start_category_preferences(user, bio_category_prior: dict[str, float]) -> tuple[dict[str, float], list[str]]:
    merged: Counter[str] = Counter()

    for category_key, weight in _selected_category_weights(user).items():
        merged[category_key] += 1.30 * float(weight)

    for category_key, weight in infer_interest_category_weights(
        str(getattr(user, "interest_text", "") or ""),
        limit=6,
    ).items():
        merged[_normalize_key(category_key)] += 1.45 * float(weight)

    for category_key, weight in infer_occupation_category_weights(
        str(getattr(user, "occupation", "") or ""),
    ).items():
        merged[_normalize_key(category_key)] += 0.60 * float(weight)

    for category_key, weight in bio_category_prior.items():
        merged[_normalize_key(category_key)] += 0.90 * float(weight)

    if str(getattr(user, "location_country", "") or "").strip():
        merged["news"] += 0.35
        merged["travel"] += 0.22
        merged["politics"] += 0.18
    elif str(getattr(user, "location_region", "") or "").strip():
        merged["news"] += 0.18
        merged["travel"] += 0.10

    if not merged:
        return {}, []

    peak = max(float(score) for score in merged.values())
    normalized = {
        category_key: float(score / max(peak, 1.0))
        for category_key, score in merged.items()
        if float(score) > 0.0
    }
    ordered = [
        category_key
        for category_key, _ in sorted(
            normalized.items(),
            key=lambda item: float(item[1]),
            reverse=True,
        )
    ]
    return normalized, ordered


def _location_match_score(article: dict, user) -> tuple[float, list[str]]:
    terms = location_terms(
        str(getattr(user, "location_region", "") or ""),
        str(getattr(user, "location_country", "") or ""),
    )
    if not terms:
        return 0.0, []

    article_text = " ".join(
        [
            str(article.get("title", "") or ""),
            str(article.get("abstract", "") or ""),
            str(article.get("source", "") or ""),
            str(article.get("category", "") or ""),
            str(article.get("subcategory", "") or ""),
            " ".join(_extract_entity_keys(_entity_value(article))),
        ]
    ).lower()

    matched_terms: list[str] = []
    for term in terms:
        normalized_term = _normalize_key(term)
        if normalized_term and normalized_term in article_text and normalized_term not in matched_terms:
            matched_terms.append(normalized_term)
        if len(matched_terms) >= 3:
            break

    if not matched_terms:
        return 0.0, []

    phrase_bonus = 0.18 if str(getattr(user, "location_country", "") or "").strip().lower() in article_text else 0.0
    score = min(1.0, (0.26 * len(matched_terms)) + phrase_bonus)
    return float(score), matched_terms[:2]


def _cold_start_reason_labels(
    article: dict,
    user,
    *,
    category_preference_score: float,
    inferred_interest_categories: dict[str, float],
    matched_terms: list[str],
    location_matches: list[str],
    bio_affinity: float,
) -> list[str]:
    reasons: list[str] = []
    category_key = _normalize_key(article.get("category", ""))
    subcategory_key = _normalize_key(article.get("subcategory", ""))
    selected_categories = {
        _normalize_key(value)
        for value in getattr(user, "top_categories", [])
        if _normalize_key(value)
    }

    if category_key in selected_categories or subcategory_key in selected_categories:
        reasons.append(f"Picked from your selected {category_key or subcategory_key} interests")
    elif category_key in inferred_interest_categories or subcategory_key in inferred_interest_categories:
        reasons.append(f"Your notes point toward {category_key or subcategory_key} stories")
    elif category_preference_score >= 0.55:
        reasons.append("Aligned with your onboarding profile")

    if matched_terms:
        reasons.append(f"Matches your notes: {', '.join(matched_terms[:2])}")

    readable_location = humanize_location(
        str(getattr(user, "location_region", "") or ""),
        str(getattr(user, "location_country", "") or ""),
    )
    if readable_location and location_matches:
        reasons.append(f"Relevant to {readable_location}")

    if bio_affinity >= 0.58:
        reasons.append("Close to the interests you described")

    if not reasons:
        reasons.append(
            f"Fits your {str(getattr(user, 'mood', 'neutral') or 'neutral')} {str(getattr(user, 'time_of_day', 'day') or 'day')} context"
        )

    return list(dict.fromkeys(reasons))[:4]


def _append_candidate_source(article: dict, source: str) -> dict:
    record = dict(article)
    existing = str(record.get("candidate_source", "") or "").strip()
    if not existing:
        record["candidate_source"] = source
    elif source and source not in existing.split("+"):
        record["candidate_source"] = f"{existing}+{source}"
    return record


def _annotate_records(records: list[dict], source: str) -> list[dict]:
    return [_append_candidate_source(article, source) for article in records]


def _build_news_id_to_idx(df: pd.DataFrame) -> dict:
    return pd.Series(df.index, index=df["news_id"]).to_dict()


def _extract_entity_keys(value) -> list[str]:
    if isinstance(value, list) and value and isinstance(value[0], str):
        return [str(item).strip() for item in value if str(item).strip()]

    entities = parse_entity_list(value)
    keys = []
    for entity in entities:
        key = entity.get("wikidata_id") or entity.get("id") or entity.get("label")
        if key:
            keys.append(str(key).strip())
    return keys


def _entity_value(record) -> object:
    entity_ids = record.get("entity_ids")
    if entity_ids is not None:
        try:
            if len(entity_ids) > 0:
                return entity_ids
        except TypeError:
            return entity_ids
    entity_labels = record.get("entity_labels")
    if entity_labels is not None:
        try:
            if len(entity_labels) > 0:
                return entity_labels
        except TypeError:
            return entity_labels
    return record.get("entities")


def build_user_profile_vector(user, article_embeddings: np.ndarray, news_id_to_idx: dict) -> np.ndarray | None:
    """Build a 384-dim user state vector.

    Tries the Transformer encoder first (seq-aware CLS pooling).
    Falls back to weighted mean of recent click embeddings if the encoder
    is unavailable or has too few history items.
    """
    try:
        from transformer_encoder import encode_user_history
        history_ids = list(user.recent_clicks[-25:]) + list(user.reading_history[-25:])
        vec = encode_user_history(history_ids, article_embeddings, news_id_to_idx)
        if vec is not None:
            return vec
    except ImportError:
        pass

    # Fallback: weighted mean (recent clicks counted twice for recency bias)
    weighted_ids = list(user.recent_clicks[-12:]) + list(user.recent_clicks[-12:]) + list(user.reading_history[-25:])
    vectors = []
    for news_id in weighted_ids:
        idx = news_id_to_idx.get(news_id)
        if idx is None:
            continue
        vectors.append(article_embeddings[idx])
    if not vectors:
        return None
    return _l2_normalize(np.mean(np.vstack(vectors), axis=0))


def _score_interest(user, category: str) -> float:
    """Normalised interest score with entropy penalty and warm-up damp.

    Bug 1 fix: if one category holds >50% of total weight (concentration),
    its score is penalized by (1 - concentration) so 6 Sports clicks can't
    produce an 0.88 interest score that drowns everything else.
    """
    positive_weights = {
        _normalize_key(cat): max(float(weight), 0.0)
        for cat, weight in user.interests.items()
    }
    total_weight = sum(positive_weights.values())
    if total_weight <= 0.0:
        return 0.0

    raw = positive_weights.get(_normalize_key(category), 0.0) / total_weight

    # Entropy / concentration penalty
    max_weight = max(positive_weights.values())
    concentration = max_weight / total_weight
    if concentration > _INTEREST_DOMINANCE_THRESHOLD:
        dominant_cat = _normalize_key(max(positive_weights, key=lambda c: positive_weights[c]))
        if _normalize_key(category) == dominant_cat:
            # e.g. concentration=0.75 → raw *= 0.25 → Sports drops from 0.75 to ~0.19
            raw = raw * (1.0 - concentration)

    # Linear warm-up damp: full weight only after _INTEREST_WARMUP_INTERACTIONS
    n = getattr(user, "total_positive_interactions", len(user.reading_history))
    if n < _INTEREST_WARMUP_INTERACTIONS:
        raw = raw * (n / _INTEREST_WARMUP_INTERACTIONS)

    return raw


def _apply_interest_ema_decay(user) -> None:
    """Decay all interest weights by _INTEREST_EMA_ALPHA on every feedback event.

    Prevents long-dormant categories from keeping accumulated weight forever.
    Called from app.py's /feedback handler.
    """
    for cat in list(user.interests.keys()):
        user.interests[cat] = user.interests[cat] * (1.0 - _INTEREST_EMA_ALPHA)
        if user.interests[cat] < 0.01:
            user.interests[cat] = 0.0


def _score_popularity(article: dict) -> float:
    popularity = article.get("popularity")
    if popularity is None:
        return 0.0
    try:
        return float(np.clip(float(popularity), 0.0, 1.0))
    except (TypeError, ValueError):
        return 0.0


def build_context_vector(user, article_embedding: np.ndarray, kg_score: float = 0.0) -> np.ndarray:
    """Build 391-dim context vector: 384-dim article embedding + 7 scalars.

    Scalars:
      1: mood (normalised 0-1)
      2: time of day (normalised 0-1)
      3: click count (normalised 0-1)
      4: category entropy (normalised Shannon entropy over interest weights)
      5: recent skip ratio (fraction of last-10 interactions that were skips)
      6: diversity hunger (unique categories in last-10 session topics / 10)
      7: kg_score — knowledge graph affinity [0,1] for this article.
         Passing this into the bandit lets it learn that KG-connected articles
         are better candidates, so RL and KG work together rather than in silos.
    """
    mood_map = {"neutral": 0, "happy": 1, "curious": 2, "stressed": 3, "tired": 4}
    time_map = {"morning": 0, "afternoon": 1, "evening": 2, "night": 3}

    mood_val        = mood_map.get(user.mood, 0) / 4.0
    time_val        = time_map.get(user.time_of_day, 0) / 3.0
    click_count_val = min(len(user.recent_clicks) / 25.0, 1.0)

    # Category entropy
    positive = {k: max(v, 0.0) for k, v in user.interests.items() if v > 0}
    total_w = sum(positive.values())
    if total_w > 0 and len(positive) > 1:
        probs = np.array(list(positive.values()), dtype=np.float32) / total_w
        raw_e = -float(np.sum(probs * np.log(probs + 1e-9)))
        cat_entropy = raw_e / max(float(np.log(len(positive))), 1e-9)
    else:
        cat_entropy = 0.0

    # Recent skip ratio
    recent_negatives = list(getattr(user, "recent_negative_actions", [])[-25:]) or list(user.recent_skips[-25:])
    n_skips = min(len(recent_negatives), 25)
    total_recent = min(len(user.recent_clicks) + len(recent_negatives), 25)
    skip_ratio = n_skips / max(total_recent, 1)

    # Diversity hunger
    last_topics = list(user.session_topics[-25:]) if hasattr(user, "session_topics") else []
    diversity_hunger = len(set(last_topics)) / 25.0

    context_signal = np.array(
        [mood_val, time_val, click_count_val, cat_entropy, skip_ratio, diversity_hunger,
         float(np.clip(kg_score, 0.0, 1.0))],
        dtype=np.float32,
    )

    article_vec = _l2_normalize(article_embedding.astype(np.float32))
    context_vec = np.concatenate([article_vec, context_signal])
    return _l2_normalize(context_vec)


def _build_history_profiles(user, df: pd.DataFrame, news_id_to_idx: dict, graph=None) -> tuple[Counter, Counter]:
    subcategory_weights: Counter = Counter()
    entity_weights: Counter = Counter()

    history_ids = list(user.recent_clicks[-12:]) + list(user.reading_history[-25:])
    weighted_history = list(enumerate(reversed(history_ids)))
    for offset, news_id in weighted_history:
        idx = news_id_to_idx.get(news_id)
        if idx is None:
            continue

        weight = 1.0 / (1.0 + (offset * 0.2))
        row = df.iloc[idx]
        subcategory = _normalize_key(row.get("subcategory", ""))
        if subcategory:
            subcategory_weights[subcategory] += weight

        entity_keys = _extract_entity_keys(_entity_value(row))
        if not entity_keys and graph is not None:
            entity_keys = get_article_entities(news_id, graph)
        for entity_key in entity_keys:
            entity_weights[_normalize_key(entity_key)] += weight

    return subcategory_weights, entity_weights


def _build_skip_profiles(user, df: pd.DataFrame, news_id_to_idx: dict, graph=None) -> tuple[Counter, Counter]:
    category_skips: Counter = Counter()
    entity_skips: Counter = Counter()

    skipped_ids = list(getattr(user, "recent_negative_actions", [])[-25:]) or list(getattr(user, "recent_skips", [])[-25:])
    weighted_skips = list(enumerate(reversed(skipped_ids)))
    for offset, news_id in weighted_skips:
        idx = news_id_to_idx.get(news_id)
        if idx is None:
            continue

        weight = 1.0 / (1.0 + (offset * 0.3))
        row = df.iloc[idx]
        category = _normalize_key(row.get("category", ""))
        if category:
            category_skips[category] += weight

        entity_keys = _extract_entity_keys(_entity_value(row))
        if not entity_keys and graph is not None:
            entity_keys = get_article_entities(news_id, graph)
        for entity_key in entity_keys:
            entity_skips[_normalize_key(entity_key)] += weight

    return category_skips, entity_skips


def _normalize_counter_score(counter: Counter, key: str) -> float:
    if not counter:
        return 0.0
    total = float(sum(counter.values()))
    if total <= 0.0:
        return 0.0
    return float(counter.get(_normalize_key(key), 0.0) / total)


def _disliked_categories(counter: Counter) -> set[str]:
    return {
        category
        for category, weight in counter.items()
        if float(weight) >= _DISLIKED_CATEGORY_THRESHOLD
    }


def _graph_bonus_map(user, graph) -> dict[str, float]:
    if graph is None:
        return {}

    scores: Counter = Counter()
    for category, weight in sorted(user.interests.items(), key=lambda item: item[1], reverse=True)[:5]:
        if float(weight) <= 0:
            continue
        for news_id in get_related_articles(category, graph, limit=250):
            scores[news_id] += float(weight)

    for offset, news_id in enumerate(reversed(user.reading_history[-8:])):
        article_weight = 1.0 / (1.0 + offset)
        for related_id in get_related_articles(news_id, graph, limit=200):
            scores[related_id] += article_weight

    if not scores:
        return {}

    max_score = max(scores.values())
    return {news_id: min(value / max_score, 1.0) for news_id, value in scores.items()}


def get_kg_related_ids(
    news_id: str,
    graph,
    news_id_to_idx: dict,
    limit: int = 20,
) -> list[str]:
    """Return article IDs that are 1-2 hops away from news_id in the KG.

    Used by app.py to propagate rewards to KG-linked articles after feedback,
    so the bandit learns to surface related articles the user hasn't seen yet.
    """
    if graph is None:
        return []
    related = get_related_articles(news_id, graph, limit=limit)
    return [nid for nid in related if nid != news_id and nid in news_id_to_idx]


def _blend_vectors(vectors: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray | None:
    valid_vectors: list[np.ndarray] = []
    valid_weights: list[float] = []
    for idx, vector in enumerate(vectors):
        if vector is None:
            continue
        valid_vectors.append(_l2_normalize(vector))
        valid_weights.append(float(weights[idx]) if weights and idx < len(weights) else 1.0)

    if not valid_vectors:
        return None

    total_weight = float(sum(valid_weights))
    if total_weight <= 0.0:
        return None

    blended = np.zeros_like(valid_vectors[0], dtype=np.float32)
    for vector, weight in zip(valid_vectors, valid_weights):
        blended += vector.astype(np.float32) * (weight / total_weight)
    return _l2_normalize(blended)


def _records_from_ids(df: pd.DataFrame, news_id_to_idx: dict, news_ids: list[str]) -> list[dict]:
    records: list[dict] = []
    for news_id in news_ids:
        idx = news_id_to_idx.get(str(news_id))
        if idx is None:
            continue
        records.append(df.iloc[int(idx)].to_dict())
    return records


def _normalize_score_map(score_map: Counter) -> dict[str, float]:
    if not score_map:
        return {}
    max_score = max(float(score) for score in score_map.values())
    if max_score <= 0.0:
        return {}
    return {
        str(news_id): float(np.clip(float(score) / max_score, 0.0, 1.0))
        for news_id, score in score_map.items()
        if float(score) > 0.0
    }


def build_long_term_memory_signal(
    user,
    df: pd.DataFrame,
    article_embeddings: np.ndarray,
    news_id_to_idx: dict,
    faiss_index=None,
    graph=None,
    seed_vector: np.ndarray | None = None,
    profile_vector: np.ndarray | None = None,
    max_candidates: int = 60,
) -> tuple[list[dict], dict[str, float]]:
    """Build additive long-term-memory candidates and bonuses.

    The source of truth for user memory is query_similar_history(); the helper
    then expands around those past reads via semantic neighbours and KG links.
    """
    if len(df) == 0:
        return [], {}

    if seed_vector is None:
        seed_vector = profile_vector
    if seed_vector is None:
        seed_vector = build_user_profile_vector(user, article_embeddings, news_id_to_idx)
    if seed_vector is None:
        return [], {}

    try:
        memory_hits = query_similar_history(
            user.user_id,
            _l2_normalize(seed_vector),
            top_k=min(max(max_candidates // 6, 6), 12),
        )
    except Exception:
        return [], {}

    if not memory_hits:
        return [], {}

    candidate_scores: Counter = Counter()
    for hit in memory_hits:
        article_id = str(hit.get("article_id") or "")
        hit_idx = news_id_to_idx.get(article_id)
        if hit_idx is None:
            continue

        similarity = float(hit.get("similarity", 0.0))
        feedback_weight = float(hit.get("feedback_weight", 1.0))
        strength = float(np.clip((similarity + 1.0) / 2.0, 0.0, 1.0))
        strength *= float(np.clip(0.5 + feedback_weight, 0.25, 2.0))
        if strength <= 0.0:
            continue

        history_embedding = _l2_normalize(article_embeddings[int(hit_idx)].astype(np.float32))

        if faiss_index is not None:
            search_k = min(len(df), max(max_candidates // 2, 12))
            _, neighbor_indices = faiss_index.search(np.expand_dims(history_embedding, axis=0), search_k)
            for rank, raw_idx in enumerate(neighbor_indices[0]):
                idx = int(raw_idx)
                if idx < 0 or idx >= len(df):
                    continue
                neighbor_id = str(df.iloc[idx]["news_id"])
                if neighbor_id == article_id:
                    continue
                candidate_scores[neighbor_id] += strength / (1.0 + (rank * 0.35))

        if graph is not None:
            for rank, related_id in enumerate(get_kg_related_ids(article_id, graph, news_id_to_idx, limit=max(12, max_candidates // 2))):
                candidate_scores[str(related_id)] += (0.70 * strength) / (1.0 + (rank * 0.25))

            row = df.iloc[int(hit_idx)]
            category = _normalize_key(row.get("category", ""))
            subcategory = _normalize_key(row.get("subcategory", ""))
            entity_keys = _extract_entity_keys(_entity_value(row))

            for rank, related_id in enumerate(get_related_articles(category, graph, limit=20) if category else []):
                if related_id != article_id and related_id in news_id_to_idx:
                    candidate_scores[str(related_id)] += (0.35 * strength) / (1.0 + (rank * 0.30))

            for rank, related_id in enumerate(get_related_articles(subcategory, graph, limit=14) if subcategory else []):
                if related_id != article_id and related_id in news_id_to_idx:
                    candidate_scores[str(related_id)] += (0.25 * strength) / (1.0 + (rank * 0.30))

            for entity_key in entity_keys[:6]:
                for rank, related_id in enumerate(get_related_articles(entity_key, graph, limit=10)):
                    if related_id != article_id and related_id in news_id_to_idx:
                        candidate_scores[str(related_id)] += (0.20 * strength) / (1.0 + (rank * 0.30))

    ranked_ids = [
        str(news_id)
        for news_id, _ in candidate_scores.most_common(max_candidates)
    ]
    return _annotate_records(_records_from_ids(df, news_id_to_idx, ranked_ids), "memory"), _normalize_score_map(candidate_scores)


def build_query_candidate_pool(
    user,
    query: str,
    df: pd.DataFrame,
    article_embeddings: np.ndarray,
    faiss_index,
    graph,
    model,
    max_candidates: int = 120,
    excluded_ids: set[str] | None = None,
) -> tuple[list[dict], dict[str, float]]:
    """Build query candidates from RAG seeds, KG expansion, and vector memory."""
    if len(df) == 0 or not query:
        return [], {}

    news_id_to_idx = _build_news_id_to_idx(df)
    read_ids = set(user.reading_history)
    skipped_ids = set(getattr(user, "recent_skips", [])[-30:])
    excluded_ids = {str(news_id) for news_id in (excluded_ids or set()) if str(news_id)}
    category_skips, _ = _build_skip_profiles(user, df, news_id_to_idx, graph=graph)
    disliked_categories = _disliked_categories(category_skips)

    seen_ids: set[str] = set()
    candidates: list[dict] = []

    def add_records(records: list[dict]):
        for article in records:
            news_id = str(article.get("news_id") or "")
            if (
                not news_id
                or news_id in seen_ids
                or news_id in read_ids
                or news_id in skipped_ids
                or news_id in excluded_ids
            ):
                continue
            if _normalize_key(article.get("category", "")) in disliked_categories:
                continue
            seen_ids.add(news_id)
            candidates.append(article)
            if len(candidates) >= max_candidates:
                break

    rag_records = retrieve_articles(
        query,
        faiss_index,
        df,
        model,
        top_k=min(len(df), max(max_candidates // 2, 24)),
    )
    add_records(rag_records)

    query_embedding = encode_query_embedding(query, model)
    profile_vector = build_user_profile_vector(user, article_embeddings, news_id_to_idx)
    memory_seed = _blend_vectors([query_embedding, profile_vector], weights=[0.65, 0.35])
    memory_records, memory_bonus_map = build_long_term_memory_signal(
        user,
        df,
        article_embeddings,
        news_id_to_idx,
        faiss_index=faiss_index,
        graph=graph,
        seed_vector=memory_seed,
        profile_vector=profile_vector,
        max_candidates=max(max_candidates // 2, 24),
    )

    if graph is not None and len(candidates) < max_candidates:
        graph_ids: list[str] = []
        for article in rag_records[:12]:
            news_id = str(article.get("news_id") or "")
            if not news_id:
                continue
            graph_ids.extend(get_kg_related_ids(news_id, graph, news_id_to_idx, limit=18))

            category = _normalize_key(article.get("category", ""))
            subcategory = _normalize_key(article.get("subcategory", ""))
            entity_keys = _extract_entity_keys(_entity_value(article))

            if category:
                graph_ids.extend([
                    related_id
                    for related_id in get_related_articles(category, graph, limit=20)
                    if related_id in news_id_to_idx
                ])
            if subcategory:
                graph_ids.extend([
                    related_id
                    for related_id in get_related_articles(subcategory, graph, limit=14)
                    if related_id in news_id_to_idx
                ])
            for entity_key in entity_keys[:6]:
                graph_ids.extend([
                    related_id
                    for related_id in get_related_articles(entity_key, graph, limit=10)
                    if related_id in news_id_to_idx
                ])

        add_records(_records_from_ids(df, news_id_to_idx, graph_ids))

    add_records(memory_records)

    if len(candidates) < max_candidates:
        fallback_pool = build_candidate_pool(
            user,
            df,
            article_embeddings,
            faiss_index=faiss_index,
            graph=graph,
            max_candidates=max_candidates,
            memory_records=memory_records,
            profile_vector=profile_vector,
            excluded_ids=excluded_ids,
        )
        add_records(fallback_pool)

    return candidates[:max_candidates], memory_bonus_map


def build_candidate_pool(
    user,
    df: pd.DataFrame,
    article_embeddings: np.ndarray,
    faiss_index=None,
    graph=None,
    max_candidates: int = 200,
    memory_records: list[dict] | None = None,
    profile_vector: np.ndarray | None = None,
    excluded_ids: set[str] | None = None,
) -> list[dict]:
    if len(df) == 0:
        return []

    news_id_to_idx = _build_news_id_to_idx(df)
    seen_ids: set[str] = set()
    read_ids = set(user.reading_history)
    skipped_ids = set(getattr(user, "recent_skips", [])[-30:])
    excluded_ids = {str(news_id) for news_id in (excluded_ids or set()) if str(news_id)}
    category_skips, _ = _build_skip_profiles(user, df, news_id_to_idx, graph=graph)
    disliked_categories = _disliked_categories(category_skips)
    candidates: list[dict] = []

    def add_records(records: list[dict], source: str | None = None):
        for raw_article in records:
            article = _append_candidate_source(raw_article, source) if source else dict(raw_article)
            news_id = article.get("news_id")
            if (
                not news_id
                or news_id in seen_ids
                or news_id in read_ids
                or news_id in skipped_ids
                or news_id in excluded_ids
            ):
                continue
            if _normalize_key(article.get("category", "")) in disliked_categories:
                continue
            seen_ids.add(news_id)
            candidates.append(article)
            if len(candidates) >= max_candidates:
                break

    if profile_vector is None:
        profile_vector = build_user_profile_vector(user, article_embeddings, news_id_to_idx)
    if profile_vector is not None and faiss_index is not None:
        search_k = min(len(df), max(max_candidates * 3, 80))
        query = np.expand_dims(profile_vector.astype(np.float32), axis=0)
        _, indices = faiss_index.search(query, search_k)
        semantic_records = [
            df.iloc[int(idx)].to_dict()
            for idx in indices[0]
            if 0 <= int(idx) < len(df)
        ]
        add_records(semantic_records, source="semantic")

    # Category-balanced FAISS: after enough interactions, also retrieve from
    # non-dominant categories to force diversity into the candidate pool.
    if (
        profile_vector is not None
        and faiss_index is not None
        and getattr(user, "total_positive_interactions", 0) >= _CAT_BALANCE_MIN_INTERACTIONS
        and user.interests
        and len(candidates) < max_candidates
    ):
        dominant_cat = _normalize_key(max(user.interests, key=lambda c: user.interests.get(c, 0.0)))
        non_dom_df = df[df["category"].fillna("").str.lower() != dominant_cat]
        if len(non_dom_df) > 0:
            nd_idx = non_dom_df.index.tolist()
            bal_k = min(len(non_dom_df), max(max_candidates // 4, 20))
            nd_embs = article_embeddings[nd_idx].astype(np.float32)
            nd_embs = np.asarray([_l2_normalize(embedding) for embedding in nd_embs], dtype=np.float32)
            query_vec = _l2_normalize(profile_vector.astype(np.float32))
            similarity_scores = nd_embs @ query_vec
            sub_res = np.argsort(-similarity_scores)[:bal_k]
            bal_records = [
                non_dom_df.iloc[int(i)].to_dict()
                for i in sub_res
                if 0 <= int(i) < len(non_dom_df)
            ]
            add_records(bal_records, source="semantic_balance")

    if memory_records is None:
        memory_records, _ = build_long_term_memory_signal(
            user,
            df,
            article_embeddings,
            news_id_to_idx,
            faiss_index=faiss_index,
            graph=graph,
            profile_vector=profile_vector,
            max_candidates=max(max_candidates // 2, 24),
        )
    if memory_records:
        add_records(memory_records, source="memory")

    if len(candidates) < max_candidates and graph is not None:
        graph_news_ids = []
        for history_id in user.reading_history[-5:]:
            graph_news_ids.extend(get_related_articles(history_id, graph, limit=80))
        for category, _ in sorted(user.interests.items(), key=lambda item: item[1], reverse=True)[:3]:
            graph_news_ids.extend(get_related_articles(category, graph, limit=60))
        graph_records = [
            df.iloc[news_id_to_idx[news_id]].to_dict()
            for news_id in graph_news_ids
            if news_id in news_id_to_idx
        ]
        add_records(graph_records, source="kg_expand")

    if len(candidates) < max_candidates and user.interests:
        preferred_categories = [
            cat
            for cat, weight in sorted(user.interests.items(), key=lambda item: item[1], reverse=True)
            if weight > 0 and _normalize_key(cat) not in disliked_categories
        ]
        for category in preferred_categories[:4]:
            cat_rows = df[df["category"].fillna("").str.lower() == _normalize_key(category)]
            if "popularity" in cat_rows.columns:
                cat_rows = cat_rows.sort_values("popularity", ascending=False)
            add_records(cat_rows.to_dict("records"), source="interest")
            if len(candidates) >= max_candidates:
                break

    if len(candidates) < max_candidates:
        unread = df[~df["news_id"].isin(read_ids | skipped_ids | excluded_ids)]
        if disliked_categories:
            unread = unread[~unread["category"].fillna("").str.lower().isin(disliked_categories)]
        if "popularity" in unread.columns:
            unread = unread.sort_values("popularity", ascending=False)
        add_records(unread.head(max_candidates).to_dict("records"), source="popular_fallback")

    if len(candidates) < max_candidates:
        unread = df[~df["news_id"].isin(read_ids | skipped_ids | seen_ids | excluded_ids)]
        if disliked_categories:
            unread = unread[~unread["category"].fillna("").str.lower().isin(disliked_categories)]
        sample_size = min(max_candidates - len(candidates), len(unread))
        if sample_size > 0:
            add_records(unread.sample(sample_size, random_state=42).to_dict("records"), source="random_fallback")

    if not candidates:
        return df.head(max_candidates).to_dict("records")

    return candidates[:max_candidates]


def _apply_mmr(
    ranked: list[dict],
    n: int,
    user,
    article_embeddings: np.ndarray,
    news_id_to_idx: dict,
    lambda_weight: float = _MMR_LAMBDA,
) -> list[dict]:
    """Maximal Marginal Relevance re-ranking (λ=0.82).

    Iteratively picks the article that maximises:
        λ * relevance_score  -  (1-λ) * max_cosine_sim_to_already_selected

    Articles in a "locked" category (≥ _CATEGORY_LOCK_THRESHOLD confirmed reads)
    are inserted ahead of MMR to preserve deliberate strong preferences.
    """
    if n <= 0 or not ranked:
        return ranked[:n]

    # Count confirmed reads per category from the ranked pool
    confirmed_reads: Counter = Counter()
    news_id_to_cat: dict[str, str] = {}
    for article in ranked:
        nid = article.get("news_id")
        cat = _normalize_key(article.get("category", ""))
        if nid:
            news_id_to_cat[nid] = cat
    for nid in user.reading_history:
        cat = news_id_to_cat.get(nid)
        if cat:
            confirmed_reads[cat] += 1

    locked: list[dict] = []
    candidates: list[dict] = []
    for article in ranked:
        cat = _normalize_key(article.get("category", ""))
        if confirmed_reads.get(cat, 0) >= _CATEGORY_LOCK_THRESHOLD:
            locked.append(article)
        else:
            candidates.append(article)

    selected: list[dict] = []
    selected_embeddings: list[np.ndarray] = []

    while len(selected) < n and (locked or candidates):
        # Insert locked articles first (user proved they want this category)
        if locked:
            article = locked.pop(0)
            selected.append(article)
            idx = news_id_to_idx.get(article.get("news_id"))
            if idx is not None:
                selected_embeddings.append(_l2_normalize(article_embeddings[idx]))
            continue

        # MMR over remaining candidates
        best_score = -1e9
        best_idx = -1
        for i, article in enumerate(candidates):
            nid = article.get("news_id")
            idx = news_id_to_idx.get(nid)
            if idx is None:
                continue
            emb_c = _l2_normalize(article_embeddings[idx])
            relevance = float(article.get("score", 0.0))
            if selected_embeddings:
                max_sim = max(float(np.dot(emb_c, e)) for e in selected_embeddings)
            else:
                max_sim = 0.0
            mmr_score = lambda_weight * relevance - (1.0 - lambda_weight) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        if best_idx < 0:
            break
        article = candidates.pop(best_idx)
        selected.append(article)
        idx = news_id_to_idx.get(article.get("news_id"))
        if idx is not None:
            selected_embeddings.append(_l2_normalize(article_embeddings[idx]))

    return selected[:n]


def _mmr_lambda_from_explore_focus(explore_focus: float | None) -> float:
    try:
        focus = float(explore_focus)
    except (TypeError, ValueError):
        focus = 55.0
    normalized = float(np.clip(focus / 100.0, 0.0, 1.0))
    return float(0.72 + (0.18 * normalized))


def _derive_reason_labels(article: dict, query_intent: dict | None, features: dict[str, float]) -> list[str]:
    reasons: list[str] = []
    if query_intent and query_intent.get("query"):
        reasons.append(f"Matches your search for '{query_intent['query']}'")

    matched_entities = article.get("matched_entities", []) if isinstance(article.get("matched_entities", []), list) else []
    if matched_entities:
        reasons.append(f"Connected via {matched_entities[0]}")

    candidate_source = str(article.get("candidate_source", "") or "")
    if candidate_source:
        reasons.append(f"Candidate source: {candidate_source}")

    if float(features.get("memory_score", 0.0)) >= 0.25:
        reasons.append("Because you engaged with similar stories recently")
    if float(features.get("kg_score", 0.0)) >= 0.25:
        reasons.append("Strong knowledge-graph affinity")
    if float(features.get("entity_score", 0.0)) >= 0.20:
        reasons.append("Entity match with your recent interests")

    return reasons[:4]


def _apply_bounded_exploration(sorted_articles: list[dict]) -> list[dict]:
    if len(sorted_articles) <= 4:
        return sorted_articles

    locked = list(sorted_articles[:3])
    exploration_slice = list(sorted_articles[3:15])
    remainder = list(sorted_articles[15:])

    if not exploration_slice:
        return sorted_articles

    raw_scores = np.asarray([float(item.get("bandit_score", 0.0)) for item in exploration_slice], dtype=np.float32)
    if float(np.max(raw_scores)) > float(np.min(raw_scores)):
        normalized = (raw_scores - float(np.min(raw_scores))) / (float(np.max(raw_scores)) - float(np.min(raw_scores)) + 1e-6)
    else:
        normalized = np.zeros_like(raw_scores)

    blended = []
    for article, bandit_norm in zip(exploration_slice, normalized):
        blended_score = (0.88 * float(article.get("score", 0.0))) + (0.12 * float(bandit_norm))
        blended.append({**article, "blended_score": blended_score})

    blended.sort(key=lambda item: item.get("blended_score", item.get("score", 0.0)), reverse=True)
    return locked + blended + remainder


def build_article_feature_map(
    user,
    article: dict,
    article_embeddings: np.ndarray,
    df: pd.DataFrame,
    *,
    bandit=None,
    G=None,
    news_id_to_idx: dict | None = None,
    profile_vector: np.ndarray | None = None,
    subcategory_weights: Counter | None = None,
    entity_weights: Counter | None = None,
    skipped_categories: Counter | None = None,
    skipped_entities: Counter | None = None,
    skipped_ids: set[str] | None = None,
    graph_bonus: dict[str, float] | None = None,
    memory_bonus_map: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict]:
    news_id_to_idx = news_id_to_idx or _build_news_id_to_idx(df)
    subcategory_weights = subcategory_weights or Counter()
    entity_weights = entity_weights or Counter()
    skipped_categories = skipped_categories or Counter()
    skipped_entities = skipped_entities or Counter()
    skipped_ids = skipped_ids or set()
    graph_bonus = graph_bonus or {}
    memory_bonus_map = memory_bonus_map or {}

    news_id = article.get("news_id")
    if not news_id or news_id not in news_id_to_idx:
        return {}, {}

    idx = news_id_to_idx[news_id]
    article_embedding = _l2_normalize(article_embeddings[idx])
    raw_kg_score = float(graph_bonus.get(news_id, 0.0))
    category = article.get("category", "")
    context_vector = build_context_vector(user, article_embedding, kg_score=raw_kg_score)
    bandit_score = float(bandit.score(news_id, context_vector, category=category)) if bandit else 0.0

    semantic_score = 0.0
    if profile_vector is not None:
        semantic_score = float((np.dot(profile_vector, article_embedding) + 1.0) / 2.0)

    interest_score = _score_interest(user, category)
    subcategory_score = _normalize_counter_score(subcategory_weights, article.get("subcategory", ""))
    candidate_entities = _extract_entity_keys(_entity_value(article))
    entity_score = 0.0
    if candidate_entities:
        entity_score = max(
            (_normalize_counter_score(entity_weights, ek) for ek in candidate_entities),
            default=0.0,
        )

    popularity_score = _score_popularity(article)
    context_multiplier = compute_context_score(category, user.mood, user.time_of_day)
    lexical_score = float(article.get("lexical_score", 0.0))
    retrieval_score = float(article.get("retrieval_score", 0.0))
    dense_score = float(article.get("dense_score", 0.0))
    memory_bonus = float(memory_bonus_map.get(str(news_id), 0.0))
    skipped_category_norm = _normalize_counter_score(skipped_categories, category)
    skipped_category_penalty = _SKIP_CATEGORY_PENALTY_WEIGHT * max(skipped_category_norm - 0.45, 0.0)
    skipped_entity_norm = max(
        (_normalize_counter_score(skipped_entities, ek) for ek in candidate_entities),
        default=0.0,
    )
    skipped_entity_penalty = _SKIP_ENTITY_PENALTY_WEIGHT * max(skipped_entity_norm - 0.60, 0.0)
    skipped_article_penalty = 0.45 if news_id in skipped_ids else 0.0
    negative_score = float(skipped_category_norm) + float(skipped_entity_norm)
    repeat_penalty = 0.40 if news_id in set(getattr(user, "reading_history", [])) else 0.0

    location_score, _ = _location_match_score(article, user)
    occupation_category_weights = infer_occupation_category_weights(
        str(getattr(user, "occupation", "") or "")
    )
    article_category_key = _normalize_key(article.get("category", ""))
    article_subcategory_key = _normalize_key(article.get("subcategory", ""))
    occupation_score = max(
        float(occupation_category_weights.get(article_category_key, 0.0)),
        float(occupation_category_weights.get(article_subcategory_key, 0.0)),
    )

    features = {
        "semantic_score": float(semantic_score),
        "retrieval_score": float(retrieval_score),
        "lexical_score": float(lexical_score),
        "dense_score": float(dense_score),
        "memory_score": float(memory_bonus),
        "kg_score": float(raw_kg_score),
        "entity_score": float(entity_score),
        "subcategory_score": float(subcategory_score),
        "popularity_score": float(popularity_score),
        "context_score": float(context_multiplier),
        "negative_score": float(negative_score),
        "interest_score": float(interest_score),
        "bandit_score": float(bandit_score),
        "location_score": float(location_score),
        "occupation_score": float(occupation_score),
    }
    extras = {
        "context_multiplier": float(context_multiplier),
        "repeat_penalty": float(repeat_penalty),
        "skipped_category_penalty": float(skipped_category_penalty),
        "skipped_entity_penalty": float(skipped_entity_penalty),
        "skipped_article_penalty": float(skipped_article_penalty),
        "candidate_entities": candidate_entities,
        "article_embedding": article_embedding,
        "news_id": news_id,
        "category": category,
    }
    return features, extras


def rank_articles(
    user,
    candidate_articles: list,
    article_embeddings: np.ndarray,
    bandit,
    df: pd.DataFrame,
    G=None,
    n: int | None = None,
    memory_bonus_map: dict[str, float] | None = None,
    ltr_scorer=None,
    query_intent: dict | None = None,
    explore_focus: float | None = None,
) -> list:
    scored = []
    news_id_to_idx = _build_news_id_to_idx(df)
    profile_vector = build_user_profile_vector(user, article_embeddings, news_id_to_idx)
    previously_seen = set(user.reading_history)
    subcategory_weights, entity_weights = _build_history_profiles(user, df, news_id_to_idx, graph=G)
    skipped_categories, skipped_entities = _build_skip_profiles(user, df, news_id_to_idx, graph=G)
    skipped_ids = set(getattr(user, "recent_negative_actions", [])[-25:]) | set(getattr(user, "recent_skips", [])[-25:])
    graph_bonus = _graph_bonus_map(user, G)
    memory_bonus_map = memory_bonus_map or {}

    for article in candidate_articles:
        features, extras = build_article_feature_map(
            user,
            article,
            article_embeddings,
            df,
            bandit=bandit,
            G=G,
            news_id_to_idx=news_id_to_idx,
            profile_vector=profile_vector,
            subcategory_weights=subcategory_weights,
            entity_weights=entity_weights,
            skipped_categories=skipped_categories,
            skipped_entities=skipped_entities,
            skipped_ids=skipped_ids,
            graph_bonus=graph_bonus,
            memory_bonus_map=memory_bonus_map,
        )
        news_id = article.get("news_id")
        if not news_id or not features:
            continue

        if ltr_scorer is not None:
            base_score = float(ltr_scorer.score(features))
        else:
            base_score = (
                (0.24 * float(features.get("semantic_score", 0.0)))
                + (0.18 * float(features.get("retrieval_score", 0.0)))
                + (0.08 * float(features.get("dense_score", 0.0)))
                + (0.15 * float(features.get("lexical_score", 0.0)))
                + (0.15 * float(features.get("interest_score", 0.0)))
                + (0.10 * float(features.get("subcategory_score", 0.0)))
                + (0.10 * float(features.get("entity_score", 0.0)))
                + (0.08 * float(features.get("popularity_score", 0.0)))
                + (0.05 * float(features.get("location_score", 0.0)))
                + (0.04 * float(features.get("occupation_score", 0.0)))
            )
        query_present = bool(query_intent and str(query_intent.get("query") or "").strip())
        candidate_source = str(article.get("candidate_source", article.get("source", "ranker")))
        memory_weight = 0.04 if query_present else _MEMORY_BONUS_WEIGHT
        kg_weight = 0.05 if query_present else 0.08
        query_focus_bonus = 0.0
        if query_present:
            query_focus_bonus = (
                (0.30 * float(features.get("retrieval_score", 0.0)))
                + (0.12 * float(features.get("lexical_score", 0.0)))
                + (0.08 * float(features.get("dense_score", 0.0)))
                + (0.04 * float(features.get("entity_score", 0.0)))
            )
            if "lexical" in candidate_source:
                query_focus_bonus += 0.03
            if "dense" in candidate_source:
                query_focus_bonus += 0.02
            if candidate_source == "session_memory":
                query_focus_bonus -= 0.03
        final_score = (
            (base_score * float(extras.get("context_multiplier", 1.0)))
            + (kg_weight * float(features.get("kg_score", 0.0)))
            + (memory_weight * float(features.get("memory_score", 0.0)))
            + float(query_focus_bonus)
            - float(extras.get("repeat_penalty", 0.0))
            - float(extras.get("skipped_category_penalty", 0.0))
            - float(extras.get("skipped_entity_penalty", 0.0))
            - float(extras.get("skipped_article_penalty", 0.0))
        )
        reasons = list(article.get("reasons", [])) if isinstance(article.get("reasons", []), list) else []
        reasons.extend(_derive_reason_labels(article, query_intent, features))
        scored.append(
            {
                **article,
                "score": float(final_score),
                "bandit_score": float(features.get("bandit_score", 0.0)),
                "ltr_score": float(base_score),
                "semantic_alignment": float(features.get("semantic_score", 0.0)),
                "entity_alignment": float(features.get("entity_score", 0.0)),
                "candidate_source": candidate_source,
                "reasons": list(dict.fromkeys(reasons))[:4],
                "score_breakdown": features,
            }
        )

    sorted_articles = sorted(scored, key=lambda x: x["score"], reverse=True)
    sorted_articles = _apply_bounded_exploration(sorted_articles)

    if n is not None:
        return _apply_mmr(
            sorted_articles,
            n,
            user,
            article_embeddings,
            news_id_to_idx,
            lambda_weight=_mmr_lambda_from_explore_focus(explore_focus),
        )

    return sorted_articles


def rank_search_articles(
    user,
    candidate_articles: list,
    article_embeddings: np.ndarray,
    df: pd.DataFrame,
    query_intent: dict | None = None,
    n: int | None = None,
) -> list:
    query_intent = query_intent or {}
    query_text = str(query_intent.get("normalized_query") or query_intent.get("query") or "").strip().lower()
    query_tokens = {_normalize_key(token) for token in query_intent.get("tokens", []) if _normalize_key(token)}
    matched_entities = {
        _normalize_key(value)
        for value in query_intent.get("matched_entities", [])
        if _normalize_key(value)
    }
    matched_categories = {
        _normalize_key(value)
        for value in query_intent.get("matched_categories", [])
        if _normalize_key(value)
    }
    matched_sources = {
        _normalize_key(value)
        for value in query_intent.get("matched_sources", [])
        if _normalize_key(value)
    }
    news_id_to_idx = _build_news_id_to_idx(df)
    profile_vector = build_user_profile_vector(user, article_embeddings, news_id_to_idx)
    subcategory_weights, history_entity_weights = _build_history_profiles(user, df, news_id_to_idx, graph=None)

    scored: list[dict] = []
    for article in candidate_articles:
        title = str(article.get("title", "") or "")
        abstract = str(article.get("abstract", "") or "")
        category = str(article.get("category", "") or "")
        subcategory = str(article.get("subcategory", "") or "")
        source = str(article.get("source", "") or "")
        text = " ".join([title, abstract, category, subcategory, source]).lower()
        text_tokens = {_normalize_key(token) for token in text.split() if _normalize_key(token)}
        article_entities = {
            _normalize_key(value)
            for value in _extract_entity_keys(_entity_value(article))
            if _normalize_key(value)
        }

        phrase_match = 1.0 if query_text and query_text in text else 0.0
        token_overlap = (len(query_tokens.intersection(text_tokens)) / float(len(query_tokens))) if query_tokens else 0.0
        entity_match = (len(matched_entities.intersection(article_entities)) / float(len(matched_entities))) if matched_entities else 0.0
        category_match = 1.0 if matched_categories and (
            _normalize_key(category) in matched_categories or _normalize_key(subcategory) in matched_categories
        ) else 0.0
        source_match = 1.0 if matched_sources and _normalize_key(source) in matched_sources else 0.0
        kg_query_score = 1.0 if "kg_expand" in str(article.get("candidate_source", "")) else 0.0
        candidate_source = str(article.get("candidate_source", "") or "search")
        source_bonus = (
            (0.03 if "lexical" in candidate_source else 0.0)
            + (0.02 if "dense" in candidate_source else 0.0)
            + (0.015 if "kg_expand" in candidate_source else 0.0)
        )
        semantic_personalization = 0.0
        news_id = str(article.get("news_id") or "")
        idx = news_id_to_idx.get(news_id)
        if profile_vector is not None and idx is not None:
            article_embedding = _l2_normalize(article_embeddings[int(idx)])
            semantic_personalization = float((np.dot(profile_vector, article_embedding) + 1.0) / 2.0)
        interest_personalization = _score_interest(user, category)
        subcategory_personalization = _normalize_counter_score(subcategory_weights, subcategory)
        history_entity_personalization = max(
            (_normalize_counter_score(history_entity_weights, entity) for entity in article_entities),
            default=0.0,
        )

        score = (
            (0.40 * float(article.get("retrieval_score", 0.0)))
            + (0.24 * float(article.get("lexical_score", 0.0)))
            + (0.18 * float(article.get("dense_score", 0.0)))
            + (0.12 * phrase_match)
            + (0.10 * token_overlap)
            + (0.10 * entity_match)
            + (0.08 * category_match)
            + (0.05 * source_match)
            + (0.05 * kg_query_score)
            + (0.02 * _score_popularity(article))
            + (0.06 * semantic_personalization)
            + (0.04 * interest_personalization)
            + (0.03 * history_entity_personalization)
            + (0.02 * subcategory_personalization)
            + source_bonus
        )

        reasons = list(article.get("reasons", [])) if isinstance(article.get("reasons", []), list) else []
        if phrase_match >= 1.0:
            reasons.append("Exact phrase match in article text")
        elif token_overlap >= 0.5:
            reasons.append("Strong token overlap with your search")
        if entity_match > 0.0:
            reasons.append("Entity overlap with your search")
        if category_match > 0.0:
            reasons.append("Matches the searched category")
        if source_match > 0.0:
            reasons.append("Matches the searched source")
        if semantic_personalization >= 0.6:
            reasons.append("Aligned with stories you engaged with before")

        scored.append(
            {
                **article,
                "score": float(score),
                "ltr_score": float(score),
                "bandit_score": 0.0,
                "semantic_alignment": 0.0,
                "entity_alignment": float(entity_match),
                "candidate_source": candidate_source,
                "reasons": list(dict.fromkeys(reasons))[:4],
                "score_breakdown": {
                    "retrieval_score": float(article.get("retrieval_score", 0.0)),
                    "lexical_score": float(article.get("lexical_score", 0.0)),
                    "dense_score": float(article.get("dense_score", 0.0)),
                    "query_phrase_match": float(phrase_match),
                    "query_token_overlap": float(token_overlap),
                    "query_entity_match": float(entity_match),
                    "query_category_match": float(category_match),
                    "query_source_match": float(source_match),
                    "kg_query_score": float(kg_query_score),
                    "popularity_score": float(_score_popularity(article)),
                    "semantic_personalization": float(semantic_personalization),
                    "interest_personalization": float(interest_personalization),
                    "history_entity_personalization": float(history_entity_personalization),
                    "subcategory_personalization": float(subcategory_personalization),
                },
            }
        )

    scored.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return scored[:n] if n is not None else scored


def cold_start_recommendations(
    user,
    df: pd.DataFrame,
    article_embeddings: np.ndarray | None = None,
    n: int = 10,
    excluded_ids: set[str] | None = None,
) -> list:
    time_category_map = {
        "morning": ["news", "politics", "business", "finance", "technology"],
        "afternoon": ["sports", "technology", "business", "news"],
        "evening": ["entertainment", "lifestyle", "sports", "health"],
        "night": ["entertainment", "lifestyle", "health", "movies"],
    }
    preferred_cats = [_normalize_key(value) for value in time_category_map.get(user.time_of_day, ["news"])]

    avoid_cats = {
        "stressed": ["politics", "health"],
        "tired": ["politics", "business", "finance"],
    }.get(user.mood, [])

    mood_boosts = {
        "curious": ["technology", "science", "business", "news"],
        "happy": ["sports", "entertainment", "lifestyle"],
        "stressed": ["entertainment", "lifestyle", "sports"],
        "tired": ["entertainment", "lifestyle", "health"],
    }.get(user.mood, [])

    news_id_to_idx = _build_news_id_to_idx(df)
    category_skips, _ = _build_skip_profiles(user, df, news_id_to_idx)
    disliked_categories = _disliked_categories(category_skips)
    skipped_ids = set(getattr(user, "recent_skips", [])[-30:])
    excluded_ids = {str(news_id) for news_id in (excluded_ids or set()) if str(news_id)}

    cat_lower = df["category"].fillna("").str.lower()
    filtered = df[
        ~cat_lower.isin(avoid_cats)
        & ~cat_lower.isin(disliked_categories)
        & ~df["news_id"].isin(skipped_ids | excluded_ids)
    ]

    if len(filtered) == 0:
        filtered = df

    bio_category_prior = _decode_bio_category_prior(user)
    category_preference_weights, weighted_preference_order = _cold_start_category_preferences(user, bio_category_prior)
    inferred_interest_categories = infer_interest_category_weights(
        str(getattr(user, "interest_text", "") or ""),
        limit=6,
    )
    onboarding_category_keys = weighted_preference_order[:4] or _preferred_onboarding_categories(user, bio_category_prior, limit=4)
    onboarding_category_set = set(onboarding_category_keys)
    interest_text = str(getattr(user, "interest_text", "") or "").strip().lower()
    interest_terms = _interest_terms_from_text(interest_text)
    bio_text_embedding = _vector_or_none(getattr(user, "bio_text_embedding", []))
    interaction_count = int(getattr(user, "total_positive_interactions", 0) or 0)
    cold_start_decay = max(0.1, 1.0 - (interaction_count / 20.0))
    popularity_values = filtered["popularity"].astype(float) if "popularity" in filtered.columns else pd.Series(dtype=float)
    popularity_min = float(popularity_values.min()) if len(popularity_values) else 0.0
    popularity_max = float(popularity_values.max()) if len(popularity_values) else 0.0
    popularity_range = max(popularity_max - popularity_min, 1.0)

    scored_records: list[dict] = []
    for article in filtered.to_dict("records"):
        category_key = _normalize_key(article.get("category", ""))
        subcategory_key = _normalize_key(article.get("subcategory", ""))
        title_text = str(article.get("title") or "").strip().lower()
        abstract_text = str(article.get("abstract") or "").strip().lower()
        popularity = float(article.get("popularity") or 0.0)
        score = (popularity - popularity_min) / popularity_range
        score += compute_context_score(category_key, user.mood, user.time_of_day) - 1.0
        if category_key in preferred_cats:
            score += 0.35
        if category_key in mood_boosts:
            score += 0.45

        onboarding_alignment = 0.0
        preferred_bucket = ""
        category_preference_score = max(
            float(category_preference_weights.get(category_key, 0.0)),
            float(category_preference_weights.get(subcategory_key, 0.0)),
        )
        if category_preference_score > 0.0:
            score += category_preference_score * 1.35 * cold_start_decay
            onboarding_alignment += category_preference_score
            if category_key in onboarding_category_set or subcategory_key in onboarding_category_set:
                preferred_bucket = category_key if category_key in onboarding_category_set else subcategory_key

        category_prior_score = max(
            float(bio_category_prior.get(category_key, 0.0)),
            float(bio_category_prior.get(subcategory_key, 0.0)),
        )
        if category_prior_score > 0.0:
            score += category_prior_score * 0.70 * cold_start_decay
            onboarding_alignment += category_prior_score * 0.85

        matched_terms: list[str] = []
        if interest_terms:
            matched_terms_list = [
                term
                for term in interest_terms
                if term in title_text
                or term in abstract_text
                or term == category_key
                or term == subcategory_key
            ][:2]
            matched_terms = matched_terms_list
            if matched_terms_list:
                score += min(0.75, len(matched_terms_list) * 0.18) * cold_start_decay
                onboarding_alignment += min(0.42, len(matched_terms_list) * 0.12)

        bio_affinity = 0.0
        if bio_text_embedding is not None and article_embeddings is not None:
            news_id = str(article.get("news_id") or "").strip()
            idx = news_id_to_idx.get(news_id)
            if idx is not None and idx < len(article_embeddings):
                article_vec = _vector_or_none(article_embeddings[idx])
                if article_vec is not None and article_vec.shape == bio_text_embedding.shape:
                    bio_affinity = max(0.0, float(np.dot(article_vec, bio_text_embedding)))
                    score += bio_affinity * 0.85 * cold_start_decay
                    onboarding_alignment += bio_affinity * 0.40

        location_match_score, location_match_terms = _location_match_score(article, user)
        if location_match_score > 0.0:
            score += location_match_score * 0.68 * cold_start_decay
            onboarding_alignment += location_match_score * 0.30

        reasons = _cold_start_reason_labels(
            article,
            user,
            category_preference_score=category_preference_score,
            inferred_interest_categories=inferred_interest_categories,
            matched_terms=matched_terms,
            location_matches=location_match_terms,
            bio_affinity=bio_affinity,
        )

        record = dict(article)
        record["candidate_source"] = "cold_start_profile"
        record["reasons"] = reasons
        record["_cold_start_score"] = float(score)
        record["_onboarding_alignment"] = float(onboarding_alignment)
        record["_preferred_bucket"] = preferred_bucket
        scored_records.append(record)

    filtered = pd.DataFrame(scored_records)
    filtered = filtered.sort_values("_cold_start_score", ascending=False)

    available_categories = [
        category
        for category in filtered["category"].fillna("").astype(str).str.lower().unique().tolist()
        if category and category not in disliked_categories
    ]
    category_priority = sorted(
        available_categories,
        key=lambda category: (
            float(
                filtered[filtered["category"].fillna("").astype(str).str.lower() == category]["_cold_start_score"].max()
                if len(filtered)
                else 0.0
            )
            + compute_context_score(category, user.mood, user.time_of_day)
            + (0.35 if category in preferred_cats else 0.0)
            + (0.45 if category in mood_boosts else 0.0)
            + (0.75 * float(category_preference_weights.get(category, 0.0)))
            + (0.60 if category in onboarding_category_set else 0.0)
            + (0.35 * float(bio_category_prior.get(category, 0.0)))
        ),
        reverse=True,
    )

    preferred_target = min(n, max(1, int(np.ceil(n * 0.625)))) if onboarding_category_set else 0
    preferred_buckets = {
        category: filtered[filtered["_preferred_bucket"] == category].to_dict("records")
        for category in onboarding_category_keys
    }
    category_buckets = {
        category: filtered[filtered["category"].fillna("").str.lower() == category].to_dict("records")
        for category in category_priority
    }

    picks: list[dict] = []
    used_ids: set[str] = set()
    while len(picks) < preferred_target:
        progressed = False
        for category in onboarding_category_keys:
            bucket = preferred_buckets.get(category, [])
            while bucket and bucket[0].get("news_id") in used_ids:
                bucket.pop(0)
            if not bucket:
                continue
            article = bucket.pop(0)
            news_id = article.get("news_id")
            if not news_id or news_id in used_ids:
                continue
            used_ids.add(news_id)
            picks.append(article)
            progressed = True
            if len(picks) >= preferred_target:
                break
        if not progressed:
            break

    while len(picks) < n:
        progressed = False
        for category in category_priority:
            bucket = category_buckets.get(category, [])
            while bucket and bucket[0].get("news_id") in used_ids:
                bucket.pop(0)
            if not bucket:
                continue
            article = bucket.pop(0)
            news_id = article.get("news_id")
            if not news_id or news_id in used_ids:
                continue
            used_ids.add(news_id)
            picks.append(article)
            progressed = True
            if len(picks) >= n:
                break
        if not progressed:
            break

    if len(picks) < n:
        extra = filtered[~filtered["news_id"].isin(used_ids)].head(n - len(picks)).to_dict("records")
        picks.extend(extra)

    for article in picks:
        article.pop("_cold_start_score", None)
        article.pop("_onboarding_alignment", None)
        article.pop("_preferred_bucket", None)

    return picks[:n]
