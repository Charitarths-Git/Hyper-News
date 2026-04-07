"""
Hybrid retrieval helpers for vNext search and query understanding.

This module keeps dependencies light so the app can still run locally without
OpenSearch, Qdrant, or a dedicated reranker service. The same interfaces can be
reused later by production adapters.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from difflib import get_close_matches
import math
import re
import time
from typing import Any

import numpy as np
import pandas as pd

try:
    from backend.graph import get_related_articles
    from backend.rag_pipeline import encode_query_embedding
except ImportError:
    from graph import get_related_articles
    from rag_pipeline import encode_query_embedding


_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_COMMON_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "about",
    "what",
    "when",
    "where",
    "have",
    "will",
    "your",
    "news",
}
_IMPORTANT_SHORT_TOKENS = {"ai", "uk", "us", "tv", "vr", "ml"}
_SEARCH_CACHE_REGISTRY: dict[int, dict[str, Any]] = {}
_EXPLICIT_SEARCH_CACHE: dict[tuple[str, str, int, int], tuple[float, list[dict[str, Any]], dict[str, Any]]] = {}
_EXPLICIT_SEARCH_CACHE_TTL_SECONDS = 120.0


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in _TOKEN_RE.findall(str(text or "").lower())
        if ((len(token) > 2) or (token in _IMPORTANT_SHORT_TOKENS)) and token not in _COMMON_STOPWORDS
    ]


def _normalize_key(value: str) -> str:
    return str(value or "").strip().lower()


def _rrf_add(score_map: dict[int, float], ranking: list[int], k: int = 60, weight: float = 1.0):
    for position, idx in enumerate(ranking, start=1):
        score_map[int(idx)] += float(weight) / float(k + position)


def _safe_contains(series: pd.Series, value: str) -> pd.Series:
    if not value:
        return pd.Series(False, index=series.index)
    return series.fillna("").astype(str).str.contains(re.escape(value), case=False, na=False)


def _entity_key_set(raw_entities) -> set[str]:
    if not isinstance(raw_entities, list):
        return set()
    return {
        _normalize_key(value)
        for value in raw_entities
        if str(value).strip()
    }


def _build_search_cache(df: pd.DataFrame) -> dict[str, Any]:
    article_text: list[str] = []
    row_category_keys: list[str] = []
    row_subcategory_keys: list[str] = []
    row_source_keys: list[str] = []
    row_entity_sets: list[set[str]] = []
    token_index: dict[str, list[int]] = defaultdict(list)
    category_index: dict[str, list[int]] = defaultdict(list)
    subcategory_index: dict[str, list[int]] = defaultdict(list)
    source_index: dict[str, list[int]] = defaultdict(list)
    entity_index: dict[str, list[int]] = defaultdict(list)
    entity_display: dict[str, str] = {}
    row_records: list[dict[str, Any]] = []

    entity_series = df.get("entity_labels", pd.Series([[] for _ in range(len(df))], index=df.index))

    for idx, (news_id, title, abstract, category, subcategory, source, entity_labels, popularity) in enumerate(
        zip(
            df.get("news_id", pd.Series("", index=df.index)).fillna("").astype(str).tolist(),
            df.get("title", pd.Series("", index=df.index)).fillna("").astype(str).tolist(),
            df.get("abstract", pd.Series("", index=df.index)).fillna("").astype(str).tolist(),
            df.get("category", pd.Series("", index=df.index)).fillna("").astype(str).tolist(),
            df.get("subcategory", pd.Series("", index=df.index)).fillna("").astype(str).tolist(),
            df.get("source", pd.Series("", index=df.index)).fillna("").astype(str).tolist(),
            entity_series.tolist(),
            df.get("popularity", pd.Series(0.0, index=df.index)).fillna(0.0).tolist(),
        )
    ):
        category_key = _normalize_key(category)
        subcategory_key = _normalize_key(subcategory)
        source_key = _normalize_key(source)
        entity_keys = _entity_key_set(entity_labels)
        if isinstance(entity_labels, list):
            for value in entity_labels:
                normalized = _normalize_key(value)
                if normalized and normalized not in entity_display:
                    entity_display[normalized] = str(value)

        text = " ".join(
            value
            for value in (title, abstract, category_key, subcategory_key, source_key)
            if str(value).strip()
        ).lower()

        article_text.append(text)
        row_category_keys.append(category_key)
        row_subcategory_keys.append(subcategory_key)
        row_source_keys.append(source_key)
        row_entity_sets.append(entity_keys)
        row_records.append(
            {
                "news_id": str(news_id),
                "title": str(title),
                "abstract": str(abstract),
                "category": str(category),
                "subcategory": str(subcategory),
                "source": str(source),
                "entity_labels": list(entity_labels) if isinstance(entity_labels, list) else [],
                "popularity": float(popularity or 0.0),
            }
        )

        if category_key:
            category_index[category_key].append(idx)
        if subcategory_key:
            subcategory_index[subcategory_key].append(idx)
        if source_key:
            source_index[source_key].append(idx)
        for entity_key in entity_keys:
            entity_index[entity_key].append(idx)

        for token in set(_tokenize(text)):
            token_index[token].append(idx)

    return {
        "size": int(len(df)),
        "corpus_fingerprint": f"{len(df)}:{row_records[0]['news_id'] if row_records else ''}:{row_records[-1]['news_id'] if row_records else ''}",
        "article_text": article_text,
        "row_category_keys": row_category_keys,
        "row_subcategory_keys": row_subcategory_keys,
        "row_source_keys": row_source_keys,
        "row_entity_sets": row_entity_sets,
        "row_records": row_records,
        "token_index": dict(token_index),
        "category_index": dict(category_index),
        "subcategory_index": dict(subcategory_index),
        "source_index": dict(source_index),
        "entity_index": dict(entity_index),
        "entity_display": entity_display,
    }


def _clone_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(article) for article in candidates]


def _clone_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in diagnostics.items():
        if isinstance(value, dict):
            cloned[key] = dict(value)
        elif isinstance(value, list):
            cloned[key] = list(value)
        else:
            cloned[key] = value
    return cloned


def prepare_search_cache(df: pd.DataFrame) -> dict[str, Any]:
    cache = _SEARCH_CACHE_REGISTRY.get(id(df))
    if cache and int(cache.get("size", -1)) == len(df):
        return cache
    cache = _build_search_cache(df)
    _SEARCH_CACHE_REGISTRY[id(df)] = cache
    return cache


def _matches_query_constraints(
    idx: int,
    *,
    query_intent: dict[str, Any],
    cache: dict[str, Any],
) -> bool:
    normalized_query = str(query_intent.get("normalized_query") or "").strip().lower()
    tokens = set(query_intent.get("tokens") or [])
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
    matched_entities = {
        _normalize_key(value)
        for value in query_intent.get("matched_entities", [])
        if _normalize_key(value)
    }

    if not (normalized_query or tokens or matched_categories or matched_sources or matched_entities):
        return True

    text = cache["article_text"][idx]
    if normalized_query and normalized_query in text:
        return True
    if tokens and any(token in text for token in tokens):
        return True

    category_key = cache["row_category_keys"][idx]
    subcategory_key = cache["row_subcategory_keys"][idx]
    if matched_categories and (category_key in matched_categories or subcategory_key in matched_categories):
        return True

    source_key = cache["row_source_keys"][idx]
    if matched_sources and source_key in matched_sources:
        return True

    entity_keys = cache["row_entity_sets"][idx]
    if matched_entities and entity_keys.intersection(matched_entities):
        return True

    return False


def build_query_intent(query: str, df: pd.DataFrame, user=None) -> dict[str, Any]:
    raw_query = str(query or "").strip()
    lowered = raw_query.lower()
    tokens = _tokenize(raw_query)
    cache = prepare_search_cache(df)

    categories = sorted(cache.get("category_index", {}).keys())
    sources = sorted(cache.get("source_index", {}).keys())

    corrections = get_close_matches(lowered, categories + sources, n=1, cutoff=0.84)
    corrected_query = corrections[0] if corrections and len(tokens) <= 2 else lowered

    matched_categories = [category for category in categories if category and category in corrected_query]
    matched_sources = [source for source in sources if source and source in corrected_query]

    entity_counts: Counter[str] = Counter()
    for normalized, rows in cache.get("entity_index", {}).items():
        if normalized and normalized in corrected_query:
            entity_counts[cache.get("entity_display", {}).get(normalized, normalized)] += int(len(rows))

    return {
        "query": raw_query,
        "normalized_query": corrected_query,
        "tokens": tokens,
        "matched_categories": matched_categories[:4],
        "matched_sources": matched_sources[:4],
        "matched_entities": [entity for entity, _ in entity_counts.most_common(6)],
    }


def encode_hybrid_query_embedding(query: str, model: Any) -> np.ndarray | None:
    if not query:
        return None
    query_text = f"{_QUERY_PREFIX}{str(query).strip()}"
    return encode_query_embedding(query_text, model)


def lexical_search(query_intent: dict[str, Any], df: pd.DataFrame, limit: int = 150) -> list[int]:
    if len(df) == 0:
        return []

    cache = prepare_search_cache(df)
    query = str(query_intent.get("normalized_query") or "").strip()
    tokens = list(query_intent.get("tokens") or [])
    matched_categories = {
        _normalize_key(value)
        for value in query_intent.get("matched_categories", [])
    }
    matched_sources = {
        _normalize_key(value)
        for value in query_intent.get("matched_sources", [])
    }
    matched_entities = {
        _normalize_key(value)
        for value in query_intent.get("matched_entities", [])
    }

    doc_freq: Counter[str] = Counter()
    for token in dict.fromkeys(tokens):
        doc_freq[token] = len(cache["token_index"].get(token, []))

    candidate_indices: set[int] = set()
    for token in tokens:
        candidate_indices.update(cache["token_index"].get(token, []))
    for category in matched_categories:
        candidate_indices.update(cache["category_index"].get(category, []))
        candidate_indices.update(cache["subcategory_index"].get(category, []))
    for source in matched_sources:
        candidate_indices.update(cache["source_index"].get(source, []))
    for entity in matched_entities:
        candidate_indices.update(cache["entity_index"].get(entity, []))

    query_lower = query.lower()
    if query_lower:
        candidate_indices.update(
            idx
            for idx, text in enumerate(cache["article_text"])
            if query_lower in text
        )

    scores: dict[int, float] = {}
    for idx in candidate_indices:
        score = 0.0
        text = cache["article_text"][idx]
        entity_keys = cache["row_entity_sets"][idx]

        if query_lower and query_lower in text:
            score += 2.4

        category_key = cache["row_category_keys"][idx]
        subcategory_key = cache["row_subcategory_keys"][idx]
        source_key = cache["row_source_keys"][idx]
        if category_key in matched_categories or subcategory_key in matched_categories:
            score += 1.8
        if source_key in matched_sources:
            score += 1.4

        for token in tokens:
            tf = text.count(token)
            if tf <= 0:
                continue
            idf = math.log((1.0 + len(df)) / (1.0 + float(doc_freq[token]))) + 1.0
            score += tf * idf

        if matched_entities and entity_keys.intersection(matched_entities):
            score += 1.1 * float(len(entity_keys.intersection(matched_entities)))

        if score > 0.0:
            scores[idx] = score

    ranking = sorted(scores, key=lambda idx: scores[idx], reverse=True)
    return ranking[: max(int(limit), 0)]


def dense_search(
    query_intent: dict[str, Any],
    index,
    df: pd.DataFrame,
    model: Any,
    limit: int = 150,
) -> tuple[list[int], dict[int, float]]:
    if len(df) == 0 or index is None or model is None:
        return [], {}

    query_embedding = encode_hybrid_query_embedding(
        str(query_intent.get("normalized_query") or ""),
        model,
    )
    if query_embedding is None:
        return [], {}

    query_matrix = np.expand_dims(query_embedding.astype("float32"), axis=0)
    search_k = min(len(df), max(int(limit), 1))
    distances, indices = index.search(query_matrix, search_k)

    ranking: list[int] = []
    score_map: dict[int, float] = {}
    for score, idx in zip(distances[0], indices[0]):
        idx = int(idx)
        if idx < 0 or idx >= len(df):
            continue
        ranking.append(idx)
        score_map[idx] = float(score)
    return ranking, score_map


def opensearch_lexical_search(
    query_intent: dict[str, Any],
    news_id_to_idx: dict[str, int],
    *,
    search_backend: Any = None,
    limit: int = 150,
) -> tuple[list[int], dict[int, float]]:
    if search_backend is None:
        return [], {}
    try:
        return search_backend.search(query_intent, news_id_to_idx, limit=limit)
    except Exception:
        return [], {}


def qdrant_dense_search(
    query_embedding: np.ndarray | None,
    news_id_to_idx: dict[str, int],
    *,
    vector_backend: Any = None,
    limit: int = 150,
) -> tuple[list[int], dict[int, float]]:
    if vector_backend is None or query_embedding is None:
        return [], {}
    try:
        return vector_backend.search_articles(query_embedding, news_id_to_idx, limit=limit)
    except Exception:
        return [], {}


def memory_search(
    query_intent: dict[str, Any],
    df: pd.DataFrame,
    news_id_to_idx: dict[str, int],
    user=None,
    query_embedding: np.ndarray | None = None,
    vector_backend: Any = None,
    limit: int = 80,
) -> list[int]:
    if user is None or len(df) == 0:
        return []

    cache = prepare_search_cache(df)
    scored: Counter[int] = Counter()
    recent_clicks = list(getattr(user, "recent_clicks", [])[-25:])
    recent_entities = {_normalize_key(value) for value in getattr(user, "recent_entities", [])[-25:]}
    recent_sources = {_normalize_key(value) for value in getattr(user, "recent_sources", [])[-25:]}
    query_entities = {_normalize_key(value) for value in query_intent.get("matched_entities", [])}
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
    query_tokens = set(query_intent.get("tokens") or [])
    query_present = bool(
        str(query_intent.get("normalized_query") or "").strip()
        or query_tokens
        or query_entities
        or matched_categories
        or matched_sources
    )

    if vector_backend is not None and query_embedding is not None:
        try:
            memory_hits = vector_backend.search_user_memory(
                getattr(user, "user_id", ""),
                query_embedding,
                limit=min(max(int(limit // 2), 8), 24),
            )
        except Exception:
            memory_hits = []
        for rank, hit in enumerate(memory_hits):
            idx = news_id_to_idx.get(str(hit.get("article_id") or ""))
            if idx is None:
                continue
            if query_present and not _matches_query_constraints(idx, query_intent=query_intent, cache=cache):
                continue
            similarity = float(hit.get("similarity", 0.0))
            feedback_weight = float(hit.get("feedback_weight", 1.0))
            scored[int(idx)] += max(0.2, similarity * min(max(feedback_weight, 0.25), 2.0)) / float(1.0 + (rank * 0.25))

    for offset, article_id in enumerate(reversed(recent_clicks)):
        idx = news_id_to_idx.get(str(article_id))
        if idx is None:
            continue
        if query_present and not _matches_query_constraints(idx, query_intent=query_intent, cache=cache):
            continue
        scored[int(idx)] += max(0.08, 0.32 - (offset * 0.015))

    candidate_indices: set[int] = set()
    for source in recent_sources | matched_sources:
        candidate_indices.update(cache["source_index"].get(source, []))
    for entity in recent_entities | query_entities:
        candidate_indices.update(cache["entity_index"].get(entity, []))
    for category in matched_categories:
        candidate_indices.update(cache["category_index"].get(category, []))
        candidate_indices.update(cache["subcategory_index"].get(category, []))
    for token in query_tokens:
        candidate_indices.update(cache["token_index"].get(token, []))

    for idx in candidate_indices:
        if query_present and not _matches_query_constraints(idx, query_intent=query_intent, cache=cache):
            continue
        row_source = cache["row_source_keys"][idx]
        row_category = cache["row_category_keys"][idx]
        row_subcategory = cache["row_subcategory_keys"][idx]
        row_entities = cache["row_entity_sets"][idx]
        text = cache["article_text"][idx]

        if matched_sources and row_source in matched_sources:
            scored[idx] += 0.45
        if recent_sources and row_source in recent_sources:
            scored[idx] += 0.20
        if matched_categories and (row_category in matched_categories or row_subcategory in matched_categories):
            scored[idx] += 0.32
        if query_entities and row_entities.intersection(query_entities):
            scored[idx] += 0.65
        if recent_entities and row_entities.intersection(recent_entities):
            scored[idx] += 0.18
        if query_tokens:
            overlap = sum(1 for token in query_tokens if token in text)
            if overlap:
                scored[idx] += 0.18 * float(overlap)

    return [idx for idx, _ in scored.most_common(max(int(limit), 0))]


def kg_expansion_search(
    query_intent: dict[str, Any],
    df: pd.DataFrame,
    graph,
    news_id_to_idx: dict[str, int],
    seed_article_ids: list[str],
    limit: int = 60,
) -> list[int]:
    if graph is None:
        return []

    scored: Counter[int] = Counter()
    references: list[str] = list(seed_article_ids[:8])
    references.extend(query_intent.get("matched_categories", [])[:3])
    references.extend(query_intent.get("matched_entities", [])[:4])

    for reference in references:
        for rank, news_id in enumerate(get_related_articles(str(reference), graph, limit=30)):
            idx = news_id_to_idx.get(str(news_id))
            if idx is None:
                continue
            scored[int(idx)] += 1.0 / float(1 + rank)

    return [idx for idx, _ in scored.most_common(max(int(limit), 0))]


def reciprocal_rank_fusion(
    rankings: dict[str, list[int]],
    k: int = 60,
    limit: int = 240,
    source_weights: dict[str, float] | None = None,
) -> tuple[list[int], dict[int, float], dict[int, list[str]]]:
    scores: dict[int, float] = defaultdict(float)
    source_map: dict[int, list[str]] = defaultdict(list)
    source_weights = source_weights or {}
    for source_name, ranking in rankings.items():
        _rrf_add(scores, ranking, k=k, weight=float(source_weights.get(source_name, 1.0)))
        for idx in ranking:
            idx = int(idx)
            if source_name not in source_map[idx]:
                source_map[idx].append(source_name)

    ordered = sorted(scores, key=lambda idx: scores[idx], reverse=True)
    return ordered[: max(int(limit), 0)], dict(scores), dict(source_map)


def rerank_candidates(
    query: str,
    articles: list[dict],
    reranker: Any = None,
    top_k: int = 60,
) -> list[dict]:
    if not articles:
        return []

    selected = list(articles[: max(int(top_k), 0)])
    if reranker is None:
        query_tokens = set(_tokenize(query))
        for article in selected:
            text = " ".join(
                [
                    str(article.get("title", "")),
                    str(article.get("abstract", "")),
                    str(article.get("category", "")),
                    str(article.get("source", "")),
                ]
            ).lower()
            overlap = sum(1 for token in query_tokens if token in text)
            article["rerank_score"] = float(article.get("retrieval_score", 0.0)) + (0.12 * overlap)
        return sorted(selected, key=lambda item: item.get("rerank_score", 0.0), reverse=True)

    pairs = [
        (query, " ".join([str(article.get("title", "")), str(article.get("abstract", ""))]).strip())
        for article in selected
    ]
    try:
        scores = reranker.predict(pairs)
        for article, score in zip(selected, scores):
            article["rerank_score"] = float(score)
        return sorted(selected, key=lambda item: item.get("rerank_score", 0.0), reverse=True)
    except Exception:
        return rerank_candidates(query, selected, reranker=None, top_k=top_k)


def build_hybrid_candidates(
    user,
    query: str,
    df: pd.DataFrame,
    index,
    model: Any,
    graph,
    news_id_to_idx: dict[str, int],
    reranker: Any = None,
    search_backend: Any = None,
    vector_backend: Any = None,
    limit: int = 120,
    rerank_k: int = 60,
) -> tuple[list[dict], dict[str, Any]]:
    cache = prepare_search_cache(df)
    query_intent = build_query_intent(query, df, user=user)
    cache_key = (
        str(query_intent.get("normalized_query") or "").strip().lower(),
        str(cache.get("corpus_fingerprint") or ""),
        int(limit),
        int(rerank_k),
    )
    cached = _EXPLICIT_SEARCH_CACHE.get(cache_key)
    if cached and (time.time() - float(cached[0])) <= _EXPLICIT_SEARCH_CACHE_TTL_SECONDS:
        cached_candidates, cached_diagnostics = cached[1], cached[2]
        return _clone_candidates(cached_candidates), _clone_diagnostics(cached_diagnostics)

    query_embedding = encode_hybrid_query_embedding(
        str(query_intent.get("normalized_query") or ""),
        model,
    )
    os_lexical_ranking, os_lexical_scores = opensearch_lexical_search(
        query_intent,
        news_id_to_idx,
        search_backend=search_backend,
        limit=150,
    )
    lexical_ranking = os_lexical_ranking or lexical_search(query_intent, df, limit=150)
    lexical_scores = os_lexical_scores or {}
    qdrant_dense_ranking, qdrant_dense_scores = qdrant_dense_search(
        query_embedding,
        news_id_to_idx,
        vector_backend=vector_backend,
        limit=150,
    )
    dense_ranking, dense_scores = dense_search(
        query_intent,
        index,
        df,
        model,
        limit=150,
    )
    if qdrant_dense_ranking:
        dense_ranking = qdrant_dense_ranking
        dense_scores = qdrant_dense_scores

    seed_article_ids = [
        str(df.iloc[idx]["news_id"])
        for idx in (dense_ranking[:8] + lexical_ranking[:8])
        if 0 <= int(idx) < len(df)
    ]
    kg_ranking = kg_expansion_search(
        query_intent,
        df,
        graph,
        news_id_to_idx,
        seed_article_ids=seed_article_ids,
        limit=60,
    )

    fused_ids, fused_scores, source_map = reciprocal_rank_fusion(
        {
            "lexical": lexical_ranking,
            "dense": dense_ranking,
            "kg_expand": kg_ranking,
        },
        k=60,
        limit=max(limit, rerank_k),
        source_weights={
            "lexical": 1.35,
            "dense": 1.20,
            "kg_expand": 0.85,
        },
    )

    candidates: list[dict] = []
    for idx in fused_ids[: max(limit, rerank_k)]:
        if idx < 0 or idx >= len(df):
            continue
        article = dict(cache["row_records"][int(idx)])
        article["retrieval_score"] = float(fused_scores.get(int(idx), 0.0))
        article["dense_score"] = float(dense_scores.get(int(idx), 0.0))
        article["lexical_score"] = float(lexical_scores.get(int(idx), 0.0))
        article["candidate_source"] = "+".join(source_map.get(int(idx), [])[:3]) or "hybrid"
        article["matched_entities"] = list(query_intent.get("matched_entities", [])[:4])
        article["matched_categories"] = list(query_intent.get("matched_categories", [])[:3])
        article["reasons"] = [
            f"Matches search: {query_intent['normalized_query']}",
            *[f"Connected via {entity}" for entity in article["matched_entities"][:2]],
        ]
        candidates.append(article)

    reranked = rerank_candidates(query_intent["normalized_query"], candidates, reranker=reranker, top_k=rerank_k)
    seen_ids: set[str] = set()
    final_candidates: list[dict] = []
    for article in reranked + candidates:
        news_id = str(article.get("news_id") or "")
        if not news_id or news_id in seen_ids:
            continue
        seen_ids.add(news_id)
        final_candidates.append(article)
        if len(final_candidates) >= max(int(limit), 0):
            break

    diagnostics = {
        "query_intent": query_intent,
        "source_counts": {
            "lexical": len(lexical_ranking),
            "dense": len(dense_ranking),
            "kg_expand": len(kg_ranking),
        },
        "backend_sources": {
            "opensearch": bool(os_lexical_ranking),
            "qdrant": bool(qdrant_dense_ranking),
        },
        "fused_candidates": len(final_candidates),
    }
    _EXPLICIT_SEARCH_CACHE[cache_key] = (
        time.time(),
        _clone_candidates(final_candidates),
        _clone_diagnostics(diagnostics),
    )
    return final_candidates, diagnostics


def suggest_queries(
    prefix: str,
    df: pd.DataFrame,
    user=None,
    recent_queries: list[str] | None = None,
    limit: int = 10,
    search_backend: Any = None,
) -> list[str]:
    value = str(prefix or "").strip().lower()
    if recent_queries is None:
        recent_queries = list(getattr(user, "recent_queries", [])[-25:]) if user is not None else []
    if not value:
        return list(dict.fromkeys(list(recent_queries)[-10:]))[-limit:][::-1]

    suggestions: list[str] = []

    def add(candidate: str):
        normalized = str(candidate or "").strip()
        if normalized and normalized.lower().startswith(value) and normalized not in suggestions:
            suggestions.append(normalized)

    if search_backend is not None:
        for suggestion in search_backend.suggest(value, limit=limit):
            add(suggestion)
            if len(suggestions) >= limit:
                return suggestions[:limit]

    for query in reversed(list(recent_queries)):
        add(query)

    for category in df.get("category", pd.Series(dtype=str)).dropna().astype(str).unique().tolist():
        add(category)
    for source in df.get("source", pd.Series(dtype=str)).dropna().astype(str).unique().tolist():
        add(source)

    if "entity_labels" in df.columns:
        for labels in df["entity_labels"].tolist():
            if not isinstance(labels, list):
                continue
            for label in labels[:6]:
                add(str(label))
                if len(suggestions) >= limit:
                    return suggestions[:limit]

    for title in df.get("title", pd.Series(dtype=str)).dropna().astype(str).tolist()[:500]:
        add(title)
        if len(suggestions) >= limit:
            break

    return suggestions[:limit]
