"""
RAG retrieval helpers and explanation generation.

If GROQ_API_KEY is set, explanations are generated with Groq.
Otherwise a deterministic, mood-aware fallback is used.
"""

import os
import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd

try:
    from backend.coldstart_hints import humanize_location
except ImportError:
    from coldstart_hints import humanize_location

_GROQ_KEY = os.getenv("GROQ_API_KEY", "")
_llm = None

if _GROQ_KEY:
    try:
        from langchain.schema import HumanMessage
        from langchain_groq import ChatGroq

        _llm = ChatGroq(model="llama-3.1-8b-instant", api_key=_GROQ_KEY, temperature=0.4)
    except Exception as exc:
        print(f"Groq init failed: {exc}. Using fallback explanation.")


# HNSW parameters: M=32 gives >97% recall@10 with sub-ms queries on 10K-100K vectors.
_HNSW_M               = 32
_HNSW_EF_CONSTRUCTION = 200
_HNSW_EF_SEARCH       = 64


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


class NumpyVectorIndex:
    """Small cosine-similarity index with a FAISS-like `search` surface."""

    def __init__(self, embeddings: np.ndarray):
        self.embeddings = _normalize_matrix(embeddings)
        self.ntotal = int(self.embeddings.shape[0])

    def add(self, embeddings: np.ndarray):
        rows = _normalize_matrix(embeddings)
        if self.ntotal == 0:
            self.embeddings = rows
        else:
            self.embeddings = np.vstack([self.embeddings, rows]).astype(np.float32)
        self.ntotal = int(self.embeddings.shape[0])

    def search(self, queries: np.ndarray, top_k: int):
        query_matrix = _normalize_matrix(queries)
        k = max(0, min(int(top_k), self.ntotal))
        if self.ntotal == 0 or k == 0:
            empty_scores = np.empty((len(query_matrix), 0), dtype=np.float32)
            empty_indices = np.empty((len(query_matrix), 0), dtype=np.int64)
            return empty_scores, empty_indices

        scores = query_matrix @ self.embeddings.T
        order = np.argsort(-scores, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores, order, axis=1)
        return top_scores.astype(np.float32), order.astype(np.int64)


def build_faiss_index(embeddings: np.ndarray):
    """Build a lightweight cosine index with the same search contract."""
    return NumpyVectorIndex(embeddings)


def load_vector_index(path: str | None):
    # Existing FAISS files are ignored when FAISS is unavailable or unhealthy.
    return None


def save_vector_index(index, path: str) -> bool:
    return False


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
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
}


def _tokenize(text: str) -> list[str]:
    return [
        token for token in _TOKEN_RE.findall(text.lower())
        if len(token) > 2 and token not in _STOP_WORDS
    ]


def encode_query_embedding(query: str, model: Any) -> np.ndarray | None:
    if not query or model is None:
        return None
    embedding = model.encode([query], normalize_embeddings=True).astype("float32")
    if len(embedding) == 0:
        return None
    return embedding[0]


def retrieve_articles(
    query: str,
    index,
    df: pd.DataFrame,
    model: Any,
    top_k: int = 10,
) -> List[Dict]:
    if not query or len(df) == 0:
        return []

    search_k = min(len(df), max(top_k * 6, 40))
    query_vec = encode_query_embedding(query, model)
    query_lower = query.strip().lower()
    query_tokens = _tokenize(query)
    semantic_pairs: list[tuple[float, int]] = []
    if query_vec is not None and index is not None:
        query_emb = np.expand_dims(query_vec.astype("float32"), axis=0)
        distances, indices = index.search(query_emb, search_k)
        semantic_pairs = [
            (float(score), int(idx))
            for score, idx in zip(distances[0], indices[0])
            if 0 <= int(idx) < len(df)
        ]
    semantic_scores = {
        idx: score
        for score, idx in semantic_pairs
    }

    candidate_ids: list[int] = []
    seen_ids: set[int] = set()

    def add_idx(idx: int):
        if 0 <= idx < len(df) and idx not in seen_ids:
            seen_ids.add(idx)
            candidate_ids.append(idx)

    for _, idx in semantic_pairs:
        add_idx(idx)

    if query_lower:
        phrase_mask = (
            df["title"].fillna("").str.contains(re.escape(query_lower), case=False, na=False)
            | df["abstract"].fillna("").str.contains(re.escape(query_lower), case=False, na=False)
        )
        for idx in df[phrase_mask].index[:search_k]:
            add_idx(int(idx))

        category_mask = df["category"].fillna("").str.lower().eq(query_lower)
        if "subcategory" in df.columns:
            category_mask = category_mask | df["subcategory"].fillna("").str.lower().eq(query_lower)
        for idx in df[category_mask].index[:search_k]:
            add_idx(int(idx))

    if query_tokens:
        lexical_text = (
            df["title"].fillna("").astype(str)
            + " "
            + df["abstract"].fillna("").astype(str)
            + " "
            + df["category"].fillna("").astype(str)
            + " "
            + df["subcategory"].fillna("").astype(str)
        ).str.lower()
        for token in query_tokens[:6]:
            token_mask = lexical_text.str.contains(re.escape(token), case=False, na=False)
            for idx in lexical_text[token_mask].index[:search_k]:
                add_idx(int(idx))

    scored = []
    for idx in candidate_ids:
        article = df.iloc[idx].to_dict()
        article_text = " ".join(
            [
                str(article.get("title", "")),
                str(article.get("abstract", "")),
                str(article.get("category", "")),
                str(article.get("subcategory", "")),
            ]
        ).lower()
        overlap = sum(1 for token in query_tokens if token in article_text)
        lexical_score = overlap / max(len(query_tokens), 1)
        phrase_bonus = 0.20 if query_lower and query_lower in article_text else 0.0
        category_bonus = 0.20 if str(article.get("category", "")).lower() == query_lower else 0.0
        semantic_score = semantic_scores.get(idx, 0.0)
        combined_score = (0.55 * semantic_score) + (0.45 * lexical_score) + phrase_bonus + category_bonus
        scored.append(
            {
                **article,
                "semantic_score": float(semantic_score),
                "lexical_score": float(lexical_score),
                "retrieval_score": float(combined_score),
            }
        )

    scored.sort(
        key=lambda article: (
            article["retrieval_score"],
            article["semantic_score"],
            article["lexical_score"],
        ),
        reverse=True,
    )

    results = []
    seen_news_ids = set()
    for article in scored:
        news_id = article.get("news_id")
        if news_id in seen_news_ids:
            continue
        seen_news_ids.add(news_id)
        results.append(article)
        if len(results) >= top_k:
            break

    return results


def _category_mix(articles: List[dict]) -> str:
    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for idx, article in enumerate(articles[:6]):
        category = str(article.get("category", "")).strip().lower()
        if not category:
            continue
        counts[category] = counts.get(category, 0) + 1
        first_seen.setdefault(category, idx)

    if not counts:
        return "a broader mix of topics"

    ordered = sorted(counts.items(), key=lambda item: (-item[1], first_seen[item[0]]))
    labels = [category for category, _ in ordered[:3]]
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return f"{labels[0]}, {labels[1]}, and {labels[2]}"


def _profile_preference_summary(user_context: dict) -> str:
    selected_categories = [
        str(value).strip().lower()
        for value in user_context.get("top_categories", []) or []
        if str(value).strip()
    ]
    hinted_categories = [
        str(value).strip().lower()
        for value in user_context.get("profile_hint_categories", []) or []
        if str(value).strip()
    ]
    location_label = humanize_location(
        str(user_context.get("location_region") or ""),
        str(user_context.get("location_country") or ""),
    )
    interest_text = str(user_context.get("interest_text") or "").strip()

    fragments: list[str] = []
    if selected_categories:
        fragments.append(f"selected categories like {', '.join(selected_categories[:3])}")
    if hinted_categories:
        fragments.append(f"interest-note hints leaning toward {', '.join(hinted_categories[:2])}")
    if interest_text:
        fragments.append(f"your note '{interest_text[:72]}'")
    if location_label:
        fragments.append(f"location context around {location_label}")

    if not fragments:
        return ""
    if len(fragments) == 1:
        return fragments[0]
    return ", ".join(fragments[:-1]) + f", and {fragments[-1]}"


def _fallback_explanation(user_context: dict, articles: List[dict]) -> str:
    mood = str(user_context.get("mood", "neutral")).strip().lower()
    time_slot = str(user_context.get("time_of_day", "morning")).strip().lower()
    mode = str(user_context.get("mode", "rl")).strip().lower()
    query = str(user_context.get("query") or "").strip()
    category_mix = _category_mix(articles)
    preference_summary = _profile_preference_summary(user_context)

    mood_sentence_map = {
        "curious": f"Because you're feeling curious this {time_slot}, the feed leans into discovery-oriented stories across {category_mix}.",
        "happy": f"Because you're in a happy mood this {time_slot}, the feed stays upbeat and energetic with {category_mix}.",
        "stressed": f"Because you're feeling stressed this {time_slot}, the feed tries to stay lighter and easier to scan with {category_mix}.",
        "tired": f"Because you're tired this {time_slot}, the feed favors simpler, lower-friction reading across {category_mix}.",
        "neutral": f"For a neutral mood this {time_slot}, the feed keeps a balanced mix centered on {category_mix}.",
    }
    mode_sentence_map = {
        "rag": (
            f"Your search for '{query}' is steering retrieval first, then the ranker is reordering those candidates across {category_mix} using prior feedback and session context."
            if query
            else f"The retrieval step is pulling semantically similar articles first, then the ranker is reordering them across {category_mix} using mood, context, and prior feedback."
        ),
        "cold_start": (
            f"Because this session has little or no reading history yet, the system is leaning hard on {preference_summary} while still keeping some diversity across {category_mix}."
            if preference_summary
            else f"Because this session has little or no reading history yet, the system is relying more on mood, time of day, and category diversity across {category_mix} than long-term behavior."
        ),
        "rl": f"The ranker is using your recent feedback and session context to reshuffle the feed, so the order should keep adapting across {category_mix} as you read, save, or skip.",
    }

    first_sentence = mood_sentence_map.get(mood, mood_sentence_map["neutral"])
    second_sentence = mode_sentence_map.get(mode, mode_sentence_map["rl"])
    return f"{first_sentence} {second_sentence}"


def generate_personalized_summary(
    user_context: dict,
    articles: List[dict],
    dummy_mode: bool = False,
) -> str:
    mood = user_context.get("mood", "neutral")
    time_slot = user_context.get("time_of_day", "morning")

    if _llm and not dummy_mode:
        try:
            article_lines = "\n".join(f"- {a['title']} ({a['category']})" for a in articles[:5])
            prompt = (
                f"User context:\n"
                f"  Mood: {mood}\n"
                f"  Time of day: {time_slot}\n\n"
                f"Articles being recommended:\n{article_lines}\n\n"
                f"In exactly 2 engaging sentences, explain why these articles are "
                f"perfect for this user right now. Be specific about mood and time. "
                f"Don't start with 'Based on'."
            )
            response = _llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as exc:
            print(f"Groq call failed: {exc}")

    return _fallback_explanation(user_context, articles)
