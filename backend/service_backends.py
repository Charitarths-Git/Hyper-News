"""
Optional production retrieval backends for OpenSearch and Qdrant.

These adapters are intentionally defensive: when the external service is not
reachable, the caller still gets a working in-process fallback path.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd


def _is_enabled(value: str, default: str = "1") -> bool:
    normalized = str(value or default).strip().lower()
    return normalized in {"1", "true", "yes", "on"}


def _normalize_key(value: str) -> str:
    return str(value or "").strip().lower()


class OpenSearchArticleStore:
    def __init__(
        self,
        *,
        url: str | None = None,
        index_name: str | None = None,
        enabled: bool | None = None,
    ):
        self.url = str(url or os.getenv("OPENSEARCH_URL", "http://localhost:9200")).strip()
        self.index_name = str(index_name or os.getenv("HYPERNEWS_OPENSEARCH_INDEX", "hypernews_articles")).strip()
        self.enabled = _is_enabled(
            os.getenv("HYPERNEWS_ENABLE_OPENSEARCH", "1"),
            default="1",
        ) if enabled is None else bool(enabled)
        self.client = None
        self.available = False
        self.last_error = ""
        self.indexed_count = 0

        if not self.enabled:
            return
        try:
            from opensearchpy import OpenSearch

            self.client = OpenSearch(
                hosts=[self.url],
                http_compress=True,
                use_ssl=self.url.startswith("https://"),
                verify_certs=False,
                ssl_assert_hostname=False,
                ssl_show_warn=False,
                timeout=5,
            )
            self.available = bool(self.client.ping())
            if not self.available:
                self.last_error = "ping failed"
        except Exception as exc:
            self.available = False
            self.last_error = str(exc)

    def ensure_index(self) -> bool:
        if not self.available or self.client is None:
            return False
        if self.client.indices.exists(index=self.index_name):
            return True

        body = {
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                }
            },
            "mappings": {
                "properties": {
                    "news_id": {"type": "keyword"},
                    "title": {"type": "text"},
                    "abstract": {"type": "text"},
                    "category": {"type": "keyword"},
                    "subcategory": {"type": "keyword"},
                    "source": {"type": "keyword"},
                    "entity_labels": {"type": "keyword"},
                    "popularity": {"type": "float"},
                }
            },
        }
        try:
            self.client.indices.create(index=self.index_name, body=body)
            return True
        except Exception as exc:
            self.last_error = str(exc)
            return False

    def sync_articles(self, df: pd.DataFrame) -> dict[str, Any]:
        if not self.ensure_index() or self.client is None:
            return {"available": False, "indexed": 0, "error": self.last_error}

        try:
            current = int(self.client.count(index=self.index_name).get("count", 0))
            if current == len(df) and current > 0:
                self.indexed_count = current
                return {"available": True, "indexed": current, "skipped": True}
        except Exception:
            pass

        try:
            from opensearchpy import helpers

            actions = []
            for record in df.to_dict("records"):
                actions.append(
                    {
                        "_index": self.index_name,
                        "_id": str(record.get("news_id", "")),
                        "_source": {
                            "news_id": str(record.get("news_id", "")),
                            "title": str(record.get("title", "")),
                            "abstract": str(record.get("abstract", "")),
                            "category": str(record.get("category", "")),
                            "subcategory": str(record.get("subcategory", "")),
                            "source": str(record.get("source", "")),
                            "entity_labels": [
                                str(value)
                                for value in (record.get("entity_labels") or [])
                                if str(value).strip()
                            ][:16],
                            "popularity": float(record.get("popularity", 0.0) or 0.0),
                        },
                    }
                )
            helpers.bulk(self.client, actions, raise_on_error=False)
            self.client.indices.refresh(index=self.index_name)
            self.indexed_count = len(df)
            return {"available": True, "indexed": len(df), "skipped": False}
        except Exception as exc:
            self.last_error = str(exc)
            return {"available": False, "indexed": 0, "error": self.last_error}

    def search(
        self,
        query_intent: dict[str, Any],
        news_id_to_idx: dict[str, int],
        *,
        limit: int = 150,
    ) -> tuple[list[int], dict[int, float]]:
        if not self.available or self.client is None:
            return [], {}

        query_text = str(
            query_intent.get("normalized_query")
            or query_intent.get("query")
            or ""
        ).strip()
        if not query_text:
            return [], {}

        should = []
        for category in query_intent.get("matched_categories", [])[:4]:
            should.append({"term": {"category": _normalize_key(category)}})
        for source in query_intent.get("matched_sources", [])[:4]:
            should.append({"term": {"source": _normalize_key(source)}})
        for entity in query_intent.get("matched_entities", [])[:6]:
            should.append({"term": {"entity_labels": str(entity)}})

        body = {
            "size": max(int(limit), 1),
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query_text,
                                "fields": [
                                    "title^4",
                                    "abstract^2",
                                    "category^2",
                                    "subcategory^1.5",
                                    "source^1.5",
                                    "entity_labels^2",
                                ],
                                "type": "most_fields",
                            }
                        }
                    ],
                    "should": should,
                }
            },
        }

        try:
            response = self.client.search(index=self.index_name, body=body)
        except Exception as exc:
            self.last_error = str(exc)
            return [], {}

        ranking: list[int] = []
        scores: dict[int, float] = {}
        for hit in response.get("hits", {}).get("hits", []):
            payload = hit.get("_source", {})
            news_id = str(payload.get("news_id") or hit.get("_id") or "")
            idx = news_id_to_idx.get(news_id)
            if idx is None:
                continue
            ranking.append(int(idx))
            scores[int(idx)] = float(hit.get("_score", 0.0))
        return ranking, scores

    def suggest(self, prefix: str, *, limit: int = 10) -> list[str]:
        if not self.available or self.client is None:
            return []

        value = str(prefix or "").strip()
        if not value:
            return []

        body = {
            "size": max(int(limit) * 3, 10),
            "_source": ["title", "category", "source", "entity_labels"],
            "query": {
                "multi_match": {
                    "query": value,
                    "fields": ["title^3", "category^2", "source^2", "entity_labels^2"],
                    "type": "bool_prefix",
                }
            },
        }
        try:
            response = self.client.search(index=self.index_name, body=body)
        except Exception as exc:
            self.last_error = str(exc)
            return []

        suggestions: list[str] = []
        seen: set[str] = set()

        def add(candidate: str):
            normalized = str(candidate or "").strip()
            if (
                normalized
                and normalized.lower().startswith(value.lower())
                and normalized not in seen
            ):
                seen.add(normalized)
                suggestions.append(normalized)

        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            add(str(source.get("title", "")))
            add(str(source.get("category", "")))
            add(str(source.get("source", "")))
            for label in source.get("entity_labels", [])[:6]:
                add(str(label))
                if len(suggestions) >= limit:
                    return suggestions[:limit]
        return suggestions[:limit]

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "available": self.available,
            "index_name": self.index_name,
            "indexed_count": self.indexed_count,
            "error": self.last_error,
        }


class QdrantVectorStore:
    def __init__(
        self,
        *,
        url: str | None = None,
        article_collection: str | None = None,
        user_collection: str | None = None,
        enabled: bool | None = None,
    ):
        self.url = str(url or os.getenv("QDRANT_URL", "http://localhost:6333")).strip()
        self.article_collection = str(
            article_collection or os.getenv("HYPERNEWS_QDRANT_ARTICLE_COLLECTION", "hypernews_articles")
        ).strip()
        self.user_collection = str(
            user_collection or os.getenv("HYPERNEWS_QDRANT_USER_COLLECTION", "hypernews_user_memory")
        ).strip()
        self.enabled = _is_enabled(
            os.getenv("HYPERNEWS_ENABLE_QDRANT", "1"),
            default="1",
        ) if enabled is None else bool(enabled)
        self.client = None
        self.available = False
        self.last_error = ""
        self.article_count = 0
        self.user_memory_count = 0

        if not self.enabled:
            return
        try:
            from qdrant_client import QdrantClient

            self.client = QdrantClient(url=self.url, timeout=5)
            self.client.get_collections()
            self.available = True
        except Exception as exc:
            self.available = False
            self.last_error = str(exc)

    def _search_collection(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        *,
        limit: int,
        query_filter=None,
    ):
        if self.client is None:
            return []
        vector = np.asarray(query_vector, dtype=np.float32).tolist()
        search_kwargs = {
            "collection_name": collection_name,
            "limit": max(int(limit), 1),
            "with_payload": True,
        }
        if query_filter is not None:
            search_kwargs["query_filter"] = query_filter
        try:
            return self.client.search(
                query_vector=vector,
                **search_kwargs,
            )
        except TypeError:
            pass
        try:
            response = self.client.query_points(
                query=vector,
                **search_kwargs,
            )
            return getattr(response, "points", response)
        except Exception as exc:
            self.last_error = str(exc)
            return []

    def ensure_collections(self, vector_size: int) -> bool:
        if not self.available or self.client is None:
            return False
        try:
            from qdrant_client.http import models

            existing = {
                collection.name
                for collection in self.client.get_collections().collections
            }
            params = models.VectorParams(size=int(vector_size), distance=models.Distance.COSINE)
            if self.article_collection not in existing:
                self.client.create_collection(
                    collection_name=self.article_collection,
                    vectors_config=params,
                )
            if self.user_collection not in existing:
                self.client.create_collection(
                    collection_name=self.user_collection,
                    vectors_config=params,
                )
            return True
        except Exception as exc:
            self.last_error = str(exc)
            return False

    def sync_article_embeddings(self, df: pd.DataFrame, embeddings: np.ndarray, *, batch_size: int = 256) -> dict[str, Any]:
        if not self.ensure_collections(int(embeddings.shape[1])) or self.client is None:
            return {"available": False, "indexed": 0, "error": self.last_error}

        try:
            info = self.client.get_collection(self.article_collection)
            current = int(getattr(info, "points_count", 0) or 0)
            if current == len(df) and current > 0:
                self.article_count = current
                return {"available": True, "indexed": current, "skipped": True}
        except Exception:
            pass

        try:
            from qdrant_client.http import models

            total = len(df)
            for start in range(0, total, max(int(batch_size), 1)):
                end = min(start + max(int(batch_size), 1), total)
                points = []
                frame = df.iloc[start:end]
                batch_embeddings = np.asarray(embeddings[start:end], dtype=np.float32)
                for record, vector in zip(frame.to_dict("records"), batch_embeddings):
                    payload = {
                        "news_id": str(record.get("news_id", "")),
                        "title": str(record.get("title", ""))[:512],
                        "category": str(record.get("category", "")),
                        "subcategory": str(record.get("subcategory", "")),
                        "source": str(record.get("source", "")),
                        "entity_labels": [
                            str(value)
                            for value in (record.get("entity_labels") or [])
                            if str(value).strip()
                        ][:16],
                        "popularity": float(record.get("popularity", 0.0) or 0.0),
                    }
                    points.append(
                        models.PointStruct(
                            id=str(record.get("news_id", "")),
                            vector=np.asarray(vector, dtype=np.float32).tolist(),
                            payload=payload,
                        )
                    )
                self.client.upsert(
                    collection_name=self.article_collection,
                    wait=False,
                    points=points,
                )
            self.article_count = total
            return {"available": True, "indexed": total, "skipped": False}
        except Exception as exc:
            self.last_error = str(exc)
            return {"available": False, "indexed": 0, "error": self.last_error}

    def search_articles(
        self,
        query_vector: np.ndarray,
        news_id_to_idx: dict[str, int],
        *,
        limit: int = 150,
    ) -> tuple[list[int], dict[int, float]]:
        if not self.available or self.client is None:
            return [], {}

        try:
            results = self._search_collection(
                self.article_collection,
                query_vector,
                limit=limit,
            )
        except Exception:
            return [], {}

        ranking: list[int] = []
        scores: dict[int, float] = {}
        for point in results:
            payload = getattr(point, "payload", {}) or {}
            news_id = str(payload.get("news_id") or getattr(point, "id", "") or "")
            idx = news_id_to_idx.get(news_id)
            if idx is None:
                continue
            ranking.append(int(idx))
            scores[int(idx)] = float(getattr(point, "score", 0.0))
        return ranking, scores

    def upsert_user_memory(
        self,
        *,
        user_id: str,
        article_id: str,
        vector: np.ndarray,
        category: str = "",
        source: str = "",
        feedback_weight: float = 1.0,
    ) -> bool:
        if not self.ensure_collections(int(np.asarray(vector).shape[0])) or self.client is None:
            return False
        try:
            from qdrant_client.http import models

            point = models.PointStruct(
                id=f"{user_id}:{article_id}",
                vector=np.asarray(vector, dtype=np.float32).tolist(),
                payload={
                    "user_id": str(user_id),
                    "article_id": str(article_id),
                    "category": str(category or ""),
                    "source": str(source or ""),
                    "feedback_weight": float(feedback_weight or 1.0),
                },
            )
            self.client.upsert(
                collection_name=self.user_collection,
                wait=False,
                points=[point],
            )
            self.user_memory_count += 1
            return True
        except Exception as exc:
            self.last_error = str(exc)
            return False

    def search_user_memory(
        self,
        user_id: str,
        query_vector: np.ndarray,
        *,
        limit: int = 24,
    ) -> list[dict[str, Any]]:
        if not self.available or self.client is None:
            return []
        try:
            from qdrant_client.http import models

            results = self._search_collection(
                self.user_collection,
                query_vector,
                limit=limit,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="user_id",
                            match=models.MatchValue(value=str(user_id)),
                        )
                    ]
                ),
            )
        except Exception as exc:
            self.last_error = str(exc)
            return []

        hits: list[dict[str, Any]] = []
        for point in results:
            payload = getattr(point, "payload", {}) or {}
            hits.append(
                {
                    "article_id": str(payload.get("article_id", "")),
                    "similarity": float(getattr(point, "score", 0.0)),
                    "feedback_weight": float(payload.get("feedback_weight", 1.0) or 1.0),
                    "category": str(payload.get("category", "")),
                    "source": str(payload.get("source", "")),
                }
            )
        return hits

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "available": self.available,
            "article_collection": self.article_collection,
            "user_collection": self.user_collection,
            "article_count": self.article_count,
            "user_memory_count": self.user_memory_count,
            "error": self.last_error,
        }
