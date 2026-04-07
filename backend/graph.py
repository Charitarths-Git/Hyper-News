"""
Knowledge graph helpers for article, category, subcategory, and entity relations.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from collections import Counter
from typing import Iterable

import networkx as nx
import pandas as pd

try:
    from backend.mind_data import parse_entity_list
except ImportError:
    from mind_data import parse_entity_list

_KG_PATH = os.path.join(os.path.dirname(__file__), "..", "graph", "knowledge_graph.pkl")
_GRAPH_VERSION = 2
_NOISY_ENTITY_LABELS = {
    "this",
    "that",
    "these",
    "those",
    "from",
    "with",
    "what",
    "when",
    "where",
    "after",
    "before",
    "best",
    "your",
    "their",
    "our",
}

try:
    import spacy

    _nlp = spacy.load("en_core_web_sm")
    _SPACY_OK = True
except (OSError, ImportError):
    _SPACY_OK = False
    _nlp = None


def _normalize_key(value: str) -> str:
    return str(value or "").strip().lower()


def _article_node_id(news_id: str) -> str:
    return f"article::{news_id}"


def _category_node_id(category: str) -> str:
    return f"category::{_normalize_key(category)}"


def _subcategory_node_id(category: str, subcategory: str) -> str:
    return f"subcategory::{_normalize_key(category)}::{_normalize_key(subcategory)}"


def _entity_node_id(entity: dict) -> str:
    stable_id = entity.get("wikidata_id") or entity.get("id") or entity.get("label")
    return f"entity::{_normalize_key(stable_id)}"


def _extract_entities_spacy(text: str) -> list[dict]:
    if _SPACY_OK and _nlp is not None:
        doc = _nlp(text)
        return [
            {
                "id": ent.text.strip().lower(),
                "label": ent.text.strip(),
                "type": ent.label_,
                "wikidata_id": None,
                "confidence": 0.5,
                "surface_forms": [ent.text.strip()],
            }
            for ent in doc.ents
            if ent.label_ in ("ORG", "PERSON", "GPE", "EVENT", "PRODUCT", "WORK_OF_ART")
            and len(ent.text.strip()) > 2
        ]

    return [
        {
            "id": word.strip().lower(),
            "label": word.strip(),
            "type": "ENTITY",
            "wikidata_id": None,
            "confidence": 0.25,
            "surface_forms": [word.strip()],
        }
        for word in text.split()
        if word.istitle() and len(word.strip()) > 3
    ]


def _coerce_entities(value, title: str = "", abstract: str = "") -> list[dict]:
    entities = parse_entity_list(value)
    if entities:
        return entities
    text = f"{title} {abstract}".strip()
    return _extract_entities_spacy(text) if text else []


def _dataset_signature(df: pd.DataFrame) -> str:
    material = "|".join(df["news_id"].astype(str).tolist())
    return hashlib.sha1(material.encode("utf-8")).hexdigest()


def _load_cached_graph(cache_path: str, signature: str):
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, "rb") as handle:
        graph = pickle.load(handle)
    meta = getattr(graph, "graph", {})
    if meta.get("version") != _GRAPH_VERSION:
        return None
    if meta.get("dataset_signature") != signature:
        return None
    return graph


def build_knowledge_graph(
    df: pd.DataFrame,
    force_rebuild: bool = False,
    cache_path: str | None = None,
) -> nx.Graph:
    cache_path = cache_path or _KG_PATH
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    signature = _dataset_signature(df)
    if not force_rebuild:
        cached = _load_cached_graph(cache_path, signature)
        if cached is not None:
            print("Loading Knowledge Graph from disk...")
            return cached

    print(f"Building structured Knowledge Graph ({len(df):,} articles)...")
    graph = nx.Graph()
    aliases: dict[str, str] = {}

    def remember(alias: str, node_id: str):
        key = _normalize_key(alias)
        if key and key not in aliases:
            aliases[key] = node_id

    for row in df.itertuples(index=False):
        news_id = str(getattr(row, "news_id"))
        category = str(getattr(row, "category", "") or "")
        subcategory = str(getattr(row, "subcategory", "") or "")
        title = str(getattr(row, "title", "") or "")
        abstract = str(getattr(row, "abstract", "") or "")

        article_node = _article_node_id(news_id)
        graph.add_node(
            article_node,
            type="article",
            label=title or news_id,
            news_id=news_id,
            category=category,
            subcategory=subcategory,
            title=title,
        )
        remember(news_id, article_node)

        if category:
            category_node = _category_node_id(category)
            graph.add_node(category_node, type="category", label=category, category=category)
            graph.add_edge(article_node, category_node, relation="belongs_to")
            remember(category, category_node)

            if subcategory:
                subcategory_node = _subcategory_node_id(category, subcategory)
                graph.add_node(
                    subcategory_node,
                    type="subcategory",
                    label=subcategory,
                    category=category,
                    subcategory=subcategory,
                )
                graph.add_edge(category_node, subcategory_node, relation="has_subcategory")
                graph.add_edge(article_node, subcategory_node, relation="in_subcategory")
                remember(subcategory, subcategory_node)

        entities = _coerce_entities(getattr(row, "entities", None), title=title, abstract=abstract)
        for entity in entities:
            entity_node = _entity_node_id(entity)
            graph.add_node(
                entity_node,
                type=entity.get("type", "ENTITY"),
                label=entity.get("label") or entity.get("wikidata_id") or entity.get("id"),
                wikidata_id=entity.get("wikidata_id"),
                confidence=float(entity.get("confidence", 1.0)),
            )
            graph.add_edge(article_node, entity_node, relation="mentions")
            remember(entity.get("label", ""), entity_node)
            remember(entity.get("wikidata_id", ""), entity_node)
            for surface in entity.get("surface_forms", []):
                remember(surface, entity_node)

    graph.graph["version"] = _GRAPH_VERSION
    graph.graph["dataset_signature"] = signature
    graph.graph["aliases"] = aliases

    with open(cache_path, "wb") as handle:
        pickle.dump(graph, handle)
    print(f"Knowledge Graph saved: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")
    return graph


def _resolve_node_ids(reference: str, graph: nx.Graph) -> list[str]:
    if not reference:
        return []

    key = _normalize_key(reference)
    aliases = graph.graph.get("aliases", {})
    if key in aliases:
        return [aliases[key]]

    direct_matches = []
    for node_id, attrs in graph.nodes(data=True):
        if key in {
            _normalize_key(node_id),
            _normalize_key(attrs.get("label")),
            _normalize_key(attrs.get("news_id")),
            _normalize_key(attrs.get("wikidata_id")),
        }:
            direct_matches.append(node_id)
    return direct_matches


def get_article_entities(news_id: str, graph: nx.Graph) -> list[str]:
    article_nodes = _resolve_node_ids(news_id, graph)
    if not article_nodes:
        return []

    entities: list[str] = []
    for neighbor in graph.neighbors(article_nodes[0]):
        if graph.nodes[neighbor].get("type") not in ("article", "category", "subcategory"):
            entities.append(neighbor)
    return entities


def get_related_articles(reference: str, graph: nx.Graph, limit: int | None = None) -> list[str]:
    related: Counter = Counter()
    for node_id in _resolve_node_ids(reference, graph):
        node_type = graph.nodes[node_id].get("type")
        if node_type == "article":
            for neighbor in graph.neighbors(node_id):
                for candidate in graph.neighbors(neighbor):
                    if candidate == node_id:
                        continue
                    if graph.nodes[candidate].get("type") == "article":
                        related[graph.nodes[candidate]["news_id"]] += 1
        else:
            for neighbor in graph.neighbors(node_id):
                if graph.nodes[neighbor].get("type") == "article":
                    related[graph.nodes[neighbor]["news_id"]] += 1

    articles = [news_id for news_id, _ in related.most_common(limit)]
    return articles


def _build_preview(graph: nx.Graph, top_entities: list[dict], max_articles: int = 30) -> tuple[list[dict], list[dict]]:
    preview_node_ids = set()
    article_counts: Counter = Counter()

    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("type") == "category":
            preview_node_ids.add(node_id)

    for entity in top_entities:
        node_id = entity["node_id"]
        preview_node_ids.add(node_id)
        for neighbor in graph.neighbors(node_id):
            if graph.nodes[neighbor].get("type") == "article":
                article_counts[neighbor] += 1

    for article_node, _ in article_counts.most_common(max_articles):
        preview_node_ids.add(article_node)

    nodes = [
        {
            "id": node_id,
            "label": graph.nodes[node_id].get("label") or graph.nodes[node_id].get("news_id") or node_id,
            "type": graph.nodes[node_id].get("type"),
            "degree": int(graph.degree(node_id)),
        }
        for node_id in preview_node_ids
    ]

    links = []
    for source, target, attrs in graph.edges(data=True):
        if source in preview_node_ids and target in preview_node_ids:
            links.append(
                {
                    "source": source,
                    "target": target,
                    "relation": attrs.get("relation", "related_to"),
                }
            )

    return nodes, links


def get_graph_stats(graph: nx.Graph) -> dict:
    entity_entries = []
    categories = []
    for node_id, attrs in graph.nodes(data=True):
        node_type = attrs.get("type")
        if node_type == "category":
            categories.append(attrs.get("label", node_id))
        elif node_type not in ("article", "subcategory"):
            label = str(attrs.get("label", node_id) or "").strip()
            normalized_label = _normalize_key(label)
            if (
                node_type == "ENTITY"
                and not attrs.get("wikidata_id")
                and (normalized_label in _NOISY_ENTITY_LABELS or len(normalized_label) <= 3)
            ):
                continue
            entity_entries.append(
                {
                    "node_id": node_id,
                    "id": label,
                    "type": node_type,
                    "connections": int(graph.degree(node_id)),
                    "wikidata_id": attrs.get("wikidata_id"),
                }
            )

    top_entities = sorted(entity_entries, key=lambda item: item["connections"], reverse=True)[:30]
    preview_nodes, preview_links = _build_preview(graph, top_entities)
    return {
        "total_nodes": graph.number_of_nodes(),
        "total_edges": graph.number_of_edges(),
        "article_count": sum(1 for _, attrs in graph.nodes(data=True) if attrs.get("type") == "article"),
        "entity_count": len(entity_entries),
        "top_entities": [
            {
                "id": entry["id"],
                "type": entry["type"],
                "connections": entry["connections"],
                "wikidata_id": entry.get("wikidata_id"),
            }
            for entry in top_entities
        ],
        "categories": sorted(categories),
        "nodes": preview_nodes,
        "links": preview_links,
    }
