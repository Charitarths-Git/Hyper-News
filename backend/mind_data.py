"""
Utilities for loading MIND-style news/behavior files and normalizing entity data.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from typing import Iterable
from urllib.parse import urlparse

import numpy as np
import pandas as pd

NEWS_COLUMNS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]

BEHAVIOR_COLUMNS = [
    "impression_id",
    "user_id",
    "time",
    "history",
    "impressions",
]

ENTITY_TYPE_MAP = {
    "P": "PERSON",
    "G": "GPE",
    "O": "ORG",
    "C": "CONCEPT",
    "L": "LOCATION",
    "E": "EVENT",
    "W": "WORK_OF_ART",
    "PR": "PRODUCT",
}


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, dict, tuple, set)):
        return False
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        return False
    return False


def _normalize_text(value, fallback: str = "") -> str:
    if _is_missing(value):
        return fallback
    return str(value).strip()


def _normalize_surface_forms(value) -> list[str]:
    if _is_missing(value):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    return []


def canonical_entity_type(raw_type: str) -> str:
    key = str(raw_type or "").strip().upper()
    return ENTITY_TYPE_MAP.get(key, key or "ENTITY")


def parse_entity_list(raw_value) -> list[dict]:
    if _is_missing(raw_value):
        return []

    parsed = raw_value
    if isinstance(raw_value, str):
        raw_value = raw_value.strip()
        if not raw_value or raw_value == "[]":
            return []
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            if "^" in raw_value:
                parsed = [{"WikidataId": token, "Label": token} for token in raw_value.split("^") if token]
            else:
                parsed = [{"Label": raw_value}]

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []

    entities: list[dict] = []
    for item in parsed:
        if isinstance(item, str):
            label = item.strip()
            if not label:
                continue
            entities.append(
                {
                    "id": label.lower(),
                    "label": label,
                    "type": "ENTITY",
                    "wikidata_id": None,
                    "confidence": 1.0,
                    "surface_forms": [label],
                }
            )
            continue

        if not isinstance(item, dict):
            continue

        label = _normalize_text(item.get("Label") or item.get("label") or item.get("name"))
        wikidata_id = _normalize_text(
            item.get("WikidataId") or item.get("wikidata_id") or item.get("id"),
            fallback="",
        )
        surface_forms = _normalize_surface_forms(item.get("SurfaceForms") or item.get("surface_forms"))
        if label and label not in surface_forms:
            surface_forms.insert(0, label)
        if not label and surface_forms:
            label = surface_forms[0]
        if not label and not wikidata_id:
            continue

        entity_id = wikidata_id or label.lower()
        raw_type = item.get("Type") or item.get("type")
        confidence = item.get("Confidence", item.get("confidence", 1.0))
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 1.0

        entities.append(
            {
                "id": entity_id,
                "label": label or wikidata_id,
                "type": canonical_entity_type(raw_type),
                "wikidata_id": wikidata_id or None,
                "confidence": confidence,
                "surface_forms": surface_forms,
            }
        )

    return entities


def merge_entities(*entity_columns) -> list[dict]:
    merged: dict[str, dict] = {}

    for raw_entities in entity_columns:
        for entity in parse_entity_list(raw_entities):
            key = str(entity["wikidata_id"] or entity["label"]).strip().lower()
            if not key:
                continue

            current = merged.get(key)
            if current is None:
                merged[key] = {
                    "id": entity["id"],
                    "label": entity["label"],
                    "type": entity["type"],
                    "wikidata_id": entity["wikidata_id"],
                    "confidence": entity["confidence"],
                    "surface_forms": list(dict.fromkeys(entity["surface_forms"])),
                    "mentions": 1,
                }
                continue

            current["confidence"] = max(float(current["confidence"]), float(entity["confidence"]))
            current["mentions"] += 1
            if not current.get("wikidata_id") and entity.get("wikidata_id"):
                current["wikidata_id"] = entity["wikidata_id"]
                current["id"] = entity["id"]
            if len(entity["label"]) > len(current["label"]):
                current["label"] = entity["label"]
            current["surface_forms"] = list(
                dict.fromkeys(list(current["surface_forms"]) + list(entity["surface_forms"]))
            )
            if current.get("type") in ("ENTITY", "CONCEPT") and entity.get("type") not in ("ENTITY", "CONCEPT"):
                current["type"] = entity["type"]

    return sorted(merged.values(), key=lambda entity: (-entity["confidence"], entity["label"].lower()))


def _compute_popularity(behavior_paths: Iterable[str]) -> tuple[Counter, Counter]:
    impression_counts: Counter = Counter()
    click_counts: Counter = Counter()

    for path in behavior_paths:
        if not path or not os.path.exists(path):
            continue

        for chunk in pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=BEHAVIOR_COLUMNS,
            usecols=["impressions"],
            dtype={"impressions": "string"},
            chunksize=5_000,
        ):
            for raw_impressions in chunk["impressions"].dropna():
                for item in str(raw_impressions).split():
                    if "-" not in item:
                        continue
                    news_id, label = item.rsplit("-", 1)
                    impression_counts[news_id] += 1
                    if label == "1":
                        click_counts[news_id] += 1

    return impression_counts, click_counts


def _extract_source(url: str) -> str:
    parsed = urlparse(url or "")
    return parsed.netloc.lower()


def load_mind_news(news_paths: Iterable[str], behavior_paths: Iterable[str] | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in news_paths:
        if not path or not os.path.exists(path):
            continue
        frames.append(pd.read_csv(path, sep="\t", header=None, names=NEWS_COLUMNS))

    if not frames:
        raise FileNotFoundError("No MIND news.tsv files were found.")

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["news_id"], keep="first")
    df["category"] = df["category"].fillna("news").astype(str).str.strip().str.lower()
    df["subcategory"] = df["subcategory"].fillna("").astype(str).str.strip().str.lower()
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["abstract"] = df["abstract"].fillna("").astype(str).str.strip()
    df["url"] = df["url"].fillna("").astype(str).str.strip()
    df["title_entities"] = df["title_entities"].apply(parse_entity_list)
    df["abstract_entities"] = df["abstract_entities"].apply(parse_entity_list)
    df["entities"] = [
        merge_entities(title_entities, abstract_entities)
        for title_entities, abstract_entities in zip(df["title_entities"], df["abstract_entities"])
    ]
    df["entity_ids"] = df["entities"].apply(
        lambda entities: [entity["wikidata_id"] or entity["id"] for entity in entities if entity.get("id")]
    )
    df["entity_labels"] = df["entities"].apply(
        lambda entities: [entity["label"] for entity in entities if entity.get("label")]
    )
    df["source"] = df["url"].apply(_extract_source)
    df["text"] = (
        df["title"].fillna("")
        + ". "
        + df["abstract"].fillna("")
        + " Category: "
        + df["category"].fillna("")
        + " Subcategory: "
        + df["subcategory"].fillna("")
    )

    behavior_paths = list(behavior_paths or [])
    impression_counts, click_counts = _compute_popularity(behavior_paths)
    df["impressions_count"] = df["news_id"].map(impression_counts).fillna(0).astype(int)
    df["click_count"] = df["news_id"].map(click_counts).fillna(0).astype(int)
    df["ctr"] = np.where(
        df["impressions_count"] > 0,
        df["click_count"] / df["impressions_count"],
        0.0,
    )

    log_impressions = np.log1p(df["impressions_count"].astype(float))
    max_log_impressions = float(log_impressions.max()) if len(log_impressions) else 1.0
    impression_signal = log_impressions / max(max_log_impressions, 1.0)
    df["popularity"] = (0.65 * df["ctr"].astype(float)) + (0.35 * impression_signal.astype(float))
    return df.reset_index(drop=True)


def resolve_default_mind_paths(base_dir: str) -> tuple[list[str], list[str]]:
    candidates = [
        os.path.join(base_dir, "data", "mind_full", "MIND-small"),
        os.path.join(base_dir, "data", "mind"),
    ]

    news_paths: list[str] = []
    behavior_paths: list[str] = []
    for candidate in candidates:
        for split in ("train", "dev", "test"):
            news_path = os.path.join(candidate, split, "news.tsv")
            behavior_path = os.path.join(candidate, split, "behaviors.tsv")
            if os.path.exists(news_path):
                news_paths.append(news_path)
            if os.path.exists(behavior_path):
                behavior_paths.append(behavior_path)

    deduped_news = list(dict.fromkeys(news_paths))
    deduped_behaviors = list(dict.fromkeys(behavior_paths))
    return deduped_news, deduped_behaviors
