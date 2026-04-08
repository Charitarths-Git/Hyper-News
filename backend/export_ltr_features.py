"""
Export LTR training rows from live app events or the bundled MIND dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile

import pandas as pd

from db import fetch_ltr_training_rows, init_db
from evaluate_mind import (
    _build_user,
    _compute_embeddings,
    _load_behaviors,
    _load_prepared_eval_assets,
    _offline_memory_bonus_map,
    _required_news_ids,
)
from graph import build_knowledge_graph
from mind_data import load_mind_news
from ranker import (
    _build_history_profiles,
    _build_news_id_to_idx,
    _build_skip_profiles,
    _graph_bonus_map,
    build_article_feature_map,
    build_user_profile_vector,
)


_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_DEFAULT_OUTPUT = os.path.join(_BASE, "data", "ltr_features.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LTR feature rows for HyperNews.")
    parser.add_argument("--source", choices=["auto", "db", "mind"], default="auto")
    parser.add_argument("--output-csv", default=_DEFAULT_OUTPUT)
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument(
        "--train-news",
        default=os.path.join(_BASE, "data", "mind_full", "MIND-small", "train", "news.tsv"),
    )
    parser.add_argument(
        "--train-behaviors",
        default=os.path.join(_BASE, "data", "mind_full", "MIND-small", "train", "behaviors.tsv"),
    )
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2")
    parser.add_argument("--limit-impressions", type=int, default=2000)
    return parser.parse_args()


def _export_db_rows(limit_rows: int = 0) -> pd.DataFrame:
    init_db()
    rows = fetch_ltr_training_rows(limit=limit_rows)
    records: list[dict] = []
    for row in rows:
        try:
            features = json.loads(row.get("features_json") or "{}")
        except Exception:
            features = {}
        records.append(
            {
                **features,
                "label": float(row.get("label", 0.0) or 0.0),
                "group_id": str(row.get("request_id") or ""),
                "article_id": str(row.get("article_id") or ""),
                "position": int(row.get("position", -1) or -1),
                "dwell_time": float(row.get("dwell_time", 0.0) or 0.0),
                "candidate_source": str(row.get("candidate_source") or ""),
            }
        )
    return pd.DataFrame(records)


def _export_mind_rows(train_news: str, train_behaviors: str, model_name: str, limit_impressions: int) -> pd.DataFrame:
    behaviors = _load_behaviors(train_behaviors, limit_impressions)
    required_ids = _required_news_ids(behaviors)
    eval_df, embeddings = _load_prepared_eval_assets(required_ids)
    if eval_df is None or embeddings is None:
        full_df = load_mind_news([train_news], behavior_paths=[train_behaviors])
        eval_df = full_df[full_df["news_id"].isin(required_ids)].reset_index(drop=True)
        embeddings = _compute_embeddings(eval_df, model_name)

    news_lookup = {row["news_id"]: row for row in eval_df.to_dict("records")}
    news_id_to_idx = _build_news_id_to_idx(eval_df)
    rows: list[dict] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        graph_path = os.path.join(tmp_dir, "ltr_kg.pkl")
        graph = build_knowledge_graph(eval_df, force_rebuild=True, cache_path=graph_path)

        for group_index, behavior in enumerate(behaviors.itertuples(index=False), start=1):
            user = _build_user(behavior, news_lookup)
            impression_pairs = [item.rsplit("-", 1) for item in str(behavior.impressions or "").split() if "-" in item]
            candidate_articles = [news_lookup[news_id] for news_id, _ in impression_pairs if news_id in news_lookup]
            if len(candidate_articles) < 2:
                continue

            profile_vector = build_user_profile_vector(user, embeddings, news_id_to_idx)
            subcategory_weights, entity_weights = _build_history_profiles(user, eval_df, news_id_to_idx, graph=graph)
            skipped_categories, skipped_entities = _build_skip_profiles(user, eval_df, news_id_to_idx, graph=graph)
            skipped_ids = set(getattr(user, "recent_negative_actions", [])[-25:]) | set(getattr(user, "recent_skips", [])[-25:])
            graph_bonus = _graph_bonus_map(user, graph)
            memory_bonus_map = _offline_memory_bonus_map(user, candidate_articles, eval_df, embeddings, graph)

            for position, (news_id, label) in enumerate(impression_pairs):
                article = news_lookup.get(news_id)
                if not article:
                    continue
                features, _extras = build_article_feature_map(
                    user,
                    article,
                    embeddings,
                    eval_df,
                    G=graph,
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
                if not features:
                    continue
                rows.append(
                    {
                        **features,
                        "label": float(label),
                        "group_id": f"mind_{group_index}",
                        "article_id": str(news_id),
                        "position": int(position),
                        "dwell_time": 0.0,
                        "candidate_source": "mind_impression",
                    }
                )

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    source = args.source

    if source in {"auto", "db"}:
        db_frame = _export_db_rows(limit_rows=args.limit_rows)
        if source == "db":
            frame = db_frame
        elif len(db_frame) >= 200:
            frame = db_frame
            source = "db"
        else:
            frame = pd.DataFrame()
    else:
        frame = pd.DataFrame()

    if frame.empty:
        frame = _export_mind_rows(
            train_news=args.train_news,
            train_behaviors=args.train_behaviors,
            model_name=args.model_name,
            limit_impressions=args.limit_impressions,
        )
        source = "mind"

    if frame.empty:
        raise SystemExit("No LTR feature rows could be exported.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
    print(
        json.dumps(
            {
                "source": source,
                "output_csv": args.output_csv,
                "rows": int(len(frame)),
                "groups": int(frame["group_id"].nunique()) if "group_id" in frame.columns else 0,
                "labels": sorted(frame["label"].unique().tolist()) if "label" in frame.columns else [],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
