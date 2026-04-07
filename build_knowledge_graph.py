"""
Build the cached knowledge graph pickle from the processed articles parquet.

Usage:
    python build_knowledge_graph.py
    python build_knowledge_graph.py --force-rebuild
    python build_knowledge_graph.py --max-articles 1000
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.graph import build_knowledge_graph

DEFAULT_PARQUET = os.path.join(ROOT, "data", "articles.parquet")
DEFAULT_GRAPH = os.path.join(ROOT, "graph", "knowledge_graph.pkl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HyperNews knowledge graph cache.")
    parser.add_argument("--parquet", default=DEFAULT_PARQUET, help="Path to the processed articles parquet.")
    parser.add_argument("--output", default=DEFAULT_GRAPH, help="Where to save the knowledge graph pickle.")
    parser.add_argument("--max-articles", type=int, default=0, help="Optional cap for local builds.")
    parser.add_argument("--force-rebuild", action="store_true", help="Ignore cached graph and rebuild.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.parquet):
        raise FileNotFoundError(
            f"Missing parquet file: {args.parquet}\n"
            "Run `python backend/generate_data.py` first."
        )

    df = pd.read_parquet(args.parquet)
    if args.max_articles and len(df) > args.max_articles:
        if "popularity" in df.columns:
            df = df.sort_values("popularity", ascending=False).head(args.max_articles).reset_index(drop=True)
        else:
            df = df.head(args.max_articles).reset_index(drop=True)

    graph = build_knowledge_graph(
        df,
        force_rebuild=args.force_rebuild,
        cache_path=args.output,
    )

    print(f"Graph file: {args.output}")
    print(f"Articles:   {len(df):,}")
    print(f"Nodes:      {graph.number_of_nodes():,}")
    print(f"Edges:      {graph.number_of_edges():,}")


if __name__ == "__main__":
    main()
