"""
Inspect the cached knowledge graph pickle from the project root.

Usage:
    python inspect_knowledge_graph.py
    python inspect_knowledge_graph.py --article-id N23093
"""

from __future__ import annotations

import argparse
import os
import pickle
from collections import Counter

DEFAULT_GRAPH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "graph",
    "knowledge_graph.pkl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect HyperNews knowledge graph cache.")
    parser.add_argument("--graph", default=DEFAULT_GRAPH, help="Path to the knowledge graph pickle.")
    parser.add_argument("--article-id", default="", help="Optional raw article id such as N23093.")
    parser.add_argument("--top", type=int, default=10, help="How many top entities/categories to print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.graph):
        raise FileNotFoundError(
            f"Missing graph file: {args.graph}\n"
            "Run `python build_knowledge_graph.py` first."
        )

    with open(args.graph, "rb") as handle:
        graph = pickle.load(handle)

    node_types = Counter(attrs.get("type", "unknown") for _, attrs in graph.nodes(data=True))
    print(f"Graph file: {args.graph}")
    print(f"Nodes:      {graph.number_of_nodes():,}")
    print(f"Edges:      {graph.number_of_edges():,}")
    print("Node types:")
    for node_type, count in node_types.most_common():
        print(f"  {node_type}: {count:,}")

    entity_nodes = [
        (node_id, attrs)
        for node_id, attrs in graph.nodes(data=True)
        if attrs.get("type") not in ("article", "category", "subcategory")
    ]
    top_entities = sorted(entity_nodes, key=lambda item: graph.degree(item[0]), reverse=True)[: args.top]
    print(f"Top {len(top_entities)} connected entities:")
    for node_id, attrs in top_entities:
        print(f"  {attrs.get('label', node_id)} ({attrs.get('type')}): degree={graph.degree(node_id)}")

    if args.article_id:
        article_node = f"article::{args.article_id}"
        if article_node not in graph:
            print(f"Article {args.article_id} not found.")
            return

        neighbors = list(graph.neighbors(article_node))
        print(f"Neighbors for {article_node}:")
        for neighbor in neighbors[:50]:
            attrs = graph.nodes[neighbor]
            print(f"  {neighbor} | type={attrs.get('type')} | label={attrs.get('label', neighbor)}")


if __name__ == "__main__":
    main()
