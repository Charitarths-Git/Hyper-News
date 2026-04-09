"""
Quick plotting helper for a 2-hop article subgraph.
"""

import os
import pickle

import matplotlib.pyplot as plt
import networkx as nx

GRAPH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph", "knowledge_graph.pkl")
ARTICLE_ID = "article::N55528"

with open(GRAPH_PATH, "rb") as handle:
    graph = pickle.load(handle)

if ARTICLE_ID in graph:
    neighbors = set(graph.neighbors(ARTICLE_ID))
    second_degree = set()
    for node in neighbors:
        second_degree.update(graph.neighbors(node))
    subgraph_nodes = {ARTICLE_ID} | neighbors | second_degree
    subgraph = graph.subgraph(subgraph_nodes)
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(subgraph, k=0.5)
    nx.draw(subgraph, pos, with_labels=True, node_size=500, font_size=8)
    plt.title(f"2-hop Knowledge Graph for {ARTICLE_ID}")
    plt.show()
else:
    print(f"Article {ARTICLE_ID} not found in the graph.")
