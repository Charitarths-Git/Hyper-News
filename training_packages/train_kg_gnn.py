"""
GraphSAGE KG GNN Training — standalone script.
================================================
Trains a 2-layer GraphSAGE over the HyperNews knowledge graph and saves:
    data/kg_embeddings.npy  — (N_nodes, 64) float32
    models/kg_gnn.pt        — model weights

Run after Phase 4 (DQN) is stable, as per the roadmap.

    python train_kg_gnn.py \\
        --graph-path  /path/to/graph/knowledge_graph.pkl \\
        --data-dir    /path/to/data \\
        --output-dir  ./outputs \\
        --device      cuda \\
        --epochs      30
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from path_utils import resolve_backend_dir

sys.path.insert(0, str(resolve_backend_dir(__file__)))
from models.kg_gnn import KGGNNModel, build_adjacency_sparse, save_kg_embeddings, save_gnn_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-path", required=True, help="Path to knowledge_graph.pkl")
    parser.add_argument("--data-dir", required=True, help="Path to data/ dir (article_embeddings.npy)")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--pos-edges", type=int, default=2048, help="Positive edges per batch")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ── Load KG ────────────────────────────────────────────────────────────────
    print(f"  Loading KG from {args.graph_path} …")
    with open(args.graph_path, "rb") as f:
        G = pickle.load(f)

    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)
    print(f"  {n_nodes:,} nodes  |  {G.number_of_edges():,} edges")

    # ── Node features ──────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    emb_npy  = data_dir / "article_embeddings.npy"
    ids_json = data_dir / "article_ids.json"

    article_emb = None
    id_to_emb_idx = {}
    if emb_npy.exists():
        article_emb = np.load(str(emb_npy)).astype(np.float32)
        if ids_json.exists():
            with open(ids_json) as f:
                id_list = json.load(f)
            id_to_emb_idx = {nid: i for i, nid in enumerate(id_list)}

    node_feats = np.zeros((n_nodes, 384), dtype=np.float32)
    for node_str, idx in node_to_idx.items():
        news_id = str(node_str).replace("article::", "").strip()
        emb_idx = id_to_emb_idx.get(news_id)
        if emb_idx is not None and article_emb is not None:
            node_feats[idx] = article_emb[emb_idx]

    x = torch.tensor(node_feats, dtype=torch.float32, device=device)
    print(f"  Node feature matrix: {x.shape}")

    # ── Build sparse adjacency ─────────────────────────────────────────────────
    edges_raw = [(u, v) for u, v in G.edges() if u in node_to_idx and v in node_to_idx]
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in edges_raw]
    edge_weights = [float(G[u][v].get("weight", 1.0)) for u, v in edges_raw]
    print(f"  Building sparse adjacency ({len(edges):,} edges) …")
    adj = build_adjacency_sparse(edges, n_nodes, edge_weights, device=str(device))
    edge_arr = np.array(edges)

    # ── Train ──────────────────────────────────────────────────────────────────
    model = KGGNNModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    print(f"\n  Training GraphSAGE: {args.epochs} epochs …")
    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()

        emb = model.gnn(x, adj)   # (N, 64)

        # Link prediction batch
        n_pos = min(args.pos_edges, len(edges))
        idx = np.random.choice(len(edges), n_pos, replace=False)
        pos_u = torch.tensor(edge_arr[idx, 0], device=device)
        pos_v = torch.tensor(edge_arr[idx, 1], device=device)
        neg_v = torch.randint(0, n_nodes, (n_pos,), device=device)

        pos_score = model.link_pred(emb[pos_u], emb[pos_v])
        neg_score = model.link_pred(emb[pos_u], emb[neg_v])
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([torch.ones(n_pos, device=device), torch.zeros(n_pos, device=device)])
        loss = F.binary_cross_entropy_with_logits(scores, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        lv = loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"    epoch {epoch+1:>3}/{args.epochs}  loss={lv:.4f}")
        if lv < best_loss:
            best_loss = lv
            save_gnn_model(model, os.path.join(args.output_dir, "kg_gnn_best.pt"))

    # ── Extract and save ───────────────────────────────────────────────────────
    print("\n  Extracting full node embeddings …")
    model.eval()
    with torch.no_grad():
        full_emb = model.gnn(x, adj).cpu().numpy()

    out_path = os.path.join(args.output_dir, "kg_embeddings.npy")
    save_kg_embeddings(full_emb, out_path)
    save_gnn_model(model, os.path.join(args.output_dir, "kg_gnn.pt"))

    print(f"\n  ✓ GNN training complete")
    print(f"    embeddings : {full_emb.shape} → {out_path}")
    print(f"    model      : {args.output_dir}/kg_gnn.pt")
    print(f"    best loss  : {best_loss:.4f}")


if __name__ == "__main__":
    main()
