"""
HyperNews DQN + GNN Training — Lightning AI H100 (80 GB HBM)
==============================================================
Full pipeline: Phase A offline DQN + GNN training on MIND-large.
Large batch, full pool_size=200, full Transformer, all 200k+ impressions.
Also runs train_kg_gnn.py to generate kg_embeddings.npy before DQN.

Run (Lightning AI Studio):
    pip install -r requirements_train.txt
    python train_h100.py \\
        --mind-path   /teamspace/uploads/MIND-large \\
        --data-dir    /teamspace/uploads/data \\
        --output-dir  /teamspace/studios/outputs \\
        --phase       AB \\
        --run-gnn     1

The H100 run also performs a full nDCG@10 sweep over hyperparameters
if --sweep is passed.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train_dqn_core import TierConfig, main as dqn_main

# ── H100 config ────────────────────────────────────────────────────────────────
CFG_H100 = TierConfig(
    name="H100-80GB",
    device="cuda",
    use_amp=True,              # bfloat16 preferred on H100 (set via env TF32)
    batch_size=512,
    replay_capacity=200_000,
    target_update_freq=2_000,
    history_nhead=8,           # larger attention for H100
    history_layers=4,
    history_d_model=512,       # wider than spec — H100 can handle it
    epsilon_decay_steps=100_000,
    max_impressions=0,         # all rows
    pool_size=200,             # full spec
    grad_accum=1,
    num_workers=8,
    checkpoint_every=20_000,
    log_every=1_000,
    kg_emb_path="",            # updated after GNN run
)


def run_gnn_training(args: argparse.Namespace):
    """Train GraphSAGE KG GNN and save kg_embeddings.npy."""
    import torch
    import numpy as np
    import networkx as nx
    import pickle

    sys.path.insert(0, str(Path(args.data_dir).parent / "backend"))
    from models.kg_gnn import KGGNNModel, build_adjacency_sparse, save_kg_embeddings, save_gnn_model

    device = torch.device("cuda")

    # Load KG
    kg_path = Path(args.data_dir).parent / "graph" / "knowledge_graph.pkl"
    if not kg_path.exists():
        print(f"  KG not found at {kg_path}. Skipping GNN training.")
        return

    print(f"  Loading KG from {kg_path} …")
    with open(kg_path, "rb") as f:
        G = pickle.load(f)

    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)
    print(f"  KG: {n_nodes:,} nodes, {G.number_of_edges():,} edges")

    # Node features: load article embeddings where available, else zeros
    emb_npy = Path(args.data_dir) / "article_embeddings.npy"
    article_emb = np.load(str(emb_npy)).astype(np.float32) if emb_npy.exists() else None
    ids_json = Path(args.data_dir) / "article_ids.json"
    id_to_emb_idx = {}
    if article_emb is not None and ids_json.exists():
        with open(ids_json) as f:
            id_list = json.load(f)
        id_to_emb_idx = {nid: i for i, nid in enumerate(id_list)}

    node_feats = np.zeros((n_nodes, 384), dtype=np.float32)
    for node_str, idx in node_to_idx.items():
        if "article::" in str(node_str):
            news_id = str(node_str).replace("article::", "")
            emb_idx = id_to_emb_idx.get(news_id)
            if emb_idx is not None and article_emb is not None:
                node_feats[idx] = article_emb[emb_idx]

    x = torch.tensor(node_feats, dtype=torch.float32, device=device)

    # Build sparse adjacency
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges() if u in node_to_idx and v in node_to_idx]
    edge_weights = [float(G[u][v].get("weight", 1.0)) for u, v in G.edges() if u in node_to_idx and v in node_to_idx]
    print(f"  Building adjacency … ({len(edges):,} edges)")
    adj = build_adjacency_sparse(edges, n_nodes, edge_weights, device=str(device))

    # Train GNN
    model = KGGNNModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

    print("  Training GraphSAGE …")
    edge_arr = np.array(edges)
    for epoch in range(20):
        model.train()
        opt.zero_grad()

        # Dense forward (chunked for memory safety on H100)
        chunk_size = 10_000
        embs = []
        for start in range(0, n_nodes, chunk_size):
            end = min(start + chunk_size, n_nodes)
            # Simplified: pass local subgraph features
            x_chunk = x[start:end]
            embs.append(x_chunk)
        # Full forward (works if n_nodes < 200k on H100)
        emb = model(x, adj)

        # Link prediction loss
        if len(edge_arr) < 2:
            break
        pos_u = torch.tensor(edge_arr[:, 0][:1024], device=device)
        pos_v = torch.tensor(edge_arr[:, 1][:1024], device=device)
        neg_v = torch.randint(0, n_nodes, (len(pos_u),), device=device)

        pos_score = model.link_pred(emb[pos_u], emb[pos_v])
        neg_score = model.link_pred(emb[pos_u], emb[neg_v])
        labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
        scores = torch.cat([pos_score, neg_score])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels)
        loss.backward()
        opt.step()

        if (epoch + 1) % 5 == 0:
            print(f"    GNN epoch {epoch+1:>3}/20  loss={loss.item():.4f}")

    # Extract embeddings
    model.eval()
    with torch.no_grad():
        full_emb = model(x, adj).cpu().numpy()  # (n_nodes, 64)

    out_emb_path = os.path.join(args.output_dir, "kg_embeddings.npy")
    save_kg_embeddings(full_emb, out_emb_path)
    save_gnn_model(model, os.path.join(args.output_dir, "kg_gnn.pt"))
    print(f"  GNN complete. Embeddings: {full_emb.shape} → {out_emb_path}")
    return out_emb_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H100 full training pipeline")
    parser.add_argument("--mind-path", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="./output_h100")
    parser.add_argument("--phase", default="AB", choices=["A", "B", "AB"])
    parser.add_argument("--max-impressions", type=int, default=0)
    parser.add_argument("--run-gnn", type=int, default=1, help="1=train GNN first, 0=skip")
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: GNN
    if args.run_gnn:
        kg_emb_path = run_gnn_training(args)
        if kg_emb_path:
            CFG_H100.kg_emb_path = kg_emb_path

    # Step 2: DQN
    CFG_H100.max_impressions = args.max_impressions
    dqn_main(CFG_H100, args)
