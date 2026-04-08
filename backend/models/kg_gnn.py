"""
GraphSAGE-based KG node encoder for HyperNews.

Replaces the scalar kg_score with a 64-d structural embedding per article node.
Architecture: 2-layer GraphSAGE with mean neighbourhood aggregation.
Training objective: link prediction (do two articles share KG neighbours?).

Outputs data/kg_embeddings.npy  (51282 × 64 float32, ~13 MB)
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


KG_EMB_DIM  = 64    # output embedding dimension
NODE_FEAT_DIM = 384  # input = article embedding from all-MiniLM-L6-v2
HIDDEN_DIM   = 128   # intermediate dimension


class SAGEConv(nn.Module):
    """Single GraphSAGE layer with mean aggregation."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.self_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.neigh_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        x: torch.Tensor,             # (N, in_dim) node features
        adj: torch.Tensor,           # (N, N) adjacency (sparse or dense)
    ) -> torch.Tensor:               # (N, out_dim)
        # mean aggregation of neighbours
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1)  # (N, 1)
        neigh_mean = (adj @ x) / deg                      # (N, in_dim)
        out = self.self_proj(x) + self.neigh_proj(neigh_mean)
        return F.gelu(self.norm(out))


class GraphSAGE(nn.Module):
    """
    2-layer GraphSAGE.
    Input:  (N, 384) article embeddings as node features
    Output: (N, 64)  structural KG embeddings
    """

    def __init__(
        self,
        in_dim: int = NODE_FEAT_DIM,
        hidden: int = HIDDEN_DIM,
        out_dim: int = KG_EMB_DIM,
    ):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, out_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        x: torch.Tensor,   # (N, 384)
        adj: torch.Tensor, # (N, N) normalised adjacency
    ) -> torch.Tensor:     # (N, 64)
        h = self.conv1(x, adj)
        h = self.dropout(h)
        h = self.conv2(h, adj)
        return F.normalize(h, p=2, dim=-1)   # unit-norm for cosine similarity


# ── Link prediction head ──────────────────────────────────────────────────────
class LinkPredHead(nn.Module):
    """Bilinear scorer for link prediction training."""

    def __init__(self, dim: int = KG_EMB_DIM):
        super().__init__()
        self.W = nn.Parameter(torch.eye(dim))

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return (u * (v @ self.W)).sum(dim=-1)


class KGGNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gnn = GraphSAGE()
        self.link_pred = LinkPredHead()

    def forward(self, x, adj):
        return self.gnn(x, adj)

    def link_score(self, emb, u_idx, v_idx):
        return self.link_pred(emb[u_idx], emb[v_idx])


# ── Adjacency builder ──────────────────────────────────────────────────────────
def build_adjacency(
    edge_list: List[Tuple[int, int]],
    n_nodes: int,
    edge_weights: Optional[List[float]] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Build row-normalised dense adjacency from edge list.
    For large graphs use sparse_coo_tensor (Phase 5 upgrade).
    """
    if edge_weights is None:
        edge_weights = [1.0] * len(edge_list)
    adj = torch.zeros(n_nodes, n_nodes, dtype=torch.float32, device=device)
    for (u, v), w in zip(edge_list, edge_weights):
        adj[u, v] = w
        adj[v, u] = w   # undirected
    # row-normalise
    deg = adj.sum(dim=1, keepdim=True).clamp(min=1)
    return adj / deg


def build_adjacency_sparse(
    edge_list: List[Tuple[int, int]],
    n_nodes: int,
    edge_weights: Optional[List[float]] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Memory-efficient sparse adjacency (use for n_nodes > 10k)."""
    if not edge_list:
        return torch.sparse_coo_tensor(
            torch.zeros(2, 0, dtype=torch.long),
            torch.zeros(0),
            (n_nodes, n_nodes),
        ).to(device)

    rows, cols, vals = [], [], []
    weights = edge_weights or [1.0] * len(edge_list)
    for (u, v), w in zip(edge_list, weights):
        rows += [u, v]
        cols += [v, u]
        vals += [w, w]

    idx = torch.tensor([rows, cols], dtype=torch.long)
    val = torch.tensor(vals, dtype=torch.float32)
    adj = torch.sparse_coo_tensor(idx, val, (n_nodes, n_nodes)).coalesce()

    # degree-normalise
    deg = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1).unsqueeze(1)
    return adj.to(device)


# ── Persistence ────────────────────────────────────────────────────────────────
def save_kg_embeddings(emb: np.ndarray, path: str = "data/kg_embeddings.npy"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, emb.astype(np.float32))
    print(f"Saved KG embeddings {emb.shape} → {path}")


def load_kg_embeddings(path: str = "data/kg_embeddings.npy") -> Optional[np.ndarray]:
    if os.path.exists(path):
        return np.load(path)
    return None


def save_gnn_model(model: KGGNNModel, path: str = "models/kg_gnn.pt"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Saved GNN model → {path}")


def load_gnn_model(path: str = "models/kg_gnn.pt", device: str = "cpu") -> KGGNNModel:
    m = KGGNNModel().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m
