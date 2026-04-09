"""
BioEncoder Training — runs on ANY GPU tier or CPU.
====================================================
Since MIND has no signup data, we use MIND user clusters as proxy labels:
  1. Cluster users by their click category distribution (k=20 clusters)
  2. Each cluster → a "synthetic bio profile" (age_bucket, occupation, interests)
  3. Train BioEncoder to map those profiles back to cluster centroids

This gives the encoder a real gradient signal. In production the encoder
will receive actual onboarding answers from OnboardingFlow.tsx.

Run:
    python train_bio_encoder.py \\
        --mind-path /path/to/MIND \\
        --data-dir  /path/to/data \\
        --output-dir ./outputs \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).parent))
from path_utils import resolve_backend_dir

sys.path.insert(0, str(resolve_backend_dir(__file__)))

from bio_categories import BIO_CATEGORY_ORDER
from models.bio_encoder import BioEncoder, VOCABS, field_to_index, save_bio_encoder

CATEGORIES = list(BIO_CATEGORY_ORDER)
N_CLUSTERS = 20


def parse_behaviors_bio(path: str, max_rows: int = 50_000) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            _, uid, _, hist, cands = parts[:5]
            clicked = [c.split("-")[0] for c in cands.split() if c.endswith("-1")]
            rows.append({"user_id": uid, "history": hist.split(), "clicked": clicked})
    return rows


def build_user_category_vectors(rows: List[Dict], news: Dict) -> np.ndarray:
    """Returns (N, len(CATEGORIES)) float32 category distribution per user."""
    cat_to_idx = {c: i for i, c in enumerate(CATEGORIES)}
    vecs = []
    for r in rows:
        v = np.zeros(len(CATEGORIES), dtype=np.float32)
        for nid in r["history"] + r["clicked"]:
            info = news.get(nid, {})
            cat = str(info.get("category", "")).lower()
            if cat in cat_to_idx:
                v[cat_to_idx[cat]] += 1.0
        norm = v.sum()
        vecs.append(v / max(norm, 1.0))
    return np.stack(vecs) if vecs else np.zeros((1, len(CATEGORIES)), dtype=np.float32)


def cluster_users(vecs: np.ndarray, k: int = N_CLUSTERS) -> np.ndarray:
    km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1024)
    labels = km.fit_predict(vecs)
    return labels, km.cluster_centers_


def make_synthetic_bio(cluster_id: int, centroid: np.ndarray) -> Dict:
    """Map cluster centroid to plausible onboarding fields."""
    top_cats = [CATEGORIES[i] for i in np.argsort(centroid)[::-1][:3]]
    age_options = list(VOCABS["age_bucket"])
    occ_options = list(VOCABS["occupation"])
    return {
        "age_bucket":    age_options[cluster_id % len(age_options)],
        "gender":        "unknown",
        "occupation":    occ_options[cluster_id % len(occ_options)],
        "location_region": "unknown",
        "top_categories": top_cats,
    }


def build_dataset(
    rows: List[Dict],
    labels: np.ndarray,
    centroids: np.ndarray,
    sentence_model,
    device: torch.device,
):
    """Returns lists of (cat_indices, text_emb_tensor, target_centroid_tensor)."""
    cat_indices_list, text_embs_list, targets_list = [], [], []

    for r, cluster_id in zip(rows, labels):
        bio = make_synthetic_bio(int(cluster_id), centroids[cluster_id])
        cat_idx = np.array([
            field_to_index("age_bucket", bio["age_bucket"]),
            field_to_index("gender", bio["gender"]),
            field_to_index("occupation", bio["occupation"]),
            field_to_index("location_region", bio["location_region"]),
        ], dtype=np.int64)
        interest_text = " ".join(bio["top_categories"])
        text_emb = sentence_model.encode(interest_text, convert_to_numpy=True).astype(np.float32)
        target = centroids[cluster_id].astype(np.float32)
        # Pad target to 64 dims (centroid is len(CATEGORIES))
        target_padded = np.zeros(64, dtype=np.float32)
        target_padded[:len(target)] = normalize(target.reshape(1, -1))[0]

        cat_indices_list.append(cat_idx)
        text_embs_list.append(text_emb)
        targets_list.append(target_padded)

    return (
        torch.tensor(np.stack(cat_indices_list), dtype=torch.long),
        torch.tensor(np.stack(text_embs_list), dtype=torch.float32),
        torch.tensor(np.stack(targets_list), dtype=torch.float32),
    )


def train_bio_encoder(
    cat_indices: torch.Tensor,
    text_embs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> BioEncoder:
    model = BioEncoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    N = len(cat_indices)

    print(f"  Training BioEncoder: {N} samples, {epochs} epochs, batch={batch_size}")
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(N)
        total_loss = 0.0
        steps = 0
        for start in range(0, N, batch_size):
            idx = perm[start: start + batch_size]
            c = cat_indices[idx].to(device)
            t = text_embs[idx].to(device)
            tgt = targets[idx].to(device)
            pred = model(c, t)
            # Cosine embedding loss — maximise alignment with cluster centroid
            loss = 1.0 - F.cosine_similarity(pred, tgt, dim=-1).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            steps += 1
        sched.step()
        if (epoch + 1) % 5 == 0:
            print(f"    epoch {epoch+1:>3}/{epochs}  loss={total_loss/max(steps,1):.4f}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mind-path", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-rows", type=int, default=50_000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Load MIND
    mind = Path(args.mind_path)
    news_path  = mind / "train" / "news.tsv"
    behav_path = mind / "train" / "behaviors.tsv"
    if not news_path.exists():
        news_path  = mind / "news.tsv"
        behav_path = mind / "behaviors.tsv"

    print("  Loading MIND news …")
    news = {}
    with open(news_path, encoding="utf-8") as f:
        for line in f:
            p = line.strip().split("\t")
            if len(p) >= 5:
                news[p[0]] = {"category": p[1], "subcategory": p[2], "title": p[3], "abstract": p[4]}

    print("  Loading behaviors …")
    rows = parse_behaviors_bio(str(behav_path), max_rows=args.max_rows)
    print(f"  {len(rows):,} users loaded")

    print("  Building category vectors …")
    vecs = build_user_category_vectors(rows, news)

    print(f"  Clustering into {N_CLUSTERS} clusters …")
    labels, centroids = cluster_users(vecs, k=N_CLUSTERS)

    print("  Loading SentenceTransformer …")
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("all-MiniLM-L6-v2", device=args.device)

    print("  Building dataset …")
    cat_idx, text_embs, targets = build_dataset(rows, labels, centroids, st, device)

    print("  Training BioEncoder …")
    model = train_bio_encoder(cat_idx, text_embs, targets, device, epochs=args.epochs)

    out_path = os.path.join(args.output_dir, "bio_encoder.pt")
    save_bio_encoder(model, out_path)
    print(f"  ✓ BioEncoder saved → {out_path}")


if __name__ == "__main__":
    main()
