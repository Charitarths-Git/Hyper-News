"""
HyperNews — DQN Training Core
==============================
Shared training logic used by all GPU tier scripts.
GPU tier scripts import this and pass a TierConfig.

Phase A: Offline DQN training on MIND behaviors.tsv with IPS correction.
Phase B: (optional) Online fine-tuning loop from /feedback events.

Usage (from a tier script):
    python train_rtx4060.py --mind-path /path/to/MIND-large --phase A
    python train_h100.py    --mind-path /path/to/MIND-large --phase AB
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ── Allow running from any working directory ──────────────────────────────────
from path_utils import resolve_backend_dir, resolve_project_root

_REPO = resolve_project_root(__file__)
_BACKEND = resolve_backend_dir(__file__)
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from models.history_encoder import HistoryEncoder
from models.bio_encoder import BioEncoder, encode_bio_fields
from models.dqn_policy import DoubleDQNAgent, Transition, SLATE_SIZE, STATE_DIM, ITEM_DIM
from models.kg_gnn import load_kg_embeddings
from logging_policy import LoggingPolicy


# ══════════════════════════════════════════════════════════════════════════════
# Tier Config  — each GPU tier script sets this
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TierConfig:
    name: str                   # e.g. "RTX4060"
    device: str                 # "cuda" or "cpu"
    use_amp: bool               # mixed precision
    batch_size: int             # DQN training batch
    replay_capacity: int        # experience replay size
    target_update_freq: int     # steps between target sync
    history_nhead: int          # Transformer heads in HistoryEncoder
    history_layers: int         # Transformer layers
    history_d_model: int        # Transformer inner dim
    epsilon_decay_steps: int    # ε-greedy annealing
    max_impressions: int        # MIND rows to train on (0 = all)
    pool_size: int              # candidate pool per step
    grad_accum: int             # gradient accumulation steps
    num_workers: int            # DataLoader workers
    checkpoint_every: int       # steps between saves
    log_every: int              # steps between console logs
    kg_emb_path: str            # path to kg_embeddings.npy (or "")


# ══════════════════════════════════════════════════════════════════════════════
# MIND dataset reader
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Impression:
    user_id: str
    history: List[str]          # clicked article IDs (ordered)
    candidates: List[str]       # [news_id]-[label] pairs
    clicked: List[str]          # IDs with label=1
    non_clicked: List[str]      # IDs with label=0


def parse_behaviors(path: str, max_rows: int = 0) -> List[Impression]:
    impressions = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            _, user_id, _, history_str, candidates_str = parts[:5]
            history = history_str.strip().split() if history_str.strip() else []
            candidates = candidates_str.strip().split()
            clicked, non_clicked = [], []
            for c in candidates:
                if "-" in c:
                    nid, lbl = c.rsplit("-", 1)
                    (clicked if lbl == "1" else non_clicked).append(nid)
                else:
                    non_clicked.append(c)
            impressions.append(Impression(user_id, history, candidates, clicked, non_clicked))
    return impressions


def parse_news(path: str) -> Dict[str, Dict]:
    news = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            news_id, category, subcategory, title, abstract = parts[:5]
            news[news_id] = {
                "news_id": news_id,
                "category": category,
                "subcategory": subcategory,
                "title": title,
                "abstract": abstract,
            }
    return news


# ══════════════════════════════════════════════════════════════════════════════
# Article embedding lookup
# ══════════════════════════════════════════════════════════════════════════════

class ArticleEmbeddingIndex:
    """
    Wraps the precomputed 384-d article_embeddings.npy.
    Falls back to zero vectors for unknown IDs.
    """

    def __init__(
        self,
        embeddings_npy: str,
        id_list_json: str,
        kg_emb_path: str = "",
    ):
        emb = np.load(embeddings_npy).astype(np.float32)
        with open(id_list_json) as f:
            ids = json.load(f)
        self.id_to_idx: Dict[str, int] = {nid: i for i, nid in enumerate(ids)}
        self.article_emb = emb                       # (N, 384)
        self.kg_emb: Optional[np.ndarray] = None
        if kg_emb_path and os.path.exists(kg_emb_path):
            self.kg_emb = np.load(kg_emb_path).astype(np.float32)

    def get(self, news_id: str) -> np.ndarray:
        """Returns 448-d [article(384) | kg(64)] or [article(384) | zeros(64)]."""
        idx = self.id_to_idx.get(news_id)
        a_emb = self.article_emb[idx] if idx is not None else np.zeros(384, np.float32)
        if self.kg_emb is not None:
            k_emb = self.kg_emb[idx] if idx is not None else np.zeros(64, np.float32)
        else:
            k_emb = np.zeros(64, np.float32)
        return np.concatenate([a_emb, k_emb])        # (448,)

    def get_batch(self, news_ids: List[str]) -> np.ndarray:
        return np.stack([self.get(nid) for nid in news_ids])


# ══════════════════════════════════════════════════════════════════════════════
# State builder
# ══════════════════════════════════════════════════════════════════════════════

class StateBuilder:
    """
    Constructs the 206-d fused user state:
        history_emb  (128-d)  — HistoryEncoder over last 50 clicks
        bio_emb      (64-d)   — BioEncoder (zeros for MIND, no signup data)
        affect_emb   (2-d)    — valence/arousal (zeros offline)
        context_feats(12-d)   — hour, weekday, device, session_len, ...
    """

    def __init__(
        self,
        article_index: ArticleEmbeddingIndex,
        history_encoder: HistoryEncoder,
        bio_encoder: Optional[BioEncoder],
        device: torch.device,
        max_seq: int = 50,
    ):
        self.article_index = article_index
        self.history_encoder = history_encoder
        self.bio_encoder = bio_encoder
        self.device = device
        self.max_seq = max_seq

    @torch.no_grad()
    def build(
        self,
        history: List[str],
        interaction_count: int = 0,
    ) -> np.ndarray:
        """Returns 206-d state vector."""
        # history embedding
        if history:
            ids = history[-self.max_seq:]
            seq = np.stack([self.article_index.get(nid)[:384] for nid in ids])  # (T, 384)
            seq_t = torch.tensor(seq, dtype=torch.float32, device=self.device).unsqueeze(0)
            hist_emb = self.history_encoder(seq_t).squeeze(0).cpu().numpy()     # (128,)
        else:
            hist_emb = np.zeros(128, dtype=np.float32)

        # bio embedding (zeros — MIND has no signup data)
        bio_emb = np.zeros(64, dtype=np.float32)

        # affect (offline: neutral)
        affect_emb = np.zeros(2, dtype=np.float32)

        # context features (12-d)
        t = time.gmtime()
        hour_sin  = np.sin(2 * np.pi * t.tm_hour / 24)
        hour_cos  = np.cos(2 * np.pi * t.tm_hour / 24)
        day_sin   = np.sin(2 * np.pi * t.tm_wday / 7)
        day_cos   = np.cos(2 * np.pi * t.tm_wday / 7)
        sess_len  = min(interaction_count / 100.0, 1.0)
        diversity_hunger = max(0.0, 1.0 - interaction_count / 50.0)
        kg_affinity = 0.5     # neutral offline
        skip_ratio  = 0.0
        source_div  = 0.5
        cat_entropy = 0.5
        recency     = 1.0     # all impressions are "recent" in offline setting
        mood_slot   = 0.5

        ctx = np.array([
            hour_sin, hour_cos, day_sin, day_cos,
            sess_len, diversity_hunger, mood_slot, float(interaction_count % 20) / 20.0,
            kg_affinity, skip_ratio, source_div, cat_entropy,
        ], dtype=np.float32)

        return np.concatenate([hist_emb, bio_emb, affect_emb, ctx])  # (206,)


# ══════════════════════════════════════════════════════════════════════════════
# Reward function (matches doc §7.2)
# ══════════════════════════════════════════════════════════════════════════════

REWARD_CLICK     =  0.20
REWARD_SAVE      =  0.30
REWARD_DWELL     =  0.40   # × normalised dwell (0–1)
REWARD_SKIP      = -0.10
REWARD_NOT_INTER = -0.15
REWARD_KG_BONUS  =  0.05

def compute_reward(clicked: bool, skip: bool = False) -> float:
    """
    Offline reward from MIND (only click signal available).
    In online mode the full reward function from app.py applies.
    """
    if clicked:
        return REWARD_CLICK + REWARD_DWELL * 0.5  # assume average dwell offline
    if skip:
        return REWARD_SKIP
    return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Phase A: Offline training loop
# ══════════════════════════════════════════════════════════════════════════════

def train_phase_a(
    cfg: TierConfig,
    impressions: List[Impression],
    article_index: ArticleEmbeddingIndex,
    agent: DoubleDQNAgent,
    state_builder: StateBuilder,
    logging_policy: LoggingPolicy,
    output_dir: str = "models",
    metrics_path: str = "training_metrics.jsonl",
) -> DoubleDQNAgent:

    os.makedirs(output_dir, exist_ok=True)
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_path).touch()
    device = torch.device(cfg.device)
    metrics: List[Dict] = []
    total_steps = 0
    running_loss = 0.0
    running_reward = 0.0

    # ── Build feature matrix for logging policy (uses article popularity proxy) ──
    print("  Fitting logging policy on impression data …")
    lp_features, lp_labels = [], []
    for imp in impressions[:5000]:   # subset for speed
        for nid, label in [(n, 1) for n in imp.clicked] + [(n, 0) for n in imp.non_clicked[:3]]:
            feat = article_index.get(nid)[:384]
            lp_features.append(feat)
            lp_labels.append(label)
    if lp_features:
        logging_policy.fit(np.stack(lp_features), np.array(lp_labels))
        logging_policy.save(os.path.join(output_dir, "logging_policy.pkl"))
    print(f"  Logging policy fitted on {len(lp_labels)} samples.")

    print(f"\n{'='*60}")
    print(f"  Phase A — Offline DQN Training  [{cfg.name}]")
    print(f"  Impressions: {len(impressions):,} | Batch: {cfg.batch_size} | AMP: {cfg.use_amp}")
    print(f"{'='*60}")

    optimizer_steps = 0
    for epoch in range(1):   # single pass over offline data is standard
        random.shuffle(impressions)
        for step_i, imp in enumerate(impressions):
            if not imp.clicked:
                continue   # skip impressions with no positive signal

            # Build state
            state = state_builder.build(imp.history, interaction_count=step_i % 30)

            # Candidate pool (clicked + sample of non-clicked, capped to pool_size)
            pool_ids = imp.clicked[:]
            neg_sample = random.sample(imp.non_clicked, min(cfg.pool_size - len(pool_ids), len(imp.non_clicked)))
            pool_ids += neg_sample
            if not pool_ids:
                continue
            pool_items = article_index.get_batch(pool_ids)   # (N, 448)

            # IPS weight for clicked article
            clicked_feat = article_index.get(imp.clicked[0])[:384].reshape(1, -1)
            ips_w = float(logging_policy.ips_weight(clicked_feat)[0])

            # For each clicked article, create one transition
            for c_id in imp.clicked[:1]:   # one positive per impression for stability
                if c_id not in pool_ids:
                    continue
                action = pool_ids.index(c_id)
                reward = compute_reward(clicked=True)

                # Next state (same user, one more interaction)
                next_history = imp.history + [c_id]
                next_state = state_builder.build(next_history, interaction_count=step_i % 30 + 1)
                next_pool_items = pool_items  # reuse same pool for offline

                t = Transition(
                    state=state,
                    items=pool_items,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    next_items=next_pool_items,
                    done=False,
                    weight=ips_w,
                )
                agent.push(t)

            # Also add negative transitions (skipped articles)
            for nc_id in neg_sample[:3]:
                if nc_id not in pool_ids:
                    continue
                action = pool_ids.index(nc_id)
                nc_feat = article_index.get(nc_id)[:384].reshape(1, -1)
                nc_ips  = float(logging_policy.ips_weight(nc_feat)[0])
                t = Transition(
                    state=state,
                    items=pool_items,
                    action=action,
                    reward=compute_reward(clicked=False, skip=True),
                    next_state=state,
                    next_items=pool_items,
                    done=False,
                    weight=nc_ips,
                )
                agent.push(t)

            # Update DQN (accumulate gradients)
            if len(agent.buffer) >= cfg.batch_size:
                loss = agent.update()
                if loss is not None:
                    running_loss += loss
                    optimizer_steps += 1

            total_steps += 1

            # Logging
            if total_steps % cfg.log_every == 0:
                avg_loss = running_loss / max(optimizer_steps, 1)
                print(
                    f"  step={total_steps:>7} | ε={agent.epsilon:.3f} | "
                    f"loss={avg_loss:.4f} | buffer={len(agent.buffer):,} | "
                    f"updates={agent.updates_done:,}"
                )
                metrics.append({
                    "step": total_steps,
                    "epsilon": agent.epsilon,
                    "loss": avg_loss,
                    "buffer_size": len(agent.buffer),
                })
                with open(metrics_path, "a") as mf:
                    mf.write(json.dumps(metrics[-1]) + "\n")
                running_loss = 0.0
                optimizer_steps = 0

            # Checkpoint
            if total_steps % cfg.checkpoint_every == 0:
                ckpt = os.path.join(output_dir, f"dqn_step{total_steps}.pt")
                agent.save(ckpt)
                print(f"  ✓ Checkpoint saved → {ckpt}")

    # Final save
    agent.save(os.path.join(output_dir, "dqn_policy.pt"))
    print(f"\n  ✓ Phase A complete. Final model → {output_dir}/dqn_policy.pt")
    return agent


# ══════════════════════════════════════════════════════════════════════════════
# Offline evaluation (IPS-corrected nDCG@10)
# ══════════════════════════════════════════════════════════════════════════════

def ndcg_at_k(ranked: List[int], relevant: set, k: int) -> float:
    dcg, idcg = 0.0, 0.0
    for i, idx in enumerate(ranked[:k]):
        if i < len(relevant):
            idcg += 1.0 / np.log2(i + 2)
        if idx in relevant:
            dcg += 1.0 / np.log2(i + 2)
    return dcg / max(idcg, 1e-9)


@torch.no_grad()
def evaluate(
    agent: DoubleDQNAgent,
    impressions: List[Impression],
    article_index: ArticleEmbeddingIndex,
    state_builder: StateBuilder,
    logging_policy: LoggingPolicy,
    k: int = 10,
    max_eval: int = 1000,
) -> Dict[str, float]:
    agent.online_net.eval()
    ndcg_scores, auc_scores, ips_rewards = [], [], []

    for imp in impressions[:max_eval]:
        if not imp.clicked or not imp.non_clicked:
            continue

        state = state_builder.build(imp.history)
        cfg_pool = min(50, len(imp.non_clicked))
        pool_ids = imp.clicked + imp.non_clicked[:cfg_pool]
        clicked_set = set(range(len(imp.clicked)))

        pool_items = article_index.get_batch(pool_ids)
        ranked_indices = agent.rank(state, pool_items, k=k, explore=False).tolist()

        ndcg = ndcg_at_k(ranked_indices, clicked_set, k)
        ndcg_scores.append(ndcg)

        # IPS-corrected reward
        lp_feat = np.stack([article_index.get(nid)[:384] for nid in pool_ids])
        weights = logging_policy.ips_weight(lp_feat)
        reward_vec = np.array([1.0 if i in clicked_set else 0.0 for i in range(len(pool_ids))])
        ips_r = float((weights * reward_vec).mean())
        ips_rewards.append(ips_r)

    return {
        "ndcg_at_10": float(np.mean(ndcg_scores)),
        "ips_corrected_reward": float(np.mean(ips_rewards)),
        "n_evaluated": len(ndcg_scores),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point (called by tier scripts)
# ══════════════════════════════════════════════════════════════════════════════

def main(cfg: TierConfig, args: argparse.Namespace):
    print(f"\n{'='*60}")
    print(f"  HyperNews DQN Training  —  {cfg.name}")
    print(f"  Device: {cfg.device}  |  AMP: {cfg.use_amp}")
    print(f"{'='*60}\n")

    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

    # ── Locate MIND data ─────────────────────────────────────────────────────
    mind_path = Path(args.mind_path)
    train_news_path = mind_path / "train" / "news.tsv"
    train_behav_path = mind_path / "train" / "behaviors.tsv"
    dev_news_path = mind_path / "dev" / "news.tsv"
    dev_behav_path = mind_path / "dev" / "behaviors.tsv"

    for p in [train_news_path, train_behav_path]:
        if not p.exists():
            # try flat structure
            alt = mind_path / p.name
            if alt.exists():
                if "news" in p.name:
                    train_news_path = alt
                else:
                    train_behav_path = alt
            else:
                print(f"  ERROR: Cannot find {p}")
                sys.exit(1)

    print("  Loading MIND …")
    print(f"    news  : {train_news_path}")
    print(f"    behaviors: {train_behav_path}")
    news = parse_news(str(train_news_path))
    train_impr = parse_behaviors(str(train_behav_path), max_rows=cfg.max_impressions)
    print(f"    → {len(news):,} articles  |  {len(train_impr):,} impressions")

    # ── Article embedding index ───────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    emb_path  = data_dir / "article_embeddings.npy"
    ids_path  = data_dir / "article_ids.json"

    if not emb_path.exists():
        print(f"  ERROR: {emb_path} not found. Run generate_data.py first.")
        sys.exit(1)

    if not ids_path.exists():
        # Build id list from news dict order (matches generate_data.py)
        id_list = list(news.keys())
        with open(ids_path, "w") as f:
            json.dump(id_list, f)
        print(f"  Created {ids_path} with {len(id_list):,} IDs")

    article_index = ArticleEmbeddingIndex(
        str(emb_path), str(ids_path), kg_emb_path=cfg.kg_emb_path
    )
    print(f"  Article embedding index: {article_index.article_emb.shape}")

    # ── Models ────────────────────────────────────────────────────────────────
    history_encoder = HistoryEncoder(
        input_dim=384,
        d_model=cfg.history_d_model,
        output_dim=128,
        nhead=cfg.history_nhead,
        num_layers=cfg.history_layers,
    ).to(device)
    history_encoder.eval()

    bio_encoder = None  # no signup data in MIND — bio_emb stays zeros offline

    state_builder = StateBuilder(article_index, history_encoder, bio_encoder, device)

    agent = DoubleDQNAgent(
        state_dim=STATE_DIM,
        item_dim=ITEM_DIM,
        gamma=0.95,
        lr=1e-4,
        batch_size=cfg.batch_size,
        target_update_freq=cfg.target_update_freq,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=cfg.epsilon_decay_steps,
        replay_capacity=cfg.replay_capacity,
        device=str(device),
        use_amp=cfg.use_amp,
    )

    # Resume from checkpoint if given
    if args.resume and os.path.exists(args.resume):
        agent.load(args.resume)
        print(f"  Resumed from {args.resume}")

    logging_policy = LoggingPolicy.load_or_create(
        os.path.join(args.output_dir, "logging_policy.pkl")
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    if "A" in args.phase.upper():
        agent = train_phase_a(
            cfg, train_impr, article_index, agent, state_builder,
            logging_policy, output_dir=args.output_dir,
            metrics_path=os.path.join(args.output_dir, "metrics.jsonl"),
        )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n  Running offline evaluation …")
    dev_impr = []
    if dev_behav_path.exists():
        dev_impr = parse_behaviors(str(dev_behav_path), max_rows=2000)
    else:
        dev_impr = train_impr[-2000:]   # fallback: last portion of train

    results = evaluate(agent, dev_impr, article_index, state_builder, logging_policy)
    print(f"\n  {'─'*40}")
    print(f"  nDCG@10                 : {results['ndcg_at_10']:.4f}  (target >0.40)")
    print(f"  IPS-corrected reward    : {results['ips_corrected_reward']:.4f}")
    print(f"  Samples evaluated       : {results['n_evaluated']}")
    print(f"  {'─'*40}")

    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  All outputs → {args.output_dir}/")
    print("  Done ✓\n")
