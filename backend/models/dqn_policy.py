"""
Double DQN with dueling architecture for HyperNews feed ranking.

State  : 206-d fused vector
         [history_emb(128) | bio_emb(64) | affect_emb(2) | context_feats(12)]
Action : index into a candidate pool of ≤200 articles (pre-filtered by KG)
         → Q-value per candidate, select top-k by Q-value
Output : ordered slate of k=15 articles

Training: offline on MIND behaviors.tsv with IPS correction (Yan et al. SIGIR)
           then online fine-tuning from /feedback events.

References:
  Double DQN — van Hasselt et al. (2016)
  Dueling DQN — Wang et al. (2016)
  Offline RL safety — Yan et al. (SIGIR 2016)
"""

from __future__ import annotations

import os
import random
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Dimensions ────────────────────────────────────────────────────────────────
STATE_DIM   = 206   # history(128) + bio(64) + affect(2) + context(12)
ITEM_DIM    = 448   # article_emb(384) + kg_emb(64)
HIDDEN_DIM  = 512
SLATE_SIZE  = 15
POOL_SIZE   = 200   # candidate pool per request


# ── Dueling DQN network ────────────────────────────────────────────────────────
class DuelingQNetwork(nn.Module):
    """
    Computes Q(state, item_i) for every item in the candidate pool.

    The state–item pair is concatenated and passed through a shared trunk,
    then split into value (V) and advantage (A) streams:
        Q(s,a) = V(s) + A(s,a) − mean_a A(s,a)

    Input:
        state : (B, STATE_DIM)
        items  : (B, N, ITEM_DIM)   N = candidate pool size
    Output:
        q_vals : (B, N)
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        item_dim: int = ITEM_DIM,
        hidden: int = HIDDEN_DIM,
    ):
        super().__init__()
        inp = state_dim + item_dim

        self.trunk = nn.Sequential(
            nn.Linear(inp, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
        )
        # Value stream
        self.value_head = nn.Sequential(
            nn.Linear(hidden // 2, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        # Advantage stream
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden // 2, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        state: torch.Tensor,    # (B, STATE_DIM)
        items: torch.Tensor,    # (B, N, ITEM_DIM)
    ) -> torch.Tensor:          # (B, N)
        B, N, _ = items.shape
        s_exp = state.unsqueeze(1).expand(-1, N, -1)        # (B, N, STATE_DIM)
        x = torch.cat([s_exp, items], dim=-1)               # (B, N, STATE_DIM+ITEM_DIM)
        x = x.reshape(B * N, -1)
        h = self.trunk(x)                                   # (B*N, hidden//2)
        v = self.value_head(h).reshape(B, N)                # (B, N)
        a = self.advantage_head(h).reshape(B, N)            # (B, N)
        q = v + (a - a.mean(dim=1, keepdim=True))           # dueling combination
        return q


# ── Replay Buffer ──────────────────────────────────────────────────────────────
@dataclass
class Transition:
    state:      np.ndarray   # (STATE_DIM,)
    items:      np.ndarray   # (N, ITEM_DIM) — candidate pool at decision time
    action:     int          # index into items that was selected
    reward:     float
    next_state: np.ndarray   # (STATE_DIM,)
    next_items: np.ndarray   # (N, ITEM_DIM)
    done:       bool
    weight:     float = 1.0  # IPS propensity weight


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, t: Transition):
        self.buffer.append(t)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


# ── Double DQN Agent ───────────────────────────────────────────────────────────
class DoubleDQNAgent:
    """
    Offline-first Double DQN agent.

    Training phases:
      Phase A (offline) — train on MIND behaviors with IPS correction.
      Phase B (online)  — fine-tune from /feedback events in production.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        item_dim: int = ITEM_DIM,
        hidden: int = HIDDEN_DIM,
        gamma: float = 0.95,
        lr: float = 1e-4,
        batch_size: int = 512,
        target_update_freq: int = 1_000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50_000,
        replay_capacity: int = 50_000,
        device: str = "cpu",
        use_amp: bool = False,
    ):
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)
        self.use_amp = use_amp and self.device.type == "cuda"

        self.online_net = DuelingQNetwork(state_dim, item_dim, hidden).to(self.device)
        self.target_net = DuelingQNetwork(state_dim, item_dim, hidden).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.buffer = ReplayBuffer(replay_capacity)

        # ε-greedy schedule
        self.epsilon_start = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_steps = epsilon_decay_steps
        self.steps_done    = 0
        self.updates_done  = 0

    @property
    def epsilon(self) -> float:
        frac = min(self.steps_done / max(self.epsilon_steps, 1), 1.0)
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (1 - frac)

    # ── inference ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def rank(
        self,
        state: np.ndarray,    # (STATE_DIM,)
        item_pool: np.ndarray,  # (N, ITEM_DIM)
        k: int = SLATE_SIZE,
        explore: bool = True,
    ) -> np.ndarray:
        """Returns indices of top-k items from item_pool ordered by Q-value."""
        self.steps_done += 1
        if explore and random.random() < self.epsilon:
            indices = np.random.permutation(len(item_pool))[:k]
            return indices.astype(np.int32)

        self.online_net.eval()
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        it = torch.tensor(item_pool, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            q = self.online_net(s, it).squeeze(0)  # (N,)
        top_k = torch.topk(q, k=min(k, len(item_pool))).indices
        return top_k.cpu().numpy().astype(np.int32)

    # ── training ───────────────────────────────────────────────────────────────
    def push(self, transition: Transition):
        self.buffer.push(transition)

    def update(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)
        loss = self._compute_loss(batch)
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.updates_done += 1
        if self.updates_done % self.target_update_freq == 0:
            self._sync_target()

        return float(loss.item())

    def _compute_loss(self, batch: List[Transition]) -> torch.Tensor:
        self.online_net.train()

        states  = torch.tensor(np.stack([t.state      for t in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([t.action for t in batch],               dtype=torch.long,    device=self.device)
        rewards = torch.tensor([t.reward for t in batch],               dtype=torch.float32, device=self.device)
        dones   = torch.tensor([t.done   for t in batch],               dtype=torch.float32, device=self.device)
        weights = torch.tensor([t.weight for t in batch],               dtype=torch.float32, device=self.device)

        # Handle variable pool sizes by zero-padding to max N
        max_n = max(t.items.shape[0] for t in batch)
        item_dim = batch[0].items.shape[1]
        items_pad      = np.zeros((len(batch), max_n, item_dim), dtype=np.float32)
        next_items_pad = np.zeros((len(batch), max_n, item_dim), dtype=np.float32)
        for i, t in enumerate(batch):
            n = t.items.shape[0]
            items_pad[i, :n]      = t.items
            next_items_pad[i, :n] = t.next_items

        items      = torch.tensor(items_pad,      dtype=torch.float32, device=self.device)
        next_items = torch.tensor(next_items_pad, dtype=torch.float32, device=self.device)
        next_states= torch.tensor(np.stack([t.next_state for t in batch]), dtype=torch.float32, device=self.device)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # Current Q(s,a)
            q_all = self.online_net(states, items)              # (B, N)
            q_sa  = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

            # Double DQN target: use online net to SELECT action, target net to EVALUATE
            with torch.no_grad():
                next_q_online = self.online_net(next_states, next_items)
                best_next_a   = next_q_online.argmax(dim=1)
                next_q_target = self.target_net(next_states, next_items)
                next_q_val    = next_q_target.gather(1, best_next_a.unsqueeze(1)).squeeze(1)

            target = rewards + self.gamma * next_q_val * (1 - dones)
            td_error = q_sa - target.detach()
            # IPS-weighted Huber loss
            loss = (weights * F.huber_loss(td_error, torch.zeros_like(td_error), reduction="none")).mean()

        return loss

    def _sync_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ── persistence ────────────────────────────────────────────────────────────
    def save(self, path: str = "models/dqn_policy.pt"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "online": self.online_net.state_dict(),
                "target": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps_done": self.steps_done,
                "updates_done": self.updates_done,
            },
            path,
        )

    def load(self, path: str = "models/dqn_policy.pt"):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online"])
        self.target_net.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps_done  = ckpt.get("steps_done", 0)
        self.updates_done = ckpt.get("updates_done", 0)
        print(f"Loaded DQN from {path} — steps={self.steps_done}, updates={self.updates_done}")
