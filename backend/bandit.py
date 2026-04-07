"""
Lightweight contextual bandit used by the RL ranking path.

The previous neural implementation imported torch during backend startup,
which made simple commands like `uvicorn backend.app:app` and
`python backend/evaluate_mind.py` stall on environments without a healthy
torch install. This module preserves the same public API using a numpy
LinUCB-style model so the backend stays responsive.
"""

from __future__ import annotations

import os
import pickle
from collections import deque
from typing import Iterable

import numpy as np


def _as_float32(vector) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim != 1:
        return arr.reshape(-1).astype(np.float32)
    return arr


class LinUCBBandit:
    """
    Numpy LinUCB bandit with small article/category priors for compatibility.

    The model keeps a single shared contextual estimator. That is enough for the
    current ranking pipeline, keeps memory modest, and avoids pulling in the
    full training stack during backend startup.
    """

    def __init__(
        self,
        context_dim: int = 391,
        alpha: float = 0.35,
        epsilon: float = 0.05,
        ridge_lambda: float = 1.0,
        replay_capacity: int = 8000,
        **_ignored,
    ):
        self.context_dim = int(context_dim)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.ridge_lambda = float(max(ridge_lambda, 1e-6))
        self.replay_capacity = int(max(replay_capacity, 1))

        self.A = np.eye(self.context_dim, dtype=np.float32) * self.ridge_lambda
        self.A_inv = np.eye(self.context_dim, dtype=np.float32) / self.ridge_lambda
        self.b = np.zeros(self.context_dim, dtype=np.float32)

        self.context_buffer: deque[np.ndarray] = deque(maxlen=self.replay_capacity)
        self.reward_buffer: deque[float] = deque(maxlen=self.replay_capacity)
        self.article_stats: dict[str, dict[str, float]] = {}
        self.category_stats: dict[str, dict[str, float]] = {}
        self.total_updates = 0

    def _normalize_reward(self, reward: float) -> float:
        value = float(reward)
        return float(np.clip(value, 0.0, 1.0))

    def _article_prior(self, article_id: str) -> tuple[float, float]:
        stats = self.article_stats.get(article_id)
        if not stats:
            return 0.0, 0.0
        mean_reward = float(stats.get("mean_reward", 0.0))
        count = float(stats.get("count", 0.0))
        prior_bonus = 0.12 * mean_reward
        explore_bonus = 0.05 / np.sqrt(max(count, 1.0))
        return prior_bonus, explore_bonus

    def _update_article_stats(self, article_id: str, normalized_reward: float):
        stats = self.article_stats.setdefault(article_id, {"count": 0.0, "mean_reward": 0.0})
        count = float(stats["count"])
        mean_reward = float(stats["mean_reward"])
        new_count = count + 1.0
        stats["count"] = new_count
        stats["mean_reward"] = mean_reward + ((normalized_reward - mean_reward) / new_count)

    def _update_category_stats(self, category: str, normalized_reward: float):
        key = str(category).strip().lower()
        if not key:
            return
        stats = self.category_stats.setdefault(key, {"count": 0.0, "mean_reward": 0.0})
        count = float(stats["count"])
        mean_reward = float(stats["mean_reward"])
        new_count = count + 1.0
        stats["count"] = new_count
        stats["mean_reward"] = mean_reward + ((normalized_reward - mean_reward) / new_count)

    def _category_prior(self, category: str) -> tuple[float, float]:
        key = str(category).strip().lower()
        stats = self.category_stats.get(key)
        if not stats:
            return 0.0, 0.0
        mean_reward = float(stats.get("mean_reward", 0.0))
        count = float(stats.get("count", 0.0))
        prior_bonus = 0.08 * mean_reward
        explore_bonus = 0.03 / np.sqrt(max(count, 1.0))
        return prior_bonus, explore_bonus

    def propagate_category_reward(
        self,
        category: str,
        sibling_article_ids: list,
        reward: float,
        decay: float = 0.3,
    ):
        propagated = float(np.clip(float(reward) * float(decay), 0.0, 1.0))
        for article_id in sibling_article_ids:
            self._update_article_stats(str(article_id), propagated)
        self._update_category_stats(category, propagated)

    def _theta(self) -> np.ndarray:
        return self.A_inv @ self.b

    def _predict(self, context: np.ndarray) -> tuple[float, float]:
        vector = _as_float32(context)
        if vector.shape[0] != self.context_dim:
            raise ValueError(f"Expected context_dim={self.context_dim}, got {vector.shape[0]}")

        theta = self._theta()
        mean_reward = float(np.clip(theta @ vector, 0.0, 1.0))
        variance = float(vector @ self.A_inv @ vector)
        uncertainty = float(np.sqrt(max(variance, 0.0)))
        return mean_reward, uncertainty

    def score(self, article_id: str, context: np.ndarray, category: str = "") -> float:
        predicted_reward, uncertainty = self._predict(context)
        prior_bonus, article_explore = self._article_prior(str(article_id))
        cat_prior_bonus, cat_explore = self._category_prior(category)

        if np.random.random() < self.epsilon:
            base = float(
                np.clip(
                    np.random.normal(predicted_reward, max(uncertainty, 1e-6)),
                    0.0,
                    1.0,
                )
            )
        else:
            base = predicted_reward + (self.alpha * uncertainty)

        return float(
            np.clip(
                base + prior_bonus + article_explore + cat_prior_bonus + cat_explore,
                0.0,
                1.5,
            )
        )

    def _append_example(self, context: np.ndarray, normalized_reward: float):
        self.context_buffer.append(_as_float32(context))
        self.reward_buffer.append(float(normalized_reward))

    def _sherman_morrison_update(self, vector: np.ndarray):
        av = self.A_inv @ vector
        denominator = float(1.0 + (vector @ av))
        self.A += np.outer(vector, vector).astype(np.float32)
        if denominator <= 1e-6:
            self.A_inv = np.linalg.pinv(self.A).astype(np.float32)
            return
        self.A_inv -= (np.outer(av, av) / denominator).astype(np.float32)

    def update(self, article_id: str, context: np.ndarray, reward: float):
        self.update_batch([article_id], [context], [reward])

    def update_batch(
        self,
        article_ids: Iterable[str],
        contexts: Iterable[np.ndarray],
        rewards: Iterable[float],
    ):
        for article_id, context, reward in zip(article_ids, contexts, rewards):
            vector = _as_float32(context)
            if vector.shape[0] != self.context_dim:
                raise ValueError(f"Expected context_dim={self.context_dim}, got {vector.shape[0]}")

            normalized_reward = self._normalize_reward(reward)
            self._append_example(vector, normalized_reward)
            self._sherman_morrison_update(vector)
            self.b += (normalized_reward * vector).astype(np.float32)
            self._update_article_stats(str(article_id), normalized_reward)
            self.total_updates += 1

    def rank(self, candidates: list[dict], context: np.ndarray) -> list[dict]:
        scored = []
        for article in candidates:
            article_id = article.get("news_id") or article.get("id")
            if not article_id:
                continue
            category = str(article.get("category", ""))
            bandit_score = self.score(str(article_id), context, category=category)
            scored.append({**article, "ucb_score": bandit_score})
        return sorted(scored, key=lambda item: item["ucb_score"], reverse=True)

    def save(self, path: str = "bandit.pkl"):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        payload = {
            "model_type": "LinUCBBandit",
            "config": {
                "context_dim": self.context_dim,
                "alpha": self.alpha,
                "epsilon": self.epsilon,
                "ridge_lambda": self.ridge_lambda,
                "replay_capacity": self.replay_capacity,
            },
            "A": self.A,
            "A_inv": self.A_inv,
            "b": self.b,
            "contexts": np.stack(list(self.context_buffer)).astype(np.float32)
            if self.context_buffer
            else np.empty((0, self.context_dim), dtype=np.float32),
            "rewards": np.asarray(list(self.reward_buffer), dtype=np.float32),
            "article_stats": self.article_stats,
            "category_stats": self.category_stats,
            "total_updates": self.total_updates,
        }

        with open(path, "wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, path: str = "bandit.pkl") -> "LinUCBBandit":
        with open(path, "rb") as handle:
            payload = pickle.load(handle)

        model_type = payload.get("model_type")
        if model_type not in {"LinUCBBandit"}:
            raise ValueError(f"Unsupported bandit state in {path}: {model_type}")

        config = dict(payload.get("config", {}))
        inst = cls(**config)
        inst.A = np.asarray(payload["A"], dtype=np.float32)
        inst.A_inv = np.asarray(payload["A_inv"], dtype=np.float32)
        inst.b = np.asarray(payload["b"], dtype=np.float32)

        for context in payload.get("contexts", []):
            inst.context_buffer.append(_as_float32(context))
        for reward in payload.get("rewards", []):
            inst.reward_buffer.append(float(reward))

        inst.article_stats = {
            str(article_id): {
                "count": float(stats.get("count", 0.0)),
                "mean_reward": float(stats.get("mean_reward", 0.0)),
            }
            for article_id, stats in payload.get("article_stats", {}).items()
        }
        inst.category_stats = {
            str(category): {
                "count": float(stats.get("count", 0.0)),
                "mean_reward": float(stats.get("mean_reward", 0.0)),
            }
            for category, stats in payload.get("category_stats", {}).items()
        }
        inst.total_updates = int(payload.get("total_updates", len(inst.reward_buffer)))
        return inst

    @classmethod
    def load_or_create(cls, path: str = "bandit.pkl", **kwargs) -> "LinUCBBandit":
        if os.path.exists(path):
            try:
                inst = cls.load(path)
                expected_dim = kwargs.get("context_dim")
                if expected_dim and inst.context_dim != int(expected_dim):
                    return cls(**kwargs)
                if "alpha" in kwargs:
                    inst.alpha = float(kwargs["alpha"])
                if "epsilon" in kwargs:
                    inst.epsilon = float(kwargs["epsilon"])
                return inst
            except Exception as exc:
                print(f"Bandit state load failed for {path}: {exc}. Recreating bandit state.")
                legacy_path = f"{path}.legacy"
                try:
                    if not os.path.exists(legacy_path):
                        os.replace(path, legacy_path)
                        print(f"Moved incompatible bandit state to {legacy_path}")
                except OSError:
                    pass
        return cls(**kwargs)


NeuralContextualBandit = LinUCBBandit
