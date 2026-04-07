"""
Logging Policy — assigns propensity scores P(action | state) to MIND impressions.

Required for safe offline DQN training (IPS correction, Yan et al. SIGIR 2016).
Without propensity weighting, offline RL evaluation is biased toward the
logging policy's distribution and will overestimate learned policy quality.

The logging policy is a simple logistic regression over article features.
It is intentionally weak so that IPS weights remain stable.
"""

from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class LoggingPolicy:
    """
    Estimates P(click | article_features) from MIND impression data.
    Used to compute importance-sampling weights for offline RL.
    """

    def __init__(self, clip_min: float = 0.05, clip_max: float = 5.0):
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.model: Optional[LogisticRegression] = None
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, features: np.ndarray, labels: np.ndarray):
        """
        features : (N, D) — article feature vectors at impression time
        labels   : (N,)   — 1 if clicked, 0 otherwise
        """
        X = self.scaler.fit_transform(features.astype(np.float32))
        self.model = LogisticRegression(
            max_iter=500, C=1.0, solver="lbfgs", class_weight="balanced"
        )
        self.model.fit(X, labels)
        self._fitted = True

    def propensity(self, features: np.ndarray) -> np.ndarray:
        """
        Returns P(click | features) clipped to [clip_min, 1].
        Shape: (N,)
        """
        if not self._fitted or self.model is None:
            return np.ones(len(features), dtype=np.float32)
        X = self.scaler.transform(features.astype(np.float32))
        probs = self.model.predict_proba(X)[:, 1].astype(np.float32)
        return np.clip(probs, 1e-4, 1.0)

    def ips_weight(self, features: np.ndarray) -> np.ndarray:
        """
        Returns clipped IPS weight = 1 / P(click | features).
        Shape: (N,)
        """
        p = self.propensity(features)
        weights = 1.0 / p
        return np.clip(weights, self.clip_min, self.clip_max).astype(np.float32)

    def save(self, path: str = "models/logging_policy.pkl"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler, "fitted": self._fitted}, f)
        print(f"Saved logging policy → {path}")

    @classmethod
    def load(cls, path: str = "models/logging_policy.pkl") -> "LoggingPolicy":
        inst = cls()
        with open(path, "rb") as f:
            d = pickle.load(f)
        inst.model   = d["model"]
        inst.scaler  = d["scaler"]
        inst._fitted = d.get("fitted", inst.model is not None)
        return inst

    @classmethod
    def load_or_create(cls, path: str = "models/logging_policy.pkl") -> "LoggingPolicy":
        if os.path.exists(path):
            try:
                return cls.load(path)
            except Exception as e:
                print(f"LoggingPolicy load failed: {e}. Creating fresh.")
        return cls()
