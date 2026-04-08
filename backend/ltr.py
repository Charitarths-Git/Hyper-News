"""
LTR scorer with a deterministic fallback.

The project can use a LightGBM LambdaMART model when available, but the app
still needs to run in lightweight environments. This wrapper allows the
production-style interface without making local startup fragile.
"""

from __future__ import annotations

import os
import json
from typing import Iterable

import numpy as np


_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DEFAULT_LTR_MODEL_PATH = os.path.join(_BASE, "models", "ltr_model.txt")


class HybridLTRScorer:
    def __init__(self, model_path: str = DEFAULT_LTR_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.feature_order: list[str] = []
        self.model_loaded = False
        self.fallback_weights: dict[str, float] = {}

        if os.path.exists(model_path):
            try:
                import lightgbm as lgb

                self.model = lgb.Booster(model_file=model_path)
                self.feature_order = list(self.model.feature_name())
                self.model_loaded = True
            except Exception as exc:
                print(f"LTR model unavailable ({exc}); using deterministic fallback scorer.")
        if not self.model_loaded:
            weights_path = f"{model_path}.weights.json"
            if os.path.exists(weights_path):
                try:
                    with open(weights_path, "r", encoding="utf-8") as handle:
                        payload = json.load(handle)
                    self.fallback_weights = {
                        str(key): float(value)
                        for key, value in (payload.get("weights") or {}).items()
                    }
                except Exception as exc:
                    print(f"LTR weight fallback unavailable ({exc}); using deterministic fallback scorer.")

    def _fallback_score(self, features: dict[str, float]) -> float:
        if self.fallback_weights:
            return float(
                sum(
                    float(features.get(name, 0.0)) * float(weight)
                    for name, weight in self.fallback_weights.items()
                )
            )
        return float(
            (0.24 * float(features.get("semantic_score", 0.0)))
            + (0.18 * float(features.get("retrieval_score", 0.0)))
            + (0.15 * float(features.get("lexical_score", 0.0)))
            + (0.12 * float(features.get("memory_score", 0.0)))
            + (0.10 * float(features.get("kg_score", 0.0)))
            + (0.09 * float(features.get("entity_score", 0.0)))
            + (0.06 * float(features.get("subcategory_score", 0.0)))
            + (0.04 * float(features.get("popularity_score", 0.0)))
            + (0.04 * float(features.get("context_score", 0.0)))
            - (0.06 * float(features.get("negative_score", 0.0)))
        )

    def score(self, features: dict[str, float]) -> float:
        if self.model_loaded and self.model is not None and self.feature_order:
            row = np.asarray(
                [[float(features.get(name, 0.0)) for name in self.feature_order]],
                dtype=np.float32,
            )
            try:
                prediction = self.model.predict(row)
                return float(prediction[0])
            except Exception:
                pass
        return self._fallback_score(features)

    def score_many(self, rows: Iterable[dict[str, float]]) -> list[float]:
        return [self.score(features) for features in rows]
