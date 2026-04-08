"""
Optional transformer-history encoder compatibility shim.

The recommender already falls back to a weighted-mean user profile whenever
`encode_user_history()` returns None. Keeping that fallback lightweight is more
important than importing torch during every backend startup, so these helpers
gracefully no-op unless a future implementation restores an explicitly opt-in
encoder path.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

_MODEL_SAVE_PATH = "models/transformer_encoder.pt"


def encode_user_history(
    news_ids: list,
    article_embeddings: np.ndarray,
    news_id_to_idx: dict,
    seq_len: int = 20,
) -> Optional[np.ndarray]:
    return None


def save_encoder(path: str = _MODEL_SAVE_PATH):
    return False


def load_encoder(path: str = _MODEL_SAVE_PATH):
    return False
