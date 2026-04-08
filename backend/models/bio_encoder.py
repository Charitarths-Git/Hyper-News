"""
BioEncoder — MLP that encodes user signup / onboarding fields into a 64-d
bio_emb used for cold-start ranking.

Text dominates (Ye et al. ECIR 2026), so the encoder is text-heavy:
  • free-text interest phrases  → SentenceTransformer → 384-d
  • categorical fields          → embedding lookup  → 4–16-d each
  • combined MLP                → 64-d bio_emb

The model is tiny and trains in <60 s on any GPU tier.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

try:
    from backend.bio_categories import BIO_CATEGORY_ORDER
except ImportError:
    from bio_categories import BIO_CATEGORY_ORDER


# ── Categorical vocabulary sizes (kept small on purpose) ──────────────────────
_AGE_BUCKETS   = ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+", "unknown"]
_GENDER_CATS   = ["male", "female", "nonbinary", "prefer_not", "unknown"]
_OCCUPATION_CATS = [
    "student", "engineer", "teacher", "doctor", "lawyer", "journalist",
    "artist", "finance", "government", "retail", "other", "unknown",
]
_LOCATION_CATS = ["north_america", "europe", "asia", "latin_america", "africa", "oceania", "unknown"]

VOCABS: Dict[str, List[str]] = {
    "age_bucket": _AGE_BUCKETS,
    "gender": _GENDER_CATS,
    "occupation": _OCCUPATION_CATS,
    "location_region": _LOCATION_CATS,
}

BIO_CATEGORY_DIM = len(BIO_CATEGORY_ORDER)
CAT_EMB_DIM = 8   # embedding dim per categorical field
TEXT_DIM    = 384  # SentenceTransformer all-MiniLM-L6-v2
BIO_DIM     = 64   # output


class BioEncoder(nn.Module):
    """
    Input features:
        cat_indices  : (B, 4) int64  — one index per categorical field
        text_emb     : (B, 384) float32 — mean-pooled interest phrase embedding
    Output: (B, 64)
    """

    def __init__(self):
        super().__init__()
        # one embedding table per categorical field
        self.cat_embeddings = nn.ModuleDict(
            {
                field: nn.Embedding(len(vocab) + 1, CAT_EMB_DIM, padding_idx=len(vocab))
                for field, vocab in VOCABS.items()
            }
        )
        cat_total = CAT_EMB_DIM * len(VOCABS)
        mlp_input = TEXT_DIM + cat_total

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, BIO_DIM),
            nn.LayerNorm(BIO_DIM),
        )

    def forward(
        self,
        cat_indices: torch.Tensor,   # (B, 4)
        text_emb: torch.Tensor,       # (B, 384)
    ) -> torch.Tensor:               # (B, 64)
        cat_parts = []
        for i, field in enumerate(VOCABS):
            cat_parts.append(self.cat_embeddings[field](cat_indices[:, i]))
        cat_concat = torch.cat(cat_parts, dim=-1)           # (B, 32)
        x = torch.cat([text_emb, cat_concat], dim=-1)       # (B, 416)
        return self.mlp(x)                                   # (B, 64)


# ── Helpers for inference ──────────────────────────────────────────────────────

def field_to_index(field: str, value: str) -> int:
    vocab = VOCABS[field]
    v = str(value).strip().lower()
    return vocab.index(v) if v in vocab else len(vocab)  # OOV → padding_idx


def encode_bio_fields(
    age_bucket: str = "unknown",
    gender: str = "unknown",
    occupation: str = "unknown",
    location_region: str = "unknown",
    interest_text_emb: Optional[np.ndarray] = None,
    model: Optional[BioEncoder] = None,
    device: str = "cpu",
) -> np.ndarray:
    """
    Convenience function for online inference (single user).
    Returns 64-d numpy float32 bio_emb.
    """
    cat = np.array(
        [
            field_to_index("age_bucket", age_bucket),
            field_to_index("gender", gender),
            field_to_index("occupation", occupation),
            field_to_index("location_region", location_region),
        ],
        dtype=np.int64,
    )
    if interest_text_emb is None:
        interest_text_emb = np.zeros(TEXT_DIM, dtype=np.float32)

    if model is None:
        # fallback: zero-init projection (just returns zeros for cold-start)
        return np.zeros(BIO_DIM, dtype=np.float32)

    model.eval()
    with torch.no_grad():
        cat_t = torch.tensor(cat, dtype=torch.long, device=device).unsqueeze(0)
        txt_t = torch.tensor(interest_text_emb, dtype=torch.float32, device=device).unsqueeze(0)
        out = model(cat_t, txt_t)
    return out.squeeze(0).cpu().numpy()


def save_bio_encoder(model: BioEncoder, path: str = "models/bio_encoder.pt"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Saved bio encoder → {path}")


def load_bio_encoder(path: str = "models/bio_encoder.pt", device: str = "cpu") -> BioEncoder:
    model = BioEncoder().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
