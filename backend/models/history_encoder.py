"""
HistoryEncoder — 4-layer Transformer that encodes a user's last N clicked
articles into a single 128-d session embedding.

Designed to be lightweight enough to fit in 4 GB VRAM (RTX 2050) with fp16
and batch_size=64, while still being the same architecture used on the H100.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class HistoryEncoder(nn.Module):
    """
    Input:  (B, T, 384)  — sequence of article embeddings (all-MiniLM-L6-v2)
    Output: (B, 128)     — fused session representation
    """

    def __init__(
        self,
        input_dim: int = 384,
        d_model: int = 256,
        output_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 50,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len + 1, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        article_embs: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        article_embs: (B, T, 384)
        padding_mask:  (B, T) bool — True where padded (ignored positions)
        returns:       (B, 128)
        """
        x = self.input_proj(article_embs)          # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        # mean-pool over valid (non-padded) positions
        if padding_mask is not None:
            mask = (~padding_mask).float().unsqueeze(-1)  # (B, T, 1)
            x = (x * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            x = x.mean(1)
        return self.output_proj(x)                 # (B, 128)


def build_history_encoder(**kwargs) -> HistoryEncoder:
    return HistoryEncoder(**kwargs)
