"""
HyperNews DQN Training — RTX 3050 / 3060 (4–6 GB VRAM)
=========================================================
Reduced Transformer (2-layer), smaller batch, pool_size capped at 50.
Mixed precision mandatory. Use MIND-small for full run; MIND-large needs
--max-impressions 100000 to avoid OOM.

Run:
    python train_rtx3050.py \\
        --mind-path   /path/to/MIND-small \\
        --data-dir    /path/to/krishrepo/hyperpersonalisedNewsReccomendation/data \\
        --output-dir  ./output_rtx3050 \\
        --phase       A

Tips for 4 GB:
    --max-impressions 80000
    --batch-size      64
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train_dqn_core import TierConfig, main

CFG = TierConfig(
    name="RTX3050/3060",
    device="cuda",
    use_amp=True,              # REQUIRED on 4 GB — do not disable
    batch_size=128,            # reduce to 64 for 4 GB
    replay_capacity=30_000,    # smaller buffer
    target_update_freq=1_000,
    history_nhead=4,
    history_layers=2,          # 2-layer Transformer instead of 4
    history_d_model=128,       # smaller inner dim
    epsilon_decay_steps=30_000,
    max_impressions=100_000,   # cap for 4–6 GB safety
    pool_size=50,              # halved candidate pool
    grad_accum=2,              # simulate larger batch without memory cost
    num_workers=2,
    checkpoint_every=5_000,
    log_every=200,
    kg_emb_path="",
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RTX 3050/3060 DQN training")
    parser.add_argument("--mind-path", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="./output_rtx3050")
    parser.add_argument("--phase", default="A", choices=["A", "B", "AB"])
    parser.add_argument("--max-impressions", type=int, default=CFG.max_impressions)
    parser.add_argument("--batch-size", type=int, default=CFG.batch_size)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()
    CFG.max_impressions = args.max_impressions
    CFG.batch_size = args.batch_size
    main(CFG, args)
