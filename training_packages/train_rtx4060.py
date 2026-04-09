"""
HyperNews DQN Training — RTX 4060 (8 GB VRAM)
================================================
Full 4-layer Transformer history encoder + Double DQN with dueling arch.
Mixed precision (fp16) enabled. Full MIND-large supported.

Run:
    python train_rtx4060.py \\
        --mind-path   /path/to/MIND-large \\
        --data-dir    /path/to/krishrepo/hyperpersonalisedNewsReccomendation/data \\
        --output-dir  ./output_rtx4060 \\
        --phase       A

Optional:
    --max-impressions 200000   (default: all)
    --resume          output_rtx4060/dqn_step50000.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train_dqn_core import TierConfig, main

# ── RTX 4060 config ────────────────────────────────────────────────────────────
CFG = TierConfig(
    name="RTX4060",
    device="cuda",
    use_amp=True,              # fp16 — saves ~1.5 GB VRAM
    batch_size=256,            # fits in 8 GB with pool_size=100
    replay_capacity=50_000,
    target_update_freq=1_000,
    history_nhead=4,           # 4-layer Transformer as per spec
    history_layers=4,
    history_d_model=256,
    epsilon_decay_steps=50_000,
    max_impressions=0,         # all rows
    pool_size=100,             # candidate pool (200 uses more VRAM)
    grad_accum=1,
    num_workers=4,
    checkpoint_every=10_000,
    log_every=500,
    kg_emb_path="",            # set to data/kg_embeddings.npy when Phase 5 done
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RTX 4060 DQN training")
    parser.add_argument("--mind-path", required=True, help="Path to MIND dataset root")
    parser.add_argument("--data-dir", required=True, help="Path to HyperNews data/ directory")
    parser.add_argument("--output-dir", default="./output_rtx4060")
    parser.add_argument("--phase", default="A", choices=["A", "B", "AB"])
    parser.add_argument("--max-impressions", type=int, default=CFG.max_impressions)
    parser.add_argument("--resume", default="", help="Path to checkpoint .pt to resume from")
    args = parser.parse_args()
    CFG.max_impressions = args.max_impressions
    main(CFG, args)
