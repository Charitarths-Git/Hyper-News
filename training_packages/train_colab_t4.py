"""
HyperNews DQN Training — Google Colab T4 (16 GB VRAM)
=======================================================
T4 sits between RTX 3060 and RTX 4060 in memory.
Use MIND-small. Mixed precision enabled.
This file can be pasted cell-by-cell into Colab or run as a script.

In Colab:
    !pip install -q torch faiss-cpu sentence-transformers lightgbm scikit-learn
    !git clone <your-repo> /content/krishrepo
    !python /content/krishrepo/training_packages/train_colab_t4.py \\
        --mind-path /content/MIND-small \\
        --data-dir  /content/krishrepo/hyperpersonalisedNewsReccomendation/data \\
        --output-dir /content/outputs

MIND-small download in Colab:
    !wget -q "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip"
    !wget -q "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip"
    !unzip -q MINDsmall_train.zip -d /content/MIND-small/train
    !unzip -q MINDsmall_dev.zip   -d /content/MIND-small/dev
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train_dqn_core import TierConfig, main

CFG_T4 = TierConfig(
    name="Colab-T4",
    device="cuda",
    use_amp=True,
    batch_size=256,            # T4 has 16 GB — comfortable at 256
    replay_capacity=50_000,
    target_update_freq=1_000,
    history_nhead=4,
    history_layers=4,          # full 4-layer Transformer
    history_d_model=256,
    epsilon_decay_steps=50_000,
    max_impressions=100_000,   # MIND-small ~72k train rows — use all
    pool_size=100,
    grad_accum=1,
    num_workers=2,             # Colab has 2 vCPUs typically
    checkpoint_every=10_000,
    log_every=200,
    kg_emb_path="",
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Colab T4 DQN training")
    parser.add_argument("--mind-path", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="/content/outputs")
    parser.add_argument("--phase", default="A", choices=["A", "B", "AB"])
    parser.add_argument("--max-impressions", type=int, default=CFG_T4.max_impressions)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()
    CFG_T4.max_impressions = args.max_impressions
    main(CFG_T4, args)
