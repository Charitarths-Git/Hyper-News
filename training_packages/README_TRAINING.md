# HyperNews — Training Packages

## Phase order (must follow this sequence)

```
Phase 1  → fix infra (already in repo)
Phase 2  → train_bio_encoder.py
Phase 4  → train_rtx4060.py / train_rtx3050.py / train_colab_t4.py / train_h100.py
Phase 5  → train_kg_gnn.py   (then retrain DQN with --kg-emb-path)
```

---

## Prerequisites

1. Run `generate_data.py` to build `data/article_embeddings.npy`
2. Create `data/article_ids.json`:
   ```bash
   .venv\Scripts\python.exe scripts/generate_article_ids.py
   ```
3. Download MIND dataset:
   - MIND-small: https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip
   - MIND-large: https://mind201910small.blob.core.windows.net/release/MINDlarge_train.zip
4. If you want a portable remote-training upload, build the repo-shaped bundle:
   ```bash
   .venv\Scripts\python.exe scripts/build_training_bundle.py
   ```

---

## GPU Tier Guide

| Script | GPU | VRAM | MIND | Batch | Notes |
|---|---|---|---|---|---|
| `train_colab_t4.py` | T4 | 16 GB | small | 256 | Free Colab GPU |
| `train_rtx4060.py` | RTX 4060 | 8 GB | large | 256 | fp16 required |
| `train_rtx3050.py` | RTX 3050/3060 | 4–6 GB | small | 128 | 2-layer Transformer |
| `train_h100.py` | H100 | 80 GB | large | 512 | GNN + DQN together |

---

## Quick Start — RTX 4060

```bash
pip install -r training_packages/requirements_train.txt

python training_packages/train_bio_encoder.py \
    --mind-path  /path/to/MIND-large \
    --data-dir   data \
    --output-dir outputs \
    --device     cuda

python training_packages/train_rtx4060.py \
    --mind-path  /path/to/MIND-large \
    --data-dir   data \
    --output-dir outputs \
    --phase      A
```

---

## Quick Start — Colab T4

```python
# Cell 1: Install
!pip install -q sentence-transformers faiss-cpu lightgbm scikit-learn

# Cell 2: Clone repo
!git clone https://github.com/YOUR/krishrepo /content/krishrepo
%cd /content/krishrepo

# Cell 3: Download MIND-small
!wget -q "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip"
!wget -q "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip"
!unzip -q MINDsmall_train.zip -d /content/MIND-small/train
!unzip -q MINDsmall_dev.zip   -d /content/MIND-small/dev

# Cell 4: Build article index
!python hyperpersonalisedNewsReccomendation/backend/generate_data.py

# Cell 5: Create article_ids.json
import pandas as pd, json
df = pd.read_parquet("hyperpersonalisedNewsReccomendation/data/articles.parquet")
json.dump(df["news_id"].tolist(),
          open("hyperpersonalisedNewsReccomendation/data/article_ids.json","w"))

# Cell 6: Train
!python training_packages/train_colab_t4.py \
    --mind-path /content/MIND-small \
    --data-dir  /content/krishrepo/hyperpersonalisedNewsReccomendation/data \
    --output-dir /content/outputs

# Cell 7: Copy model back
import shutil
shutil.copy("/content/outputs/dqn_policy.pt",
            "hyperpersonalisedNewsReccomendation/models/dqn_policy.pt")
```

---

## Quick Start — Lightning AI H100

```bash
# In Lightning AI Studio terminal:
pip install -r training_packages/requirements_train.txt

python training_packages/train_h100.py \
    --mind-path /teamspace/uploads/MIND-large \
    --data-dir  hyperpersonalisedNewsReccomendation/data \
    --output-dir /teamspace/studios/outputs \
    --phase AB \
    --run-gnn 1
```

---

## Output files

After training, copy these to your backend:

```
outputs/
├── dqn_policy.pt          → models/dqn_policy.pt
├── bio_encoder.pt         → models/bio_encoder.pt
├── kg_gnn.pt              → models/kg_gnn.pt
├── kg_embeddings.npy      → data/kg_embeddings.npy
├── logging_policy.pkl     → models/logging_policy.pkl
├── metrics.jsonl          → (training logs — keep for analysis)
└── eval_results.json      → (nDCG@10 + IPS reward)
```

---

## Target metrics

| Phase | nDCG@10 | Feed p95 |
|---|---|---|
| Baseline | 0.373 | ~300ms |
| Phase 2 +Bio | >0.38 | ~300ms |
| Phase 4 +DQN | >0.42 | <200ms |
| Phase 5 +GNN | >0.44 | <200ms |

---

## Critical rules (never break these)

1. **Feed must never call RAG** — DQN selects all feed candidates
2. **No raw face data** — only (valence, arousal) pair
3. **Always use IPS correction** — the logging_policy.pkl is non-negotiable
4. **Search context = ranking bonus only** — not candidate injection
5. **DQN before PPO** — DQN first (off-policy), PPO only after stable baseline
