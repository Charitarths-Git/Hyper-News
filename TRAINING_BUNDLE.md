# HyperNews Training Bundle

This repo now supports a portable training bundle for `RTX 4060` iteration and
`Lightning AI H100` final training without moving the full app.

## What it does

- keeps the local app, frontend, DB, FAISS indexes, and logs in place
- generates `data/article_ids.json` aligned to `article_embeddings.npy`
- builds a portable bundle with the exact layout the training scripts expect
- exports two zip files:
  - `hypernews_training_bundle.zip`
  - `hypernews_training_bundle_h100_gnn.zip`

## Commands

Generate the missing article ID index:

```powershell
.venv\Scripts\python.exe .\scripts\generate_article_ids.py
```

Build the portable bundle and export zips to `artifacts/` and `Downloads/`:

```powershell
.venv\Scripts\python.exe .\scripts\build_training_bundle.py
```

Smoke-test the exported bundle against the local `MIND-small` copy:

```powershell
.venv\Scripts\python.exe .\scripts\verify_training_bundle.py
```

## Bundle layout

```text
artifacts/
├── training_bundle/
│   ├── training_packages/
│   └── hyperpersonalisedNewsReccomendation/
├── training_bundle_h100_gnn/
│   ├── training_packages/
│   └── hyperpersonalisedNewsReccomendation/
└── zips/
    ├── hypernews_training_bundle.zip
    └── hypernews_training_bundle_h100_gnn.zip
```

## What to upload

### RTX 4060

Upload:

- `artifacts/training_bundle/` or `hypernews_training_bundle.zip`
- `MIND-small train + dev`

Run:

```powershell
python training_packages/train_bio_encoder.py --mind-path /path/to/MIND-small --data-dir hyperpersonalisedNewsReccomendation/data --output-dir outputs --device cuda
python training_packages/train_rtx4060.py --mind-path /path/to/MIND-small --data-dir hyperpersonalisedNewsReccomendation/data --output-dir outputs --phase A
```

### Lightning AI H100

Upload:

- `artifacts/training_bundle/` or `hypernews_training_bundle.zip`
- full `MIND-large train + dev`
- `artifacts/training_bundle_h100_gnn/` or `hypernews_training_bundle_h100_gnn.zip` only if running `--run-gnn 1`

Run:

```powershell
python training_packages/train_h100.py --mind-path /path/to/MIND-large --data-dir hyperpersonalisedNewsReccomendation/data --output-dir outputs --phase AB --run-gnn 0
```

If KG GNN is part of the final run:

```powershell
python training_packages/train_h100.py --mind-path /path/to/MIND-large --data-dir hyperpersonalisedNewsReccomendation/data --output-dir outputs --phase AB --run-gnn 1
```

## What not to upload

- `frontend/`
- `.next/`
- `node_modules/`
- `data/faiss_mind.index`
- `data/news_faiss.index`
- `data/qdrant_local_py312/`
- `data/hypernews.db`
- `data/ltr_features*.csv`
- runtime log files
- Docker files
- the whole `.git/` directory

## Outputs to bring back

Always:

- `bio_encoder.pt`
- `dqn_policy.pt`
- `logging_policy.pkl`
- `eval_results.json`
- `metrics.jsonl`

If KG is trained:

- `kg_gnn.pt`
- `kg_embeddings.npy`
