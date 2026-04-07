# Local Runtime Data

This folder is kept in the repo so the expected paths exist, but the generated
runtime assets inside it are intentionally not committed to GitHub.

Common local files:

- `articles.parquet`
- `article_embeddings.npy`
- `faiss_mind.index`
- `hypernews.db`

Rebuild the core retrieval assets locally with:

```powershell
python backend/generate_data.py
```

Notes:

- If raw MIND files are available under `data/mind*`, the generator prefers
  them.
- If not, the generator falls back to a small synthetic demo dataset.
- Database files, cached indices, exports, and embeddings stay local by design.
