# HyperNews Run Guide

This guide covers the finished vNext setup:

- Auth.js-backed web login
- FastAPI recommendation backend
- hybrid search with lexical + memory + KG fallback
- trained LTR model loading with runtime fallback weights
- optional production services: PostgreSQL, Redis, Qdrant, OpenSearch

## 1. What Is In The Repo Now

### Web app
- Auth.js credentials login is implemented in `frontend/auth.ts`.
- The browser no longer picks a user ID from `sessionStorage`.
- Frontend requests go through authenticated Next.js API routes under `frontend/app/api/...`.

### Backend
- Auth endpoints:
  - `POST /auth/register`
  - `POST /auth/validate`
- Recommendation endpoints:
  - `POST /recommend`
  - `POST /feedback`
  - `GET /search/suggest`
  - `GET /me/profile`
  - `GET /me/history`
  - `GET /me/searches`
- LTR feature rows are logged into `ranking_feature_events`.

### Model training
- Feature export script: `backend/export_ltr_features.py`
- Trainer: `backend/train_ltr.py`
- Generated artifacts stay local and are ignored by GitHub:
  - `models/ltr_model.txt`
  - `models/ltr_model.txt.weights.json`

## 2. Where User Data Is Stored

### Current live local default
- Durable profile data: `data/hypernews.db`
- Tables:
  - `users`
  - `auth_users`
  - `reading_history_vectors`
  - `feedback_events`
  - `recommendation_events`
  - `search_events`
  - `user_recent_state`
  - `user_profile_snapshots`
  - `ranking_feature_events`

### Short-term state
- Redis when `REDIS_URL` is available
- in-memory fallback when Redis is not available
- managed in `backend/user_profile.py`

### Recent vs old clicks
- Recent `25` clicks: `user_recent_state.recent_clicks_json`
- Recent `25` negative actions: `user_recent_state.recent_negative_actions_json`
- Recent `10` queries: `user_recent_state.recent_queries_json`
- Old click history: `users.reading_history` and `reading_history_vectors`

## 3. Recommended Local Run

This is the fastest stable way to run on this machine without Docker.

### Backend
From repo root:

```powershell
Copy-Item .env.example .env
.\scripts\start_backend.ps1
```

If you want a lighter startup:

```powershell
.\scripts\start_backend.ps1 -DisableGraph -MaxArticles 5000
```

If you want dense query embedding and reranking turned on:

```powershell
.\scripts\start_backend.ps1 -UseDenseModels
```

Important:
- `-UseDenseModels` enables the SentenceTransformer query encoder and the cross-encoder reranker.
- first load can be slow because models may need to initialize or download.
- the default local script keeps those heavy models off so the app boots fast.

### Frontend
In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:3000
```

Create an account on `/login`, then sign in.

## 4. Production-Shaped Run

If Docker is installed on the machine, the repo already includes `docker-compose.yml` for:

- PostgreSQL
- Redis
- Qdrant
- OpenSearch
- backend
- frontend

Run:

```powershell
docker compose up --build
```

Before that, make sure `.env` contains at least:

```env
DATABASE_URL=postgresql://hypernews:hypernews@localhost:5432/hypernews
REDIS_URL=redis://localhost:6379
QDRANT_URL=http://localhost:6333
OPENSEARCH_URL=http://localhost:9200
HYPERNEWS_ENABLE_QDRANT=1
HYPERNEWS_ENABLE_OPENSEARCH=1
HYPERNEWS_ENABLE_SENTENCE_TRANSFORMER=1
HYPERNEWS_ENABLE_RERANKER=1
AUTH_SECRET=change-me-to-a-long-random-string
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
HYPERNEWS_BACKEND_URL=http://127.0.0.1:8000
```

## 4A. Recommended Free Cloud Deployment

For the current cloud-first path, use:

- `Vercel Hobby` for `frontend/`
- `Render Free Web Service` for the backend using the repo `Dockerfile`
- `Supabase Free Postgres` for durable storage

The repo now includes a dedicated guide:

- [DEPLOY_FREE_CLOUD.md](DEPLOY_FREE_CLOUD.md)

Important:
- `docker-compose.yml` is still useful for local/full-stack testing.
- It is not the recommended deployment artifact for the free cloud split.
- The backend Docker image uses `requirements.runtime.txt`, the root `Dockerfile`, and the slim runtime asset set prepared by `scripts/prepare_cloud_assets.py`.

## 5. Train The LTR Model

### Fast path

```powershell
.\scripts\train_ltr_model.ps1
```

This does two things:
- exports feature rows with `backend/export_ltr_features.py`
- trains the model with `backend/train_ltr.py`

### Manual path

Export features:

```powershell
python backend/export_ltr_features.py --source auto --output-csv data/ltr_features.auto.csv --limit-impressions 2000
```

Train with Python 3.12 if available:

```powershell
C:\Users\rinak\AppData\Local\Programs\Python\Python312\python.exe backend/train_ltr.py --features-csv data/ltr_features.auto.csv --output-model models/ltr_model.txt
```

Why Python 3.12:
- LightGBM wheels are more reliable there on Windows.
- the runtime app can still use `models/ltr_model.txt.weights.json` even if `lightgbm` is not installed in the main interpreter.

## 6. Migrate SQLite To PostgreSQL

If you move from local SQLite to PostgreSQL:

```powershell
python backend/migrate_sqlite_to_postgres.py --sqlite-path .\data\hypernews.db
```

Make sure `DATABASE_URL` points to PostgreSQL first.

## 7. Health Checks

### Backend health

```powershell
curl http://127.0.0.1:8000/health
```

### Ranking health

```powershell
curl http://127.0.0.1:8000/admin/ranking-health
```

### Offline evaluator

```powershell
python backend/evaluate_mind.py --limit-impressions 100
```

## 8. What I Verified In This Repo

The following were run successfully in this workspace:

- backend compile smoke
- frontend `npx tsc --noEmit`
- frontend `npm run build`
- backend auth register/login smoke
- backend feed recommendation smoke
- backend search recommendation smoke with lexical fallback
- `backend/export_ltr_features.py` on MIND data
- `backend/evaluate_mind.py --limit-impressions 100`

### Notes on current machine constraints
- Docker is not installed here, so I could not bring up the full compose stack live in this session.
- The local default runtime therefore used:
  - SQLite for durable storage
  - Redis fallback in memory
  - FAISS/local retrieval fallback
  - trained LTR fallback weights instead of direct `lightgbm` runtime loading

## 9. Useful Commands

Reset one user profile:

```powershell
curl -X POST http://127.0.0.1:8000/reset/<user_id>
```

Export feature rows from live labeled events when enough traffic exists:

```powershell
python backend/export_ltr_features.py --source db --output-csv data/ltr_features.db.csv
```

Export feature rows from MIND:

```powershell
python backend/export_ltr_features.py --source mind --output-csv data/ltr_features.mind.csv --limit-impressions 2000
```
