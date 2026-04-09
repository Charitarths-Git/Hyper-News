# HyperNews — Setup & Deployment

## Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10+ | FastAPI backend |
| Node.js | 18+ | Next.js frontend |
| npm / pnpm | latest | Frontend package manager |
| SQLite | bundled | Default DB (no install needed) |
| Git LFS | optional | Large model files |

Optional (for production features):
- PostgreSQL 14+
- Redis 7+
- Qdrant (vector DB)
- OpenSearch 2.x

---

## Directory Structure

```
hyperpersonalisedNewsReccomendation/
├── backend/
│   ├── app.py                  # FastAPI entry point
│   ├── ranker.py               # Scoring & bandit
│   ├── candidates.py           # build_candidate_pool
│   ├── feedback.py             # Reward + interest update
│   ├── cold_start.py           # Bio-based new user recs
│   ├── search.py               # Hybrid search pipeline
│   ├── knowledge_graph.py      # NetworkX KG builder
│   ├── bio_encoder.py          # BioEncoder model
│   ├── models/
│   │   ├── bio_encoder.pt      # Trained BioEncoder weights
│   │   └── ltr_model.txt       # LightGBM LambdaMART
│   ├── data/
│   │   ├── articles.parquet    # 51K article rows
│   │   └── article_embeddings.npy  # 51K × 384 float32 (78MB)
│   └── requirements.txt
├── frontend/
│   ├── app/                    # Next.js App Router
│   ├── components/             # React components
│   ├── public/
│   │   └── faceapi-models/     # face-api.js model weights
│   ├── .env.local              # Environment variables
│   └── package.json
└── docs/                       # This documentation
```

---

## Backend Setup

### 1. Create virtual environment

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Core dependencies:
```
fastapi==0.111.0
uvicorn[standard]==0.29.0
sentence-transformers==2.7.0
faiss-cpu==1.8.0
lightgbm==4.3.0
networkx==3.3
pandas==2.2.0
numpy==1.26.4
torch==2.3.0
rank-bm25==0.2.2
redis==5.0.4           # optional
psycopg2-binary==2.9.9 # optional, for PostgreSQL
qdrant-client==1.9.1   # optional
langchain-groq==0.1.3  # optional, for LLM explanations
```

### 3. Environment variables

Create `backend/.env`:

```env
# Required
DATABASE_URL=sqlite:///./hypernews.db

# Optional: PostgreSQL
# DATABASE_URL=postgresql://user:pass@localhost:5432/hypernews

# Optional: Redis
REDIS_URL=redis://localhost:6379/0

# Optional: Groq (LLM explanations)
GROQ_API_KEY=your_groq_api_key_here

# Optional: Qdrant
QDRANT_URL=http://localhost:6333

# Optional: OpenSearch
OPENSEARCH_URL=http://localhost:9200
```

### 4. Data files

Place these in `backend/data/`:
- `articles.parquet` — article metadata (title, abstract, category, publish_date, etc.)
- `article_embeddings.npy` — precomputed 384-d embeddings (must match article order in parquet)

Generate embeddings if not provided:
```bash
python scripts/build_embeddings.py --input data/articles.parquet --output data/article_embeddings.npy
```

### 5. Start backend

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

On startup, the backend:
1. Loads `articles.parquet` into memory (pandas DataFrame)
2. Loads `article_embeddings.npy` into RAM as numpy array
3. Builds FAISS IndexFlatIP index (~0.5s for 51K articles)
4. Builds NetworkX knowledge graph from article entities (~2-5s)
5. Builds BM25 index (~2s)
6. Creates SQLite tables if not present
7. Loads `bio_encoder.pt` and `ltr_model.txt`

Startup time: ~10–30 seconds depending on data size.

---

## Frontend Setup

### 1. Install dependencies

```bash
cd frontend
npm install
```

Key packages:
```json
{
  "@vladmandic/face-api": "^1.7.14",
  "next": "15.x",
  "react": "18.x",
  "next-auth": "5.x",
  "recharts": "2.x",
  "d3": "7.x",
  "tailwindcss": "3.x"
}
```

### 2. Environment variables

Create `frontend/.env.local`:

```env
# NextAuth
AUTH_SECRET=your_random_secret_here_min_32_chars
NEXTAUTH_URL=http://localhost:3000

# Backend URL
NEXT_PUBLIC_API_URL=http://localhost:8000

# Optional: Gemini API (not used in current affect implementation)
GEMINI_API_KEY=your_gemini_key_here
```

Generate `AUTH_SECRET`:
```bash
openssl rand -base64 32
```

### 3. Face-api.js model weights

The face-api.js models must be served as static files:

```bash
# Copy from node_modules to public/
cp -r node_modules/@vladmandic/face-api/model/* public/faceapi-models/
```

Required files in `public/faceapi-models/`:
```
tiny_face_detector_model-weights_manifest.json
tiny_face_detector_model-shard1
face_expression_model-weights_manifest.json
face_expression_model-shard1
```

### 4. Start frontend

```bash
npm run dev
```

Frontend starts on `http://localhost:3000`.

---

## Database Initialisation

SQLite tables are auto-created on first backend start. For PostgreSQL, run migrations manually:

```bash
python scripts/init_db.py --url postgresql://user:pass@localhost:5432/hypernews
```

Or use the raw SQL from `docs/05_DATABASE_AND_STATE.md`.

---

## Production Deployment

### Backend (FastAPI)

```bash
# Gunicorn with Uvicorn workers
gunicorn app:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 60 \
  --preload
```

`--preload` loads FAISS + embeddings once in the master process, then forks workers that share the memory (copy-on-write). Without `--preload`, each worker loads 78MB embeddings independently.

Nginx reverse proxy:
```nginx
upstream hypernews_api {
    server 127.0.0.1:8000;
}

server {
    listen 443 ssl;
    server_name api.hypernews.example.com;

    location / {
        proxy_pass http://hypernews_api;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Frontend (Next.js)

```bash
npm run build
npm start   # Production server on port 3000
```

Or deploy to Vercel:
```bash
vercel --prod
```

Set environment variables in Vercel dashboard matching `.env.local`.

---

## Redis Setup (Optional but Recommended)

```bash
# Docker
docker run -d --name redis -p 6379:6379 redis:7-alpine

# Or direct install (Ubuntu)
sudo apt install redis-server
sudo systemctl start redis
```

The backend detects Redis availability at startup:
```python
try:
    redis_client = redis.Redis.from_url(REDIS_URL)
    redis_client.ping()
    REDIS_AVAILABLE = True
except:
    REDIS_AVAILABLE = False
    # Falls back to TTLCache in-process dict
```

Without Redis, session state is lost on server restart and doesn't persist across multiple backend workers.

---

## API Endpoints Summary

### Public (no auth)

| Method | Path | Description |
|---|---|---|
| POST | `/public/recommend` | Feed recommendations for guests |
| POST | `/public/feedback` | Feedback for guests |
| GET | `/health` | Health check |

### Authenticated (require `user_id`)

| Method | Path | Description |
|---|---|---|
| POST | `/recommend` | Feed or search recommendations |
| POST | `/feedback` | User interaction feedback |
| GET | `/user/{user_id}/profile` | Get user profile |
| POST | `/user/{user_id}/profile` | Update profile (bio, consent) |
| GET | `/user/{user_id}/interests` | Get interest weights |
| GET | `/user/{user_id}/history` | Get reading history |

### Next.js API Routes (proxies with auth injection)

| Method | Path | Proxies to |
|---|---|---|
| POST | `/api/recommend` | `POST /recommend` |
| POST | `/api/feedback` | `POST /feedback` |
| POST | `/api/public/recommend` | `POST /public/recommend` |
| GET/POST | `/api/me/profile` | `/user/{id}/profile` |
| ANY | `/api/auth/[...nextauth]` | NextAuth handlers |

---

## Monitoring & Logs

Backend logs to stdout in JSON format. Key log events:

```
startup:completed     → models loaded, ready
recommend:served      → {user_id, mode, n_candidates, latency_ms}
feedback:processed    → {user_id, article_id, action, reward}
bandit:updated        → {user_id, context_norm}
cold_start:served     → {user_id, total_interactions}
redis:miss            → {user_id} — fell back to DB
```

Track these metrics in production:
- `recommend:latency_ms` — target P99 < 500ms
- `cold_start:served` ratio — should decrease as users mature
- `redis:miss` rate — indicates Redis health
- `feedback:reward` distribution — model performance indicator
