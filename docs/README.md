# HyperNews Documentation

Complete technical documentation for the HyperNews hyper-personalised news recommendation system.

## Documents

| File | Contents |
|---|---|
| [01_ARCHITECTURE.md](01_ARCHITECTURE.md) | System overview, component diagram, request lifecycle, storage tiers, frontend structure |
| [02_MODELS_AND_ML.md](02_MODELS_AND_ML.md) | All 9 ML models: SentenceTransformer, FAISS, LinUCB bandit, LightGBM LTR, Knowledge Graph, BioEncoder, DQN, face-api, MMR |
| [03_RECOMMENDATION_PIPELINE.md](03_RECOMMENDATION_PIPELINE.md) | Step-by-step code walkthrough of a single feed request, feedback loop |
| [04_AFFECT_AND_MOOD.md](04_AFFECT_AND_MOOD.md) | Webcam mood detection, face-api.js pipeline, GDPR consent, mood influence on ranking |
| [05_DATABASE_AND_STATE.md](05_DATABASE_AND_STATE.md) | SQL schemas, UserProfile dataclass, interest EMA algorithm, Redis schema, auth |
| [06_SEARCH_PIPELINE.md](06_SEARCH_PIPELINE.md) | Hybrid search: FAISS + BM25 + KG + RRF fusion, cross-encoder reranking |
| [07_COLD_START_AND_ONBOARDING.md](07_COLD_START_AND_ONBOARDING.md) | New user onboarding, BioEncoder, cold-start scoring, warm-up ramp |
| [08_SETUP_AND_DEPLOYMENT.md](08_SETUP_AND_DEPLOYMENT.md) | Installation, environment variables, face-api model setup, production deployment |

## Quick Start

```bash
# Backend
cd backend && pip install -r requirements.txt
uvicorn app:app --port 8000

# Frontend
cd frontend && npm install
npm run dev
```

See [08_SETUP_AND_DEPLOYMENT.md](08_SETUP_AND_DEPLOYMENT.md) for full setup instructions.

## System at a Glance

```
Browser (Next.js 15)
  └─ RecommendationSurface
       ├─ AffectSensor (face-api.js, TF.js WebGL, on-device only)
       ├─ ContextBar (mood pills, explore/focus slider)
       └─ NewsCard × 10

FastAPI Backend
  ├─ /recommend  → FAISS + KG + Memory → LTR + LinUCB → MMR → Groq explanation
  ├─ /feedback   → reward → EMA interests → bandit update → KG propagation
  └─ /search     → FAISS + BM25 + KG → RRF fusion → optional cross-encoder

Storage
  ├─ Hot:  Redis sorted sets (TTL 1hr) + in-process USERS dict
  ├─ Warm: SQLite/PostgreSQL (7 tables) + NumPy embeddings in RAM
  └─ Cold: FAISS index (51K articles) + NetworkX KG (18K nodes)
```
