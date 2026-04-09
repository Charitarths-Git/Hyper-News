# HyperNews — Hyper-Personalised News Recommender

> 🌐 **Live Demo:** [https://hyper-news-zeta.vercel.app/](https://hyper-news-zeta.vercel.app/)

A production-grade news recommender that combines deep RL, knowledge graphs,
multimodal signals, and RAG — with a hard architectural separation between
the feed path and the search path.

---

## Search and Feed are completely separate — no intersection

This is the most important architectural rule in the system.

```
surface = "feed"                    surface = "search" + query
        │                                       │
        ▼                                       ▼
  req.query = None  (hard override)     build_hybrid_candidates()
  assert not search_stack_used           FAISS + BM25 + KG + RRF
  assert query_intent is None            cross-encoder reranker
        │                                       │
        ▼                                       ▼
  build_candidate_pool()              rank_search_articles()
  KG expansion + memory               search-specific LTR weights
        │                                       │
        ▼                                       ▼
  rank_articles()                 store search_context blob
  LinUCB bandit → DQN (Phase 4)  (384-d embedding, TTL 10min)
  LTR scorer                              │
  MMR diversity                           ▼
        │                       feed after search:
        ▼                       tiny cosine bonus (0.03×)
  10 articles returned           on already-selected candidates
                                 NO new candidates injected
```

Enforced at `backend/app.py` lines 952–955 and 1038–1086:
```python
if req.surface == "feed":
    req.query = None              # hard override — query cannot leak into feed

assert req.surface == "feed"
assert req.query is None
assert not search_stack_used
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Frontend  (Next.js, port 3000)                                  │
│  ├── Feed page    → POST /api/recommend { surface: "feed" }      │
│  ├── Search page  → POST /api/recommend { surface: "search" }    │
│  ├── Profile page → GET  /api/me/profile                         │
│  └── Feedback     → POST /api/feedback  { action, dwell_time }   │
└────────────────────────┬────────────────────────────────────────┘
                         │  HTTP (localhost)
┌────────────────────────▼────────────────────────────────────────┐
│  Backend  (FastAPI, port 8000)                                   │
│                                                                  │
│  POST /recommend ──► surface router                             │
│        │                                                         │
│        ├── feed  ──► KG expand + memory ──► LinUCB / DQN       │
│        │             rank_articles() + MMR                       │
│        │                                                         │
│        └── search ──► FAISS + BM25 + KG + RRF                  │
│                       cross-encoder rerank                       │
│                       rank_search_articles()                     │
│                                                                  │
│  POST /feedback ──► update interests + bandit + pgvector        │
└──────────────────────────────────────────────────────────────────┘
         │                    │                    │
    SQLite/Postgres       Redis               FAISS / Qdrant
    (events, users)    (hot session)        (vector search)
```

---

## Data Flow End-to-End

### 1. Startup (once, ~15 seconds)
```
articles.parquet        → pandas DataFrame (51k rows, in RAM)
article_embeddings.npy  → numpy array (51k × 384, ~78 MB, in RAM)
faiss_mind.index        → FAISS index (in RAM)
knowledge_graph.pkl     → NetworkX graph (~18k nodes, in RAM)
bandit_model.pkl        → LinUCB bandit (in RAM)
ltr_model.txt           → LightGBM LambdaMART (in RAM)
```

### 2. Feed Request
```
POST /recommend { user_id, surface: "feed", mood, n: 10 }
        │
        ├── Load session from Redis → UserProfile
        │   { interests, reading_history, recent_clicks, mood }
        │
        ├── Cold-start? (< 5 positive interactions)
        │   YES → popular + diverse + bio_affinity bonus (Phase 2)
        │   NO  → warm path below
        │
        ├── build_candidate_pool() → ~200 candidates
        │   ├── FAISS: top-60 cosine similar to user_profile_vector
        │   ├── KG:    BFS from user interests + recent reads
        │   └── Memory: pgvector similarity to past reads
        │
        ├── score each candidate
        │   semantic + interest + memory + kg + entity +
        │   subcategory + popularity + context + negative_penalty
        │   → LightGBM LTR score
        │   → LinUCB bandit score (→ DQN in Phase 4)
        │
        ├── MMR diversity filter (λ=0.82)
        │   guarantees ≥ 3 categories in slate
        │
        └── Return 10 articles + candidate_source_distribution
```

### 3. Search Request
```
POST /recommend { user_id, surface: "search", query: "singer", n: 10 }
        │
        ├── build_hybrid_candidates()
        │   ├── FAISS dense: top-40 by cosine(query_emb, article_emb)
        │   ├── BM25 lexical: token_to_docids lookup
        │   ├── KG expansion: entities from query → graph neighbors
        │   └── RRF fusion (Reciprocal Rank Fusion)
        │
        ├── Optional cross-encoder reranking
        │
        ├── rank_search_articles() with search-specific weights
        │
        ├── Store search_context blob in session
        │   { embedding: 384-d, top_entities: [3], TTL: 10min }
        │
        └── Return 10 results
```

### 4. Feedback
```
POST /feedback { user_id, article_id, action, dwell_time }
        │
        ├── reward = REWARD_MAP[action] × dwell_multiplier
        │   click=0.5, read_full=1.0, save=2.0
        │   skip=-0.3, not_interested=-0.8
        │
        ├── update user.interests[category] += reward × 0.25
        ├── EMA decay all interests (prevents lock-in)
        ├── LinUCB.update(article_id, context_391d, reward)
        ├── propagate 12% reward to KG sibling articles
        ├── push embedding to pgvector (long-term memory)
        └── persist session to Redis
```

---

## Embeddings — four separate spaces

| Space | Dim | What it encodes | Where used |
|---|---|---|---|
| Article embedding | 384-d | title + abstract text | FAISS search, scoring, user profile |
| User profile vector | 384-d | weighted mean of clicked articles | candidate retrieval seed |
| Context vector (bandit) | 391-d | article_emb + 7 session scalars | LinUCB input (current) |
| Fused state (DQN) | 206-d | history(128) + bio(64) + affect(2) + ctx(12) | DQN input (Phase 4) |

All article embeddings use `all-MiniLM-L6-v2` (SentenceTransformer).
All vectors are L2-normalised before use.

---

## Implementation Phases

### Phase 1 — Retrieval Infrastructure ✅ Complete
- Precomputed `token_to_docids` and `entity_to_docids` at startup
- FAISS IVFFlat replaces brute-force scan
- Hard `assert` guards enforce feed/search separation at code level
- `candidate_source_distribution` logged on every `/recommend`
- Frontend: `viewMode` state decoupled from query text

**Result:** Search fallback `11–21s → <500ms`. Feed unchanged at ~300ms.

---

### Phase 2 — Bio Cold-Start
**Goal:** new users get relevant articles on first visit, not just popular noise.

**What to build:**
- `frontend/components/OnboardingFlow.tsx` — 5–7 question form shown on first login
- `backend/models/bio_encoder.py` — MLP: signup fields → 64-d `bio_emb` ✅ created
- `backend/train_bio_encoder.py` — train on MIND user clusters ✅ created

**How it works:**
```python
# cold_start_recommendations() gains:
bio_affinity = cosine(article_emb, bio_emb)
score += 0.30 * bio_affinity * cold_start_decay(interaction_count)
cold_start_decay = max(0.1, 1 - interaction_count / 20)
# bonus fades as user warms up — no permanent distortion
```

**Run:**
```bash
python training_packages/train_bio_encoder.py \
    --mind-path /path/to/MIND \
    --data-dir  data/ \
    --output-dir outputs/
cp outputs/bio_encoder.pt models/
```

**Target:** cold-start nDCG@5 > 0.32

---

### Phase 3 — Face Affect Sensor (consent-gated)
**Goal:** mood updates automatically from camera, not just manual buttons.

**What to build:**
- `frontend/components/AffectSensor.tsx` — ONNX inference in browser
- Consent gate at onboarding + persistent opt-out toggle in settings

**Privacy rules (non-negotiable):**
- All face inference runs on-device via ONNX.js
- Only `{valence: float, arousal: float}` transmitted — never raw frames
- Face data = GDPR Article 9 biometric, requires explicit opt-in + DPIA

**How it works:**
```
camera frame → MTCNN detect → MobileNetV2 INT8 → valence/arousal
→ transmitted with each /recommend header
→ backend: affect_emb (2-d) enters fused state
→ if valence < -0.3: reduce heavy-news category weights by 0.2
```

**Weight:** `affect_emb` weight = 0.15 in fusion. Text/bio always dominate.

**Target:** nDCG@10 > 0.39

---

### Phase 4 — Deep DQN Policy
**Goal:** replace single-step LinUCB with a sequential policy that optimises long-horizon reward.

**What to build:**
- `backend/models/history_encoder.py` — 4-layer Transformer, 128-d output ✅ created
- `backend/models/dqn_policy.py` — Double DQN + dueling architecture ✅ created
- `backend/logging_policy.py` — propensity model for IPS correction ✅ created
- `training_packages/train_dqn_core.py` — offline training on MIND ✅ created

**State vector (206-d):**
```
history_emb   128-d  HistoryEncoder over last 50 clicks
bio_emb        64-d  BioEncoder (signup fields)
affect_emb      2-d  valence, arousal
context_feats  12-d  hour, weekday, session_len, skip_ratio...
```

**Training pipeline:**
```
Phase A (offline): MIND behaviors.tsv + IPS correction → DQN weights
Phase B (online):  update every 64 /feedback events in production
```

**Run:**
```bash
# Colab T4
python training_packages/train_colab_t4.py \
    --mind-path /content/MIND-small \
    --data-dir  data/ \
    --output-dir outputs/

# RTX 4060
python training_packages/train_rtx4060.py \
    --mind-path /path/to/MIND-large \
    --data-dir  data/ \
    --output-dir outputs/ \
    --phase A

cp outputs/dqn_policy.pt models/
```

**A/B deploy:** 5% DQN, 95% LinUCB. Ramp to 100% after 2 weeks stable.

**Target:** nDCG@10 > 0.42, feed p95 < 200ms

---

### Phase 5 — GNN Knowledge Graph
**Goal:** replace scalar `kg_score` with a 64-d structural embedding per KG node.

**What to build:**
- `backend/models/kg_gnn.py` — 2-layer GraphSAGE ✅ created
- `training_packages/train_kg_gnn.py` — link prediction training ✅ created

**How it integrates:**
```python
# Before (scalar):
kg_score = graph_bonus_map[article_id]    # 0.0–1.0

# After (vector):
kg_emb = kg_embeddings[article_idx]       # (64,)
item_features = concat(article_emb, kg_emb)  # (448,)
Q(state, item_448d) — DQN reasons about topical structure
```

**Run:**
```bash
python training_packages/train_kg_gnn.py \
    --graph-path graph/knowledge_graph.pkl \
    --data-dir   data/ \
    --output-dir outputs/

cp outputs/kg_embeddings.npy data/
cp outputs/kg_gnn.pt         models/
# Then retrain DQN with --kg-emb-path outputs/kg_embeddings.npy
```

**Target:** nDCG@10 > 0.44

---

### Phase 6 — Production Retrieval Stack
**Goal:** replace FAISS fallback with Qdrant + OpenSearch for production scale.

```bash
docker-compose up -d qdrant opensearch

# Index all articles
python scripts/index_qdrant.py
python scripts/index_opensearch.py

# Enable in .env
HYPERNEWS_ENABLE_QDRANT=1
HYPERNEWS_ENABLE_SENTENCE_TRANSFORMER=1
HYPERNEWS_ENABLE_RERANKER=1
```

**Target:** search p95 < 200ms, feed p95 < 150ms, 100 concurrent users

---

## Target Metrics

| Phase | AUC | nDCG@10 | Search p95 | Feed p95 |
|---|---|---|---|---|
| Baseline (now) | 0.574 | 0.373 | ~2–5s | ~300ms |
| Phase 1 done | 0.574 | 0.373 | < 500ms | ~300ms |
| Phase 2 +Bio | 0.580 | > 0.38 | < 500ms | ~300ms |
| Phase 3 +Affect | 0.582 | > 0.39 | < 500ms | ~300ms |
| Phase 4 +DQN | 0.600 | > 0.42 | < 500ms | < 200ms |
| Phase 5 +GNN | 0.615 | > 0.44 | < 500ms | < 200ms |
| Phase 6 +Prod | 0.615 | > 0.44 | < 200ms | < 150ms |

---

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- (optional) Redis, PostgreSQL, Qdrant, OpenSearch via Docker

### 1. Install backend
```bash
cd hyperpersonalisedNewsReccomendation
pip install -r requirements.txt
python backend/generate_data.py
```

### 2. Start backend
```bash
# Windows PowerShell
.\scripts\start_backend_local.ps1

# or directly
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Start frontend
```bash
cd frontend
npm install
npm run dev
# open http://localhost:3000
```

### 4. Verify
```bash
curl http://localhost:8000/health
```

---

## Free Cloud Deployment

The repo now includes a free-cloud deployment path with:

- `Vercel Hobby` for the frontend
- `Render Free Web Service` for the backend using the root `Dockerfile`
- `Supabase Free Postgres` for durable storage

Use [DEPLOY_FREE_CLOUD.md](DEPLOY_FREE_CLOUD.md) for the step-by-step flow.

`docker-compose.yml` remains a local/full-stack reference and is not the recommended cloud deployment artifact for the free tier setup.

---

## Training Packages (ready-made zips)

| File | GPU | VRAM | Notes |
|---|---|---|---|
| `training_HyperNews_ColabT4.zip` | Colab T4 | 16 GB | Free Colab GPU |
| `training_HyperNews_RTX4060.zip` | RTX 4060 | 8 GB | Full 4-layer Transformer |
| `training_HyperNews_RTX3050.zip` | RTX 3050/3060 | 4–6 GB | 2-layer Transformer, smaller batch |
| `training_HyperNews_H100_LightningAI.zip` | Lightning AI H100 | 80 GB | GNN + DQN together |

---

## Environment Variables

| Variable | Default | Effect |
|---|---|---|
| `DATABASE_URL` | sqlite:///data/hypernews.db | DB connection |
| `REDIS_URL` | redis://localhost:6379 | Session cache |
| `GROQ_API_KEY` | (unset) | LLM explanations on cards |
| `QDRANT_URL` | http://localhost:6333 | Vector DB |
| `OPENSEARCH_URL` | http://localhost:9200 | BM25 search |
| `HYPERNEWS_ENABLE_GRAPH` | 1 | Build KG on startup |
| `HYPERNEWS_ENABLE_SENTENCE_TRANSFORMER` | 0 | Dense query encoding for search |
| `HYPERNEWS_ENABLE_RERANKER` | 0 | Cross-encoder reranking |
| `HYPERNEWS_DEBUG_SCORE_BREAKDOWN` | 0 | Show score breakdown on cards |

---

## Project Structure

Runtime data, embeddings, database files, trained weights, and generated graph
artifacts are intentionally kept out of GitHub. The placeholder `README.md`
files under `data/`, `models/`, `graph/`, and `artifacts/` explain how to
rebuild those assets locally.

```
hyperpersonalisedNewsReccomendation/
├── backend/
│   ├── app.py                 # FastAPI app, all endpoints, global state
│   ├── ranker.py              # scoring, MMR, candidate building
│   ├── bandit.py              # LinUCB contextual bandit (current RL)
│   ├── ltr.py                 # LightGBM LambdaMART wrapper
│   ├── graph.py               # KG builder (NetworkX)
│   ├── hybrid_search.py       # FAISS + BM25 + RRF search pipeline
│   ├── rag_pipeline.py        # FAISS index + LLM explanations
│   ├── user_profile.py        # session state (Redis + SQLite)
│   ├── db.py                  # database abstraction layer
│   ├── train_ltr.py           # LambdaMART training script
│   ├── logging_policy.py      # IPS propensity model (Phase 4)
│   └── models/
│       ├── history_encoder.py # Transformer history encoder (Phase 4)
│       ├── dqn_policy.py      # Double DQN + dueling arch (Phase 4)
│       ├── bio_encoder.py     # MLP over signup fields (Phase 2)
│       └── kg_gnn.py          # GraphSAGE KG embeddings (Phase 5)
├── frontend/
│   ├── app/
│   │   ├── page.tsx           # main feed
│   │   ├── search/page.tsx    # search
│   │   └── profile/page.tsx   # user profile + KG visualisation
│   └── components/
│       ├── NewsCard.tsx
│       ├── RecommendationSurface.tsx
│       ├── InterestChart.tsx
│       └── KnowledgeGraphPanel.tsx
├── data/
│   └── README.md              # local parquet, embeddings, FAISS, and DB files
├── models/
│   └── README.md              # local trained model artifacts
├── graph/
│   └── README.md              # local knowledge graph cache files
├── artifacts/
│   └── README.md              # local bundles, smoke outputs, and zips
├── training_packages/         # GPU-tier training scripts + zips
└── docker-compose.yml
```

---

## Seven Rules That Must Never Break

1. `surface="feed"` never touches FAISS retrieval, RAG, or the search stack. DQN/bandit owns all candidate generation.
2. `surface="search"` never uses the DQN. LTR + RRF fusion owns search ranking.
3. Search context is a `0.03 ×` cosine bonus on already-selected feed candidates only. It never injects new candidates.
4. Face data never leaves the device. Only `{valence: float, arousal: float}` is transmitted.
5. Offline DQN training always uses IPS correction. Never naive log replay.
6. `bio_emb` is a cold-start ranking prior only. It never generates feed candidates.
7. DQN before PPO. DQN first (off-policy from logs). PPO only after DQN is stable and proven.

---

## GDPR Notes

- Face/biometric data = GDPR Article 9 Special Category. Requires DPIA + explicit opt-in.
- All face inference runs on-device (ONNX.js). Only `{valence, arousal}` crosses the wire.
- Behavioural logs are session-scoped with TTL purge policy.
- Users can delete all data via `DELETE /reset/{user_id}`.
