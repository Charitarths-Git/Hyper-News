# HyperNews — System Architecture

## Overview

HyperNews is a hyper-personalised news recommendation system that combines deep reinforcement learning, knowledge graphs, learning-to-rank, and multimodal signals (including real-time facial affect detection) into a single unified pipeline.

```
┌─────────────────────────────────────────────────────────┐
│                     Browser (Next.js)                    │
│                                                          │
│  ┌──────────┐  ┌───────────────┐  ┌──────────────────┐  │
│  │ Feed Page │  │  Search Page  │  │  Profile/Onboard │  │
│  └────┬─────┘  └──────┬────────┘  └────────┬─────────┘  │
│       │               │                    │             │
│  ┌────▼───────────────▼────────────────────▼──────────┐  │
│  │            RecommendationSurface                    │  │
│  │  - mood state  - explore/focus slider               │  │
│  │  - AffectSensor (face-api, TinyFaceDetector + expr) │  │
│  └────────────────────┬────────────────────────────────┘  │
└───────────────────────┼────────────────────────────────┘
                        │ HTTP/JSON
                        ▼
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Backend (port 8000)             │
│                                                          │
│  POST /recommend                POST /feedback            │
│       │                               │                  │
│  ┌────▼──────────────┐         ┌──────▼─────────────┐   │
│  │  Candidate Pool   │         │  Reward Computation │   │
│  │  - FAISS (top 60) │         │  - dwell multiplier │   │
│  │  - KG BFS (top 40)│         │  - interest update  │   │
│  │  - Memory (top 20)│         │  - bandit update    │   │
│  └────┬──────────────┘         └──────┬─────────────┘   │
│       │                               │                  │
│  ┌────▼──────────────┐                │                  │
│  │   Ranker           │◄──────────────┘                  │
│  │  - LTR (LightGBM) │                                   │
│  │  - LinUCB Bandit  │                                   │
│  │  - MMR Diversity  │                                   │
│  └────┬──────────────┘                                   │
│       │                                                  │
│  ┌────▼──────────────┐                                   │
│  │  Explanation Gen  │                                   │
│  │  (Groq or static) │                                   │
│  └───────────────────┘                                   │
│                                                          │
│  Storage Layer:                                          │
│  SQLite/PostgreSQL │ Redis │ FAISS │ NetworkX KG         │
└─────────────────────────────────────────────────────────┘
```

---

## Hard Architectural Rule: Feed ≠ Search

The single most important rule in the system. Feed and Search are completely separate execution paths. No candidate, score, or context leaks between them.

| Concern | Feed | Search |
|---|---|---|
| Candidate source | FAISS + KG + Memory | Hybrid (semantic + BM25 + KG) |
| Ranking driver | LinUCB Bandit + LTR | LTR only (lighter weights) |
| Query field | Forced to `None` | Required |
| Context stored | No | Yes (10-min TTL, 384-d embedding) |
| Session pollution | Not possible | Isolated |

This prevents a user's search for "cancer treatment" from injecting health articles into their entertainment feed.

---

## Request Lifecycle

### Feed Request

```
POST /recommend { user_id, mood, surface="feed", n=10, explore_focus=55 }

1. Load session (Redis → PostgreSQL fallback)
2. Cold-start check (total_positive_interactions < 5?)
   YES → cold_start_recommendations() [bio + category + popularity]
   NO  → full pipeline:
         a. build_candidate_pool() → ~200 articles
         b. rank_articles() → 10 scored articles
         c. MMR diversity filter
3. Generate explanation
4. Log recommendation_event
5. Return JSON
```

### Search Request

```
POST /recommend { user_id, query="AI ethics", surface="search", n=10 }

1. Encode query → 384-d embedding
2. build_hybrid_candidates():
   - FAISS top-40 (semantic)
   - BM25 token index (lexical)
   - KG entity expansion
   - RRF fusion
   - Optional cross-encoder reranking
3. rank_search_articles() → 10 scored articles
4. Store search_context in session
5. Return JSON
```

### Feedback Request

```
POST /feedback { user_id, article_id, action, dwell_time }

1. Compute reward (dwell-adjusted)
2. Update user interests (EMA)
3. Update bandit (Sherman-Morrison)
4. Propagate to KG siblings
5. Save reading vector to DB
6. Log events
```

---

## Storage Architecture

```
Hot Path (milliseconds):
  Redis            → session: recent_clicks, skips, mood, topics
  In-process dict  → USERS[user_id]: UserProfile objects
  NumPy RAM        → EMBEDDINGS[51K × 384], FAISS index

Warm Path (tens of ms):
  PostgreSQL/SQLite → long-term interests, reading history
  NetworkX          → knowledge graph (18K nodes, RAM)
  LightGBM          → ltr_model.txt

Cold Path (seconds):
  Disk              → article_embeddings.npy (78 MB)
  Disk              → articles.parquet (51K rows)
  Optional:
    Qdrant          → user memory vectors (pgvector-style)
    OpenSearch      → full-text BM25 search
```

---

## Deployment Architecture

```
Dev:
  next dev --webpack --hostname 127.0.0.1  (port 3000)
  uvicorn app:app --port 8000

Prod (target):
  Next.js → Vercel / Nginx
  FastAPI → Gunicorn + Uvicorn workers
  PostgreSQL + Redis + Qdrant + OpenSearch
```

---

## Frontend Architecture

```
app/
  page.tsx               → Feed (RecommendationSurface surface="feed")
  search/page.tsx        → Search (RecommendationSurface surface="search")
  profile/page.tsx       → User profile + KG viz
  onboarding/page.tsx    → First-run bio collection
  dashboard/page.tsx     → Analytics
  api/
    recommend/           → Proxy to FastAPI /recommend (auth'd)
    feedback/            → Proxy to FastAPI /feedback
    public/recommend/    → Proxy to /public/recommend (guest)
    me/profile/          → Profile CRUD
    auth/                → NextAuth handlers

components/
  RecommendationSurface  → Core feed/search UI + state machine
  ContextBar             → Header: mood pills, camera toggle, search
  AffectSensor           → Face detection + expression → mood
  NewsCard               → Article card + feedback buttons
  ExplanationBanner      → AI reasoning display
  KnowledgeGraphPanel    → Force-directed KG viz
  InterestChart          → Category interest weights bar chart
```
