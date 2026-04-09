# HyperNews — Recommendation Pipeline

## End-to-End Feed Flow

This document traces a single `/recommend` request from browser to response.

---

## Step 0: Request Origin

The user opens the app. `RecommendationSurface` mounts, reads mood from `localStorage`, reads explore_focus, then fires:

```json
POST http://localhost:3000/api/recommend
{
  "mood": "stressed",
  "surface": "feed",
  "n": 10,
  "explore_focus": 55,
  "exclude_ids": [],
  "request_id": "req_1713456789_42"
}
```

The Next.js API route at `app/api/recommend/route.ts` injects `user_id` from the session cookie and proxies to FastAPI:

```json
POST http://localhost:8000/recommend
{
  "user_id": "user_abc123",
  "mood": "stressed",
  "surface": "feed",
  "n": 10,
  "explore_focus": 55,
  "exclude_ids": []
}
```

---

## Step 1: Session Load

```python
user = USERS.get(user_id)
if user is None:
    user = load_user_session(user_id)  # Redis → DB → new profile

user.mood = req.mood                   # Always apply latest mood from request
user.time_of_day = get_time_of_day()   # Derived from current hour
```

User state includes:
- `interests`: `{"sports": 0.42, "technology": 0.28, ...}`
- `recent_clicks`: last 25 article IDs
- `recent_skips`: last 25 skipped IDs
- `total_positive_interactions`: 47

---

## Step 2: Cold-Start Gate

```python
if user.total_positive_interactions < COLD_START_THRESHOLD (5):
    return cold_start_recommendations(user, DF, EMBEDDINGS, n=10)
```

**Cold-start path** (new user):
1. Score each article by `bio_category_affinity` (from top_categories, interest_text, occupation)
2. Apply popularity boost
3. Apply location match bonus
4. Return top-10, no bandit, no LTR

Since this user has 47 interactions, we continue to the full pipeline.

---

## Step 3: Build Candidate Pool (~200 articles)

```python
candidates = build_candidate_pool(user, DF, EMBEDDINGS, FAISS_INDEX, KG_GRAPH,
                                   max_candidates=200,
                                   memory_records=long_term_memory,
                                   excluded_ids=req.exclude_ids)
```

### 3a. User Profile Vector

```python
recent_indices = [news_id_to_idx[id] for id in user.recent_clicks[-15:] if id in news_id_to_idx]
profile_vector = np.mean(EMBEDDINGS[recent_indices], axis=0)
profile_vector /= np.linalg.norm(profile_vector)
# shape: (384,)
```

### 3b. FAISS Top-60

```python
scores, indices = FAISS_INDEX.search(profile_vector.reshape(1, -1), k=60)
faiss_candidates = [DF.iloc[idx].news_id for idx in indices[0]]
# Tagged with candidate_source="faiss"
```

### 3c. KG BFS — Top-40

```python
related_ids = set()
for article_id in user.recent_clicks[-5:]:
    neighbors = get_related_articles(article_id, KG_GRAPH, max_distance=2, limit=20)
    related_ids.update(neighbors)
kg_candidates = list(related_ids - set(user.reading_history))[:40]
# Tagged with candidate_source="kg"
```

### 3d. Memory Pool — Top-20

```python
# pgvector similarity search against past reads in PostgreSQL
# Falls back to FAISS if pgvector unavailable
memory_records = build_long_term_memory_signal(user, DF, EMBEDDINGS, ...)
memory_candidates = [r.news_id for r in memory_records[:20]]
# Tagged with candidate_source="memory"
```

### 3e. Deduplication & Filtering

```python
all_candidates = faiss_candidates + kg_candidates + memory_candidates
# Remove: reading_history, recent_skips, recent_negative_actions, exclude_ids
filtered = [c for c in all_candidates if c not in excluded]
# Final pool: ~120–200 articles
```

---

## Step 4: Score Each Candidate

For each article in the pool:

### 4a. Context Score (Mood × Time)

```python
mood_mult = MOOD_WEIGHTS.get(user.mood, {}).get(article.category, 1.0)
time_mult = TIME_WEIGHTS.get(user.time_of_day, {}).get(article.category, 1.0)
context_score = mood_mult * time_mult
# For stressed user reading politics: 0.4 × 1.0 = 0.4
# For stressed user reading entertainment: 1.5 × 1.3 (evening) = 1.95
```

### 4b. Semantic Score

```python
semantic_score = float(profile_vector @ EMBEDDINGS[article_idx])
# Cosine similarity ∈ [-1, 1], typically [0, 0.8]
```

### 4c. Entity & KG Score

```python
article_entities = get_article_entities(article_id, KG_GRAPH)
entity_overlap = len(set(article_entities) & set(user.recent_entities))
entity_score = min(entity_overlap / 3.0, 1.0)

kg_score = kg_bonus_map.get(article_id, 0.0)
# 1.0 for direct neighbor, 0.5 for 2-hop, 0 for not in KG path
```

### 4d. Negative Penalty

```python
skip_entity_penalty = sum(
    1 for entity in article_entities if entity in user_skipped_entities
) * SKIP_ENTITY_PENALTY_WEIGHT (0.05)

skip_category_penalty = (
    SKIP_CATEGORY_PENALTY_WEIGHT (0.12)
    if article.category in user_skipped_categories
    else 0
)
negative_score = skip_entity_penalty + skip_category_penalty
```

### 4e. Build Feature Dict

```python
features = {
    "semantic_score":    semantic_score,        # 0.63
    "retrieval_score":   0.9 if faiss else 0.7, # source weight
    "lexical_score":     token_overlap,          # 0.15
    "memory_score":      memory_bonus_map.get(article_id, 0),
    "kg_score":          kg_score,               # 0.5
    "entity_score":      entity_score,           # 0.33
    "subcategory_score": subcategory_match,      # 0.0 or 1.0
    "popularity_score":  article.popularity,     # 0.72
    "context_score":     context_score,          # 1.95
    "negative_score":    negative_score,         # 0.0
}
```

### 4f. LTR Score

```python
ltr_score = ltr_model.predict(features)
# LightGBM LambdaMART output ∈ [0, 1]
# OR deterministic weighted sum if no trained model
```

### 4g. Build Bandit Context Vector (391-d)

```python
article_emb = EMBEDDINGS[article_idx]                    # 384-d

mood_map = {"neutral":0, "happy":1, "curious":2, "stressed":3, "tired":4}
mood_val = mood_map[user.mood] / 4.0                    # ∈ [0, 1]

time_map = {"morning":0, "afternoon":1, "evening":2, "night":3}
time_val = time_map[user.time_of_day] / 3.0             # ∈ [0, 1]

category_interest = user.interests.get(article.category, 0)
category_total = sum(max(v,0) for v in user.interests.values()) + 1e-9
category_norm = category_interest / category_total

context = np.concatenate([
    article_emb,                                         # 384-d
    [mood_val, time_val, click_count_norm,
     skip_ratio, category_norm, kg_score]                # 7-d
])  # shape: (391,)
```

### 4h. Bandit Score

```python
theta = bandit.A_inv @ bandit.b
mean_reward = np.clip(theta @ context, 0, 1)
uncertainty = np.sqrt(context @ bandit.A_inv @ context)

if random() < 0.02:  # epsilon-greedy
    base = np.clip(np.random.normal(mean_reward, uncertainty), 0, 1)
else:
    base = mean_reward + 0.15 * uncertainty

# Article and category priors
article_prior = 0.12 * article_mean + 0.05 / sqrt(article_count)
category_prior = 0.08 * category_mean + 0.03 / sqrt(category_count)

bandit_score = np.clip(base + article_prior + category_prior, 0, 1.5)
```

### 4i. Final Score

```python
explore = req.explore_focus / 100.0  # 0.55
final_score = ltr_score * explore + bandit_score * (1 - explore)
# = ltr_score * 0.55 + bandit_score * 0.45
```

---

## Step 5: MMR Diversity Filter

```python
selected = []
remaining = sorted(candidates, key=lambda c: c.final_score, reverse=True)

while len(selected) < n and remaining:
    best = None
    best_mmr = -inf

    for candidate in remaining:
        if selected:
            max_sim = max(
                cosine(EMBEDDINGS[candidate_idx], EMBEDDINGS[s_idx])
                for s in selected
            )
        else:
            max_sim = 0

        mmr = 0.82 * candidate.final_score - 0.18 * max_sim

        # Hard category cap: no more than 35% of slate from one category
        cat_count = sum(1 for s in selected if s.category == candidate.category)
        if cat_count >= ceil(n * 0.35):
            continue

        if mmr > best_mmr:
            best_mmr = mmr
            best = candidate

    if best:
        selected.append(best)
        remaining.remove(best)

# Result: 10 articles spanning 3+ categories
```

---

## Step 6: Generate Explanation

```python
top_categories = [a.category for a in selected[:3]]

if groq_available:
    prompt = f"""
    User mood: {user.mood}, time: {user.time_of_day}.
    Recent interests: {list(user.interests.keys())[:3]}.
    Top articles: {[a.title for a in selected[:3]]}.
    In 1-2 sentences, explain why these articles suit them.
    """
    explanation = groq_llm.invoke(prompt)
else:
    explanation = (
        f"For your {user.mood} {user.time_of_day} mood, showing "
        f"{', '.join(top_categories)} stories "
        f"based on your reading history and current context."
    )
```

---

## Step 7: Log & Return

```python
log_recommendation_event(user_id, surface, mood, mode, request_id, articles)
update_user_session(user)

return {
    "articles": [serialize(a) for a in selected],
    "explanation": explanation,
    "mode": "rl",   # or cold_start, rag
    "request_id": request_id,
    "user_id": user_id,
}
```

---

## Feedback Flow (Closing the Loop)

When the user clicks "Read" on article E12345:

```
POST /feedback { article_id: "E12345", action: "read_full", dwell_time: 45 }

1. reward = 1.0 × clip(0.45 + 0.65 × (45/30), 0.25, 1.35) = 1.0 × 1.425 = 1.35

2. user.interests["entertainment"] += 1.35 × 0.25 = +0.34
   Apply EMA decay to all categories: interest[c] *= (1 - 0.05)

3. Build context_vector (391-d) for this article + current user state

4. bandit.A += context ⊗ context    (Sherman-Morrison)
   bandit.b += 1.35 × context

5. Propagate:
   - All "entertainment" articles: reward × 0.06
   - KG neighbors of E12345: reward × 0.12

6. save_reading_vector(user_id, "E12345", embedding, weight=1.35)

7. user.recent_clicks.append("E12345")
   user.total_positive_interactions += 1

8. Save session to Redis
```

Next feed request incorporates this feedback: the bandit has updated θ, user profile vector shifts toward entertainment embeddings, and E12345's KG neighbors get a boost.

---

## Candidate Source Distribution (Typical)

```
FAISS:   50–60%  (strongest semantic signal)
KG:      25–35%  (novelty and serendipity)
Memory:  10–15%  (long-term personalisation)
```

In practice, the bandit learns to prefer KG candidates when they have higher click rates in specific contexts — this is why the bandit context vector includes `kg_score`.

---

## Mode Selection

| Condition | Mode | Path |
|---|---|---|
| total_positive_interactions < 5 | `cold_start` | Bio + category + popularity |
| surface=search | `rag` | Hybrid retrieval + lighter LTR |
| surface=feed, warm user | `rl` | FAISS + KG + LTR + LinUCB |

---

## Latency Budget (Target)

```
Session load:        5–15 ms   (Redis hit)
Candidate pool:      30–50 ms  (FAISS + KG BFS)
Scoring (200 cands): 20–40 ms  (numpy vectorised)
LTR:                 5–10 ms   (LightGBM predict)
Bandit:              5–10 ms   (matrix multiply)
MMR:                 5–10 ms   (pairwise cosine)
Explanation:         0–50 ms   (Groq) or 0ms (static)
Session save:        5–10 ms   (Redis write)
Total:               ~100–200 ms
```
