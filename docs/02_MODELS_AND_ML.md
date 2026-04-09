# HyperNews — Models & ML Techniques

## Model Inventory

| Model | Type | Size | Location | Used For |
|---|---|---|---|---|
| SentenceTransformer (all-MiniLM-L6-v2) | Transformer | ~80 MB | HuggingFace | Article/query encoding → 384-d |
| FAISS Index | Vector Index | ~50 MB | data/faiss_mind.index | ANN search |
| LinUCB Bandit | Contextual Bandit | ~2 MB | models/bandit_model.pkl | Feed ranking |
| LightGBM LTR | Gradient Boosting | ~1 MB | models/ltr_model.txt | Feature scoring |
| NetworkX KG | Graph | ~30 MB RAM | graph/knowledge_graph.pkl | Semantic links |
| BioEncoder | MLP (PyTorch) | ~8 MB | models/bio_encoder.pt | User cold-start |
| DQN Policy | Dueling DQN (PyTorch) | ~8 MB | models/dqn_policy.pt | Slate RL (Phase 4) |
| TinyFaceDetector | CNN (TF.js) | ~190 KB | public/faceapi-models/ | Face detection in browser |
| FaceExpressionNet | CNN (TF.js) | ~310 KB | public/faceapi-models/ | Expression → mood |

---

## 1. Article Embeddings (SentenceTransformer)

### What it does
Encodes each article's `title + " " + abstract` into a 384-dimensional dense vector that captures semantic meaning. Similar articles are close in this space.

### Model
`sentence-transformers/all-MiniLM-L6-v2` — a distilled MiniLM fine-tuned on semantic similarity tasks. Chosen for speed/quality tradeoff: 384-d output, ~5ms per sentence on CPU.

### How it's used

```python
# At startup — precomputed once
embeddings = np.load("data/article_embeddings.npy")  # shape: (51000, 384)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # L2 normalize

# User profile vector — centroid of recent clicks
user_profile_vector = np.mean([embeddings[idx] for idx in recent_click_indices], axis=0)
user_profile_vector /= np.linalg.norm(user_profile_vector)

# Similarity
score = user_profile_vector @ article_embedding  # dot product = cosine (since L2 normed)
```

### Normalization
All embeddings are L2-normalized at startup. This means dot product equals cosine similarity, which FAISS can exploit for fast exact cosine search.

---

## 2. FAISS Index

### What it is
Facebook AI Similarity Search — a library for fast approximate nearest-neighbor (ANN) search in high-dimensional spaces.

### Configuration
```python
dimension = 384
index = faiss.IndexFlatIP(dimension)  # Inner product = cosine (on L2-normed vectors)
index.add(embeddings)                 # Add all 51K article embeddings
```

`IndexFlatIP` = exact brute-force inner product search. No approximation errors but ~10ms for 51K × 384.

### Query
```python
scores, indices = index.search(user_vector.reshape(1, -1), k=60)
# Returns top-60 most similar articles by cosine similarity
```

### Why 60?
Candidates go through ranking, so we oversample. 60 from FAISS + 40 from KG + 20 from memory = ~120 total, ranked down to 10.

---

## 3. LinUCB Contextual Bandit

### What problem it solves
The exploration-exploitation dilemma: should we recommend articles we're confident the user likes (exploit) or try new ones to learn their preferences (explore)?

### Algorithm: Linear Upper Confidence Bound

The core idea: model the reward of each article as a linear function of its context, and add an uncertainty bonus to encourage exploration of under-served arms.

```
For each candidate article:
  θ = A⁻¹ b                          # estimated reward weights (391-d)
  mean_reward = θ · context           # expected reward
  uncertainty = √(context · A⁻¹ · context)   # how uncertain we are
  
  score = mean_reward + α × uncertainty   # UCB score
  
  α = 0.15  (exploration parameter — higher = more exploration)
```

### State Matrices

```
A  ∈ ℝ^(391×391)   Gram matrix: accumulates context outer products
                    Initialized as: ridge_lambda × I (identity)
                    Updated per feedback: A += context ⊗ context

A⁻¹               Maintained via Sherman-Morrison formula (O(d²) update)
                    Avoids O(d³) matrix inversion on every feedback

b  ∈ ℝ^391         Reward accumulator
                    Updated: b += reward × context
```

### Context Vector (391-d)

```
[article_embedding (384-d),   ← semantic content of the article
 mood_normalized (1-d),        ← current user mood, mapped to [0,1]
 time_of_day_normalized (1-d), ← morning/afternoon/evening/night → [0,1]
 click_count_normalized (1-d), ← recent engagement level
 skip_penalty (1-d),           ← recent disengagement signal
 category_interest (1-d),      ← user's historical weight for this category
 kg_score (1-d)]               ← knowledge graph affinity
```

The 384-d article embedding is the dominant signal. The 7 scalars add user context so the bandit learns mood-conditional and time-conditional preferences.

### Sherman-Morrison Update

Instead of inverting a 391×391 matrix every feedback, we use the rank-1 update formula:

```
A_new = A + x·xᵀ
A_new⁻¹ = A⁻¹ - (A⁻¹·x·xᵀ·A⁻¹) / (1 + xᵀ·A⁻¹·x)
```

This is O(d²) = O(391²) ≈ 150K operations vs O(d³) ≈ 60M for full inversion.

### Reward Computation

```python
REWARD_MAP = {
  "click":           0.5,
  "read_full":       1.0,
  "save":            2.0,
  "skip":           -0.3,
  "not_interested": -0.8,
  "less_from_source":-0.6,
}

dwell_ratio = dwell_time / max(avg_dwell_time, 30)
multiplier = clip(0.45 + 0.65 × dwell_ratio, 0.25, 1.35)
final_reward = base_reward × multiplier   # ∈ [-1, 2.5]
```

### Epsilon-Greedy Fallback

With probability ε=0.02 (2%), the bandit ignores the UCB formula and samples from N(mean_reward, uncertainty). This provides uniform exploration across the action space.

### Reward Propagation

When a user reads article A in category "technology":

```
1. Update bandit for article A directly
2. Propagate to all articles in "technology" with decay 0.06
3. Propagate to KG-connected articles (1-hop neighbors) with decay 0.12
```

This allows one strong positive signal to bootstrap adjacent article weights.

---

## 4. LightGBM Learning-to-Rank (LTR)

### What it does
Learns to rank articles by combining multiple weak signals into a single strong relevance score, using gradient boosting on ranking objectives.

### Algorithm: LambdaMART

LambdaMART is a gradient boosting algorithm that optimises NDCG (Normalized Discounted Cumulative Gain) — a standard ranking metric that weights correct placement at the top of the list more heavily.

### Features (9-10 inputs)

```
semantic_score      cosine(user_profile, article_embedding) ∈ [0, 1]
retrieval_score     FAISS/KG/memory source weight ∈ [0, 1]
lexical_score       BM25-style token overlap ∈ [0, 1]
memory_score        similarity to past reads (pgvector) ∈ [0, 1]
kg_score            knowledge graph path bonus ∈ [0, 1]
entity_score        entity overlap with recent clicks ∈ [0, 1]
subcategory_score   subcategory match ∈ [0, 1]
popularity_score    article popularity ∈ [0, 1]
context_score       mood × time_of_day multiplier ∈ [0.3, 1.5]
negative_score      skip/not-interested penalty ∈ [0, 1]
```

### Fallback Weights (when no trained model)

```python
weights = {
  "semantic_score":    0.24,
  "retrieval_score":   0.18,
  "lexical_score":     0.15,
  "memory_score":      0.12,
  "kg_score":          0.10,
  "entity_score":      0.09,
  "subcategory_score": 0.06,
  "popularity_score":  0.04,
  "context_score":     0.04,
  "negative_score":   -0.06,
}
```

### Training Data

Collected via `ranking_feature_events` table:
- Label 0: skip, not_interested
- Label 1: click
- Label 2: read_full
- Label 3: save

One request_id groups all articles shown in that impression. LambdaMART learns to rank label-3 above label-0 within each group.

### Combined Score

```python
ltr_score = ltr_model.predict(features)
bandit_score = bandit.score(article_id, context, category)

# explore_focus ∈ [0, 100], default 55
# High explore_focus → trust LTR more (focus on known preferences)
# Low explore_focus → trust bandit more (explore new territory)

final_score = ltr_score × (explore_focus / 100) + bandit_score × (1 - explore_focus / 100)
```

---

## 5. Knowledge Graph

### Structure

A NetworkX undirected graph with ~18K nodes and edges typed by relationship.

```
Node types:
  article::E12345      ← 51K article nodes
  category::sports     ← ~50 category nodes
  subcategory::sports::soccer  ← ~500 subcategory nodes
  entity::Elon_Musk    ← ~10K–50K entity nodes (NER extracted)

Edge types:
  article ↔ category      (weight 1.0)
  article ↔ subcategory   (weight 1.0)
  article ↔ entity        (weight 1.0 title match, 0.5 abstract)
  entity  ↔ entity        (weight 1.0 if co-occur in same article)
```

### Building the Graph

```python
for article in df.itertuples():
  G.add_node(f"article::{article.news_id}")
  G.add_edge(f"article::{article.news_id}", f"category::{article.category}")

  # Extract entities (spacy NER or title-case heuristic)
  for entity in extract_entities(article.title, article.abstract):
    G.add_edge(f"article::{article.news_id}", f"entity::{entity}", weight=1.0)

  # Entity co-occurrence edges
  for e1, e2 in combinations(entities, 2):
    G.add_edge(f"entity::{e1}", f"entity::{e2}")
```

### Usage in Ranking

```python
# BFS from recently clicked articles
related = []
for clicked_id in user.recent_clicks[-5:]:
  neighbors = nx.single_source_shortest_path_length(G, f"article::{clicked_id}", cutoff=2)
  related += [n for n in neighbors if n.startswith("article::")]

# KG bonus: articles found in BFS get a bonus score
kg_bonus_map = {article_id: 1.0 / (distance + 1) for article_id, distance in neighbors.items()}
```

### Why Knowledge Graph?

FAISS finds semantically similar articles (same topic, same words). KG finds **structurally related** articles — an article about Elon Musk connects to Tesla, SpaceX, and Twitter even if the embeddings don't overlap perfectly. This handles novelty and serendipitous discovery.

---

## 6. BioEncoder

### Purpose
New users have no reading history. We initialise their interest vector using their signup profile: age, gender, occupation, location, interests.

### Architecture

```
Input:
  cat_indices: [age_bucket_idx, gender_idx, occupation_idx, location_idx]
  text_emb: SentenceTransformer("I like action movies and cricket")  → 384-d

Embedding tables (learned):
  age_emb:        vocab_size=8  → 8-d
  gender_emb:     vocab_size=5  → 8-d
  occupation_emb: vocab_size=12 → 8-d
  location_emb:   vocab_size=7  → 8-d

Concatenate: [384 + 8 + 8 + 8 + 8] = 416-d

MLP:
  Linear(416 → 256) → LayerNorm → GELU → Dropout(0.1)
  Linear(256 → 128) → GELU
  Linear(128 → 64)  → LayerNorm

Output: 64-d bio_embedding
```

### Usage

The 64-d bio_embedding feeds into the DQN state (Phase 4) and augments the cold-start profile vector used to initialise FAISS queries.

---

## 7. DQN Policy (Phase 4)

### What it does
Replaces the LinUCB bandit with a deep reinforcement learning policy that takes the user's full state (history, bio, affect) and recommends a ranked slate of articles.

### Architecture: Dueling DQN

```
State (206-d):
  history_embedding (128-d)    ← Transformer encoder on recent 25 clicks
  bio_embedding (64-d)         ← BioEncoder output
  affect_embedding (2-d)       ← mood one-hot (e.g. [1,0,0,0,0] for happy)
  context_features (12-d)      ← time_of_day, session_len, skip_ratio, etc.

Item (448-d):
  article_embedding (384-d)    ← SentenceTransformer
  kg_embedding (64-d)          ← mean of KG neighbor embeddings

Trunk: Linear(654 → 512 → 256) + LayerNorm + GELU + Dropout(0.1)

Value head:     Linear(256 → 128 → 1)    → V(s): scalar state value
Advantage head: Linear(256 → 128 → 1)    → A(s,a): per-article advantage

Q(s,a) = V(s) + A(s,a) − (1/n)∑A(s,aᵢ)   (dueling decomposition)
```

### Why Dueling?
Separating value (how good is this state?) from advantage (how much better is this article than average?) stabilises training. Many articles have similar Q-values; the advantage head learns to distinguish them with lower variance.

### Training
Offline on MIND `behaviors.tsv` with IPS (Inverse Propensity Scoring) to correct for the logging policy's bias. Without IPS correction, the model would only learn to recommend what the previous system showed users, not what they actually prefer.

### Status: Phase 4 — model architecture implemented, offline training pending

---

## 8. History Encoder

### Purpose
Encode the user's last 25 clicks into a single 128-d vector representing their current interest state.

### Architecture

```
Input: sequence of 25 article embeddings, each 384-d
       (padded with zeros if fewer clicks)

Transformer Encoder:
  d_model = 128
  nhead = 4
  num_layers = 2
  dim_feedforward = 256
  dropout = 0.1
  
Positional encoding: learnable

Pool: mean of all non-padding token outputs → 128-d
```

### Why not just mean-pool raw embeddings?
A Transformer captures the **sequence** of interests — reading 5 technology articles then 5 sports articles is different from interleaved reading. It also learns to weight recent clicks more than old ones through attention.

---

## 9. Affect Sensing (Face Expression Detection)

### Pipeline

```
Browser Webcam (640×480)
        ↓
TinyFaceDetector (TF.js, WebGL)
  - Input: full video frame
  - Output: bounding box + confidence
  - Model: MobileNet-based, ~190KB
  - Threshold: scoreThreshold=0.3
        ↓
Face crop (extracted by face-api internally)
        ↓
FaceExpressionNet (TF.js, WebGL)
  - Input: 224×224 face crop (normalised)
  - Output: 7 class probabilities
    [angry, disgusted, fearful, happy, neutral, sad, surprised]
  - Model: MobileNet-based, ~310KB
        ↓
Argmax → expression label + confidence
        ↓
Mapping to app mood keys:
  happy     → "happy"
  surprised → "curious"
  neutral   → "neutral"
  angry     → "stressed"
  fearful   → "stressed"
  sad       → "tired"
  disgusted → "tired"
        ↓
onMoodDetected(mood) → setMood → fetchRecommendations()
```

### Key Design Decisions

**Why not the ONNX model?**
The original `facial_expression_recognition_mobilefacenet_2022july.onnx` requires a pre-cropped 112×112 face patch. Without a face detector upstream, the full webcam frame (mostly background) produces garbage predictions — always `sad` → mapped to `tired`. Face detection is mandatory before expression classification.

**Why face-api.js?**
`@vladmandic/face-api` bundles TinyFaceDetector + FaceExpressionNet together with proper JavaScript post-processing. It handles the face detection → crop → expression pipeline in one library call, runs entirely in the browser via TF.js WebGL, and requires no server round-trip.

**Why in the browser?**
- Zero latency (no network round-trip)
- No raw video ever leaves the device (GDPR compliance)
- Only the mood string (`"happy"`) is transmitted to the backend
- Runs at ~15ms per frame on integrated GPU

### Timing

Every 5 seconds:
1. Run `faceapi.detectSingleFace(video).withFaceExpressions()`
2. Pick argmax expression
3. If different from current mood → update mood state → re-fetch feed

### Consent

`affect_consent` field on user profile (GDPR Article 9 — biometric data). Displayed as a consent modal on first camera enable. Stored in localStorage (guests) and backend profile (authenticated users).

---

## 10. MMR Diversity

### Problem
Without diversity constraints, the bandit and LTR both optimise for the user's strongest interest. A sports fan would see 10 sports articles — likely boring and filter-bubble-inducing.

### Algorithm: Maximal Marginal Relevance

```python
λ = 0.82   # relevance weight (high — we still mostly want relevance)

selected = []
while len(selected) < n:
  best = argmax over remaining candidates of:
    λ × relevance_score[c]
    - (1-λ) × max(cosine(embed[c], embed[s]) for s in selected)
  
  # Hard constraint: no more than 35% from same category
  if category_count[category[best]] < 0.35 × n:
    selected.append(best)
  else:
    try next best
```

With λ=0.82: relevance is weighted 4.6× more than diversity. This ensures a top article still ranks first even if it's similar to others, but pushes the 3rd-5th articles toward different topics.

### Guarantee
The final 10 articles will always span at least 3 categories.

---

## 11. Mood-Based Category Weights

Applied as a multiplier on article scores **before** passing to LTR/bandit:

```python
MOOD_WEIGHTS = {
  "stressed": {"entertainment": 1.5, "sports": 1.3, "politics": 0.4, "health": 0.5},
  "curious":  {"technology": 1.5, "science": 1.4, "business": 1.2},
  "tired":    {"entertainment": 1.4, "lifestyle": 1.3, "politics": 0.3},
  "happy":    {"sports": 1.2, "entertainment": 1.2, "technology": 1.2},
  "neutral":  {},
}

TIME_WEIGHTS = {
  "morning":   {"politics": 1.3, "business": 1.2, "technology": 1.1},
  "afternoon": {"sports": 1.2, "business": 1.2},
  "evening":   {"entertainment": 1.3, "lifestyle": 1.2},
  "night":     {"entertainment": 1.5, "lifestyle": 1.4, "politics": 0.5},
}

context_score = MOOD_WEIGHTS[mood][category] × TIME_WEIGHTS[time][category]
```

This is the **first level** of mood influence. The bandit's context vector (which includes mood as a scalar) is the **second level** — the bandit learns *which specific articles within a category* match different moods.

---

## 12. RAG Pipeline

### Purpose
Retrieve semantically relevant articles for search queries using dense vector search, then generate natural language explanations.

### Hybrid Retrieval (RRF Fusion)

```
Query: "electric vehicles India"

1. Semantic:  FAISS top-40 by cosine(query_emb, article_embs)
              → articles about EVs, batteries, Tata Motors

2. Lexical:   Inverted token index
              → articles containing "electric", "vehicle", "india"

3. KG:        Entity extraction from query: [India, electric_vehicle]
              → BFS from entity nodes → related articles

4. RRF fusion:
   rrf_score[doc] = Σ 1 / (k + rank_in_list)
   k = 60 (standard RRF constant)

5. Optional: cross-encoder reranker (ms-marco-MiniLM-L6-v2)
             Re-scores top-24 by query-document relevance
```

### Explanation Generation

```python
if GROQ_API_KEY:
  # LLama 3.1-8b-instant via Groq API
  prompt = f"User mood: {mood}. Time: {time}. Articles: {titles}. Explain briefly."
  explanation = groq.invoke(prompt)
else:
  # Deterministic template
  explanation = f"For your {mood} {time} mood, showing {top_categories} stories..."
```
