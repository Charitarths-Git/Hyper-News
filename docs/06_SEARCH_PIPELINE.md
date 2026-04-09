# HyperNews — Search Pipeline

## Overview

Search is an entirely separate execution path from the feed. A user searching "AI ethics" should get semantically relevant results, not feed recommendations polluted by their browsing history.

```
POST /recommend { surface="search", query="AI ethics", n=10 }
          │
          ▼
1. Encode query → 384-d embedding
          │
          ▼
2. build_hybrid_candidates()
   ├── FAISS semantic top-40
   ├── BM25 lexical top-40
   ├── KG entity expansion top-30
   └── RRF fusion → ~80 unique candidates
          │
          ▼
3. Optional cross-encoder reranking (top-20 → top-10)
          │
          ▼
4. rank_search_articles() → LTR scoring (lighter weights)
          │
          ▼
5. Store search_context in session
          │
          ▼
6. Return top-10 with explanation
```

---

## Step 1: Query Encoding

```python
# SentenceTransformer: all-MiniLM-L6-v2
query_embedding = SENTENCE_MODEL.encode(query, normalize_embeddings=True)
# shape: (384,), L2-normalised → dot product = cosine similarity
```

The model handles multi-word queries, misspellings via subword tokenisation, and semantic equivalents ("AI" ≈ "artificial intelligence" ≈ "machine learning").

---

## Step 2: Hybrid Candidate Retrieval

### 2a. FAISS Semantic Search

```python
scores, indices = FAISS_INDEX.search(query_embedding.reshape(1, -1), k=40)
semantic_candidates = [
    {"news_id": DF.iloc[idx].news_id, "score": float(scores[0][i]), "source": "faiss"}
    for i, idx in enumerate(indices[0])
]
```

Finds articles whose title+abstract embedding is cosine-close to the query. Handles paraphrase and semantic drift but misses exact keyword matches.

### 2b. BM25 Lexical Search

```python
# rank_bm25.BM25Okapi over tokenised article titles + abstracts
tokenised_query = query.lower().split()
bm25_scores = BM25_INDEX.get_scores(tokenised_query)
top_bm25_indices = np.argsort(bm25_scores)[::-1][:40]
lexical_candidates = [
    {"news_id": DF.iloc[idx].news_id, "score": bm25_scores[idx], "source": "bm25"}
    for idx in top_bm25_indices if bm25_scores[idx] > 0
]
```

Exact term matching. Complementary to FAISS: catches rare proper nouns ("Rajiv Gandhi", "GPT-4o") that embedding models may not represent well.

### 2c. KG Entity Expansion

```python
# Extract named entities from query via spaCy NER
query_entities = [ent.text for ent in nlp(query).ents]  # ["AI", "ethics"]

related_ids = set()
for entity in query_entities:
    if entity in KG_GRAPH:
        neighbors = list(nx.neighbors(KG_GRAPH, entity))
        related_ids.update(neighbors)

kg_candidates = [
    {"news_id": nid, "score": 0.6, "source": "kg"}
    for nid in related_ids if nid in news_id_to_idx
][:30]
```

Finds articles connected to entities in the query through the knowledge graph. Enables thematic expansion: query "Elon Musk" might surface Tesla and SpaceX articles.

### 2d. RRF Fusion

Reciprocal Rank Fusion combines the three ranked lists without needing score normalisation:

```python
RRF_K = 60  # standard constant, reduces sensitivity to outlier ranks

def rrf_score(rank: int) -> float:
    return 1.0 / (RRF_K + rank)

# Build unified score map
article_scores: Dict[str, float] = {}
for ranked_list in [semantic_candidates, lexical_candidates, kg_candidates]:
    for rank, item in enumerate(ranked_list):
        nid = item["news_id"]
        article_scores[nid] = article_scores.get(nid, 0) + rrf_score(rank)

# Sort by fused score
fused = sorted(article_scores.items(), key=lambda x: x[1], reverse=True)
```

An article appearing at rank 3 in FAISS and rank 5 in BM25 scores:
```
1/(60+3) + 1/(60+5) = 0.01587 + 0.01538 = 0.03125
```
vs an article only in FAISS at rank 1:
```
1/(60+1) = 0.01639
```
The multi-source article wins despite lower individual ranks.

---

## Step 3: Optional Cross-Encoder Reranking

If a cross-encoder model is available (sentence-transformers/ms-marco-MiniLM-L-6-v2):

```python
# Take top-20 from RRF, rerank with cross-encoder
top_candidates = fused[:20]
pairs = [(query, DF.loc[nid].abstract) for nid, _ in top_candidates]
ce_scores = CROSS_ENCODER.predict(pairs)

reranked = sorted(
    zip(top_candidates, ce_scores),
    key=lambda x: x[1],
    reverse=True
)
```

Cross-encoders jointly encode query+document (vs bi-encoder which encodes separately). Much slower (~100ms for 20 pairs) but significantly better precision. Used only for search, never feed.

If cross-encoder is unavailable (default in dev), the RRF-fused order is used directly.

---

## Step 4: Search-Specific LTR Scoring

Search uses a lighter version of the LTR features, emphasising query-relevance signals over interest personalisation:

```python
search_features = {
    "semantic_score":    query_embedding @ EMBEDDINGS[article_idx],  # query cosine
    "bm25_score":        bm25_scores[idx],
    "rrf_score":         article_scores[nid],
    "entity_overlap":    len(set(article_entities) & set(query_entities)),
    "recency_score":     recency_factor(article.publish_date),  # newer = higher
    "popularity_score":  article.popularity,
    # Note: mood weights NOT applied in search — query intent overrides mood
    # Note: bandit NOT used in search — no UCB exploration on query results
}
ltr_score = search_ltr_predict(search_features)
```

Key differences from feed scoring:
- No mood/time multipliers (query trumps mood)
- No bandit score (deterministic)
- Recency weighted more heavily (recent articles more relevant to queries)

---

## Step 5: Search Context Storage

After returning results, the search context is stored in the user session:

```python
search_context = {
    "query": query,
    "embedding": query_embedding.tolist(),  # 384-d
    "entities": query_entities,             # ["AI", "ethics"]
    "timestamp": time.time(),
    "ttl": 600                              # 10-minute expiry
}
user.recent_queries.append(query)           # for interest update
user.recent_entities.extend(query_entities) # for next feed's KG scoring
```

This means: if you search for "climate change" and then open the feed, the feed's KG scoring will have climate/environment entities in `user.recent_entities`, giving a gentle lift to related articles.

---

## Search vs Feed: Key Differences

| Property | Feed | Search |
|---|---|---|
| Candidate source | FAISS + KG + Memory | FAISS + BM25 + KG (RRF) |
| Ranking driver | LTR + LinUCB bandit | LTR only |
| Query | Always `None` | Required, encoded to 384-d |
| Mood weights | Applied | NOT applied |
| UCB exploration | Yes (15% weight on uncertainty) | No |
| Cross-encoder | Never | Optional (top-20 rerank) |
| Session pollution | Not possible | Entities stored (gentle signal) |
| Explanation | Interest + mood context | Query intent |

The "session pollution" rule is critical: a search for "cancer treatment" must not inject health articles into the entertainment feed. Entity propagation is intentional and limited (10-min TTL, low weight).

---

## Query Intent Classification

For the explanation banner, the search query is classified by intent:

```python
def classify_query_intent(query: str, embedding: np.ndarray) -> str:
    # Simple heuristics + category similarity
    category_embeddings = {cat: SENTENCE_MODEL.encode(cat) for cat in CATEGORIES}
    similarities = {cat: float(embedding @ e) for cat, e in category_embeddings.items()}
    top_cat = max(similarities, key=similarities.get)
    return top_cat  # "technology", "sports", etc.
```

Used to generate the explanation:
```
"Showing results for 'AI ethics' — technology and science articles 
 matching your query with recent coverage prioritised."
```

---

## BM25 Index Construction

Built at startup from all article titles + abstracts:

```python
from rank_bm25 import BM25Okapi

corpus = [
    (row.title + " " + (row.abstract or "")).lower().split()
    for _, row in DF.iterrows()
]
BM25_INDEX = BM25Okapi(corpus)
# ~51K documents, builds in ~2 seconds, ~50MB RAM
```

BM25 parameters (Okapi defaults):
- `k1 = 1.5` (term frequency saturation)
- `b = 0.75` (document length normalisation)

---

## Fallback: OpenSearch

If OpenSearch is configured (`OPENSEARCH_URL` env var), BM25 queries are routed there instead of in-memory `rank_bm25`. OpenSearch provides:
- Persistent index (no rebuild on restart)
- Fuzzy matching ("artficial" → "artificial")
- Highlight snippets
- Aggregations for analytics

The in-memory BM25 is the default for dev/lightweight deployments.
