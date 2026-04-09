# HyperNews — Cold Start & Onboarding

## The Cold-Start Problem

A new user has no reading history, no interest weights, no bandit feedback. The FAISS profile vector requires at least one clicked article to compute a centroid. The LinUCB bandit has no signal. Standard collaborative filtering would default to "most popular articles" — essentially the same feed for everyone.

HyperNews solves this with a **bio-driven cold start**: onboarding collects structured demographic + interest signals that are immediately useful for personalisation, even before the first click.

---

## Cold-Start Gate

```python
COLD_START_THRESHOLD = 5  # total positive interactions

if user.total_positive_interactions < COLD_START_THRESHOLD:
    return cold_start_recommendations(user, DF, EMBEDDINGS, n=10)
```

"Positive interaction" = click, read_full, or save (not skip or not_interested).

After 5 positive interactions, the user transitions to the full RL pipeline permanently. There's no gradual blend — it's a hard threshold.

---

## Onboarding Flow

```
/onboarding page
      │
      ▼
Step 1: Age bucket    ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
Step 2: Gender        ["male", "female", "nonbinary", "prefer_not"]
Step 3: Occupation    ["student", "engineer", "doctor", "teacher", "artist",
                       "business", "journalist", "researcher", "other"]
Step 4: Location      [region dropdown → country dropdown]
Step 5: Interests     Free-text: "I like cricket, action movies, and tech startups"
Step 6: Categories    Multi-select from 12 categories (top 3)
      │
      ▼
POST /api/me/profile → saves to auth_users
      │
      ▼
Redirect to feed (now uses bio-based cold start)
```

---

## Bio Encoding: BioEncoder

The structured onboarding fields are fed into a learned embedding model.

### Input Features (206-d one-hot)

```python
age_buckets     = ["<18","18-24","25-34","35-44","45-54","55-64","65+"]      # 7-d
genders         = ["male","female","nonbinary","prefer_not"]                  # 4-d
occupations     = ["student","engineer","doctor","teacher","artist",
                   "business","journalist","researcher","other"]               # 9-d
regions         = ["asia","europe","north_america","south_america",
                   "africa","middle_east","oceania"]                           # 7-d

# One-hot encode each field
age_vec  = one_hot(age_bucket, age_buckets)       # (7,)
gen_vec  = one_hot(gender, genders)               # (4,)
occ_vec  = one_hot(occupation, occupations)       # (9,)
reg_vec  = one_hot(location_region, regions)      # (7,)

# Category interest scores (normalised)
cat_vec  = [user.interests.get(cat, 0) for cat in CATEGORIES]  # (12,)

bio_input = np.concatenate([age_vec, gen_vec, occ_vec, reg_vec, cat_vec])
# shape: (39,)  — note: simplified here, full model may include more fields
```

### BioEncoder Architecture

```python
class BioEncoder(nn.Module):
    def __init__(self, input_dim=39, hidden_dim=128, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        emb = self.net(x)
        return F.normalize(emb, p=2, dim=-1)  # L2-normalised 64-d output
```

Output: **64-d bio embedding** stored in `auth_users.bio_embedding_json`.

### Text Embedding

The free-text `interest_text` field is separately encoded:

```python
# "I like cricket, action movies, and tech startups"
bio_text_embedding = SENTENCE_MODEL.encode(interest_text)  # 384-d
# Stored in auth_users.bio_text_embedding_json
```

---

## Cold-Start Scoring

```python
def cold_start_recommendations(user, DF, EMBEDDINGS, n=10):
    scores = {}
    for _, row in DF.iterrows():
        score = 0.0

        # 1. Category affinity from top_categories
        if row.category in user.top_categories:
            score += 0.5

        # 2. Category affinity from interests (onboarding multi-select)
        score += user.interests.get(row.category, 0) * 0.4

        # 3. Bio text embedding similarity
        if user.bio_text_embedding:
            bio_vec = np.array(user.bio_text_embedding)
            article_vec = EMBEDDINGS[news_id_to_idx[row.news_id]]
            text_sim = float(bio_vec @ article_vec)  # cosine (both normalised)
            score += text_sim * 0.3

        # 4. Occupation → category affinity
        occ_boosts = OCCUPATION_CATEGORY_AFFINITIES.get(user.occupation, {})
        score += occ_boosts.get(row.category, 0) * 0.2

        # 5. Popularity boost
        score += row.popularity * 0.15

        # 6. Location match (country or region)
        if row.location_country == user.location_country:
            score += 0.1
        elif row.location_region == user.location_region:
            score += 0.05

        scores[row.news_id] = score

    top_ids = sorted(scores, key=scores.get, reverse=True)[:n]
    return [DF[DF.news_id == nid].iloc[0] for nid in top_ids]
```

### Occupation → Category Affinities

```python
OCCUPATION_CATEGORY_AFFINITIES = {
    "student":    {"education": 0.4, "technology": 0.3, "science": 0.2},
    "engineer":   {"technology": 0.4, "science": 0.3, "business": 0.15},
    "doctor":     {"health": 0.5, "science": 0.3, "technology": 0.1},
    "teacher":    {"education": 0.4, "science": 0.2, "lifestyle": 0.1},
    "journalist": {"politics": 0.3, "entertainment": 0.3, "world": 0.2},
    "business":   {"business": 0.4, "finance": 0.3, "technology": 0.2},
    "artist":     {"entertainment": 0.4, "lifestyle": 0.3, "culture": 0.2},
    "researcher": {"science": 0.4, "technology": 0.3, "health": 0.1},
    "other":      {},  # no prior
}
```

---

## Guest Cold Start

Unauthenticated users start with a randomly generated guest ID and zero profile data:

```typescript
// frontend: localStorage
const guestId = `guest_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`
```

Guest cold start uses only:
1. Default popular articles (popularity score only)
2. Manual mood selection (user clicks a pill)
3. Rapidly bootstrapped via clicks (hits warm pipeline at 5 interactions)

Guests have no bio, no occupation affinity, no text embedding — but the system becomes useful faster than a registered user because they start clicking immediately rather than going through onboarding.

---

## Interest Warm-Up Ramp

Even after collecting interests at onboarding, early signals are noisy. One accidental click on a politics article shouldn't dominate the profile.

```python
INTEREST_WARMUP_INTERACTIONS = 5

if user.total_positive_interactions < INTEREST_WARMUP_INTERACTIONS:
    ramp = user.total_positive_interactions / INTEREST_WARMUP_INTERACTIONS
    # Scale all interest scores down proportionally
    effective_interests = {k: v * ramp for k, v in user.interests.items()}
else:
    effective_interests = user.interests
```

At 0 interactions: ramp=0 → interests are all zero → relies entirely on bio affinity + popularity.
At 3 interactions: ramp=0.6 → 60% of learned interests, 40% from cold-start signals.
At 5+ interactions: ramp=1.0 → full RL pipeline.

---

## Transition to Full Pipeline

When `total_positive_interactions` crosses 5:

1. Next `/recommend` call skips `cold_start_recommendations()`
2. FAISS profile vector computed from `recent_clicks[-15:]`
3. LinUCB bandit starts making predictions (initially with high uncertainty → high exploration)
4. LTR model receives real feature vectors
5. Interest EMA has 5 updates → meaningful category weights

The transition is seamless — the user doesn't notice. The explanation banner changes from "Showing popular articles based on your interests" to "For your curious morning mood, showing technology and science stories based on your reading history."

---

## Cold-Start Mode Indicator

The response includes a `mode` field:

```json
{
  "articles": [...],
  "mode": "cold_start",    // or "rl" or "rag"
  "explanation": "Showing articles based on your interests and what's popular right now."
}
```

The frontend `ExplanationBanner` renders this identically — the mode is used for analytics and logging, not UI differentiation.
