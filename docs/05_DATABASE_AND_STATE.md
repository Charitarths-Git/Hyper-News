# HyperNews — Database & State Management

## Storage Layer Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Hot Path (< 5ms)                         │
│  In-process dict:  USERS[user_id] → UserProfile             │
│  Redis:            history:{user_id} sorted set (TTL 1hr)   │
└─────────────────────────────────────────────────────────────┘
          ↓ on miss
┌─────────────────────────────────────────────────────────────┐
│                     Warm Path (5–30ms)                       │
│  PostgreSQL / SQLite:                                        │
│    users, auth_users, reading_history_vectors,               │
│    feedback_events, user_recent_state                        │
└─────────────────────────────────────────────────────────────┘
          ↓ optional
┌─────────────────────────────────────────────────────────────┐
│                     Vector Stores (optional)                  │
│  Qdrant:     article embeddings, user memory vectors         │
│  pgvector:   reading_history_vectors (if PostgreSQL)         │
│  OpenSearch: BM25 full-text search index                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Database Schema

### Table: `auth_users`

Stores registered user accounts and their complete bio profile.

```sql
CREATE TABLE auth_users (
    user_id              TEXT PRIMARY KEY,
    email                TEXT UNIQUE NOT NULL,
    password_hash        TEXT NOT NULL,         -- scrypt format
    display_name         TEXT,
    created_at           TIMESTAMP,

    -- Bio profile (collected at onboarding)
    age_bucket           TEXT,                  -- "<18","18-24","25-34","35-44","45-54","55-64","65+"
    gender               TEXT,                  -- "male","female","nonbinary","prefer_not"
    occupation           TEXT,                  -- "student","engineer","doctor",...
    location_region      TEXT,                  -- "asia","europe","north_america",...
    location_country     TEXT,

    -- Interest profile
    interest_text        TEXT,                  -- free-form: "I like action movies and cricket"
    top_categories_json  TEXT,                  -- JSON array: ["sports","technology","movies"]

    -- Embeddings
    bio_embedding_json       TEXT,              -- JSON array: 64-d float (BioEncoder output)
    bio_text_embedding_json  TEXT,              -- JSON array: 384-d float (SentenceTransformer)

    -- Consent & onboarding
    affect_consent           INTEGER DEFAULT 0, -- 1 = consented to camera mood detection
    onboarding_completed     INTEGER DEFAULT 0,
    onboarding_completed_at  TIMESTAMP
);
```

### Table: `users`

Lightweight interest/history store (used for both guests and registered users).

```sql
CREATE TABLE users (
    user_id         TEXT PRIMARY KEY,
    interests       TEXT,              -- JSON: {"sports": 0.42, "technology": 0.28}
    reading_history TEXT,              -- JSON array of article IDs (all-time)
    avg_dwell_time  REAL DEFAULT 30.0,
    total_reads     INTEGER DEFAULT 0
);
```

### Table: `reading_history_vectors`

Per-article embedding + feedback weight for long-term memory retrieval.

```sql
CREATE TABLE reading_history_vectors (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          TEXT NOT NULL,
    article_id       TEXT NOT NULL,
    embedding_json   TEXT NOT NULL,    -- JSON: 384-d float32
    feedback_weight  REAL DEFAULT 1.0, -- reward at time of read: -1 to 2.5
    read_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, article_id)        -- one entry per article per user
);
```

Used by `build_long_term_memory_signal()` to retrieve the user's historically preferred articles via cosine similarity (pgvector if available, FAISS fallback).

### Table: `feedback_events`

Raw event log for every user interaction.

```sql
CREATE TABLE feedback_events (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          TEXT NOT NULL,
    article_id       TEXT NOT NULL,
    action           TEXT NOT NULL,    -- click|read_full|save|skip|not_interested|less_from_source
    dwell_time       REAL DEFAULT 0,
    position         INTEGER DEFAULT -1,
    query_text       TEXT DEFAULT '',
    source_feedback  TEXT DEFAULT '',
    session_id       TEXT DEFAULT '',
    request_id       TEXT DEFAULT '',
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Table: `recommendation_events`

Logs each recommendation batch served to a user.

```sql
CREATE TABLE recommendation_events (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id               TEXT NOT NULL,
    surface               TEXT,              -- feed|search
    mood                  TEXT,
    mode                  TEXT,              -- cold_start|rl|rag
    query_text            TEXT,
    candidate_sources_json TEXT,             -- {"faiss":60,"kg":40,"memory":20}
    impression_ids_json   TEXT,              -- JSON array of article_ids shown
    request_id            TEXT,
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Table: `ranking_feature_events`

Used to collect training data for LightGBM LTR model.

```sql
CREATE TABLE ranking_feature_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id   TEXT NOT NULL,    -- groups all articles in one impression
    article_id   TEXT NOT NULL,
    position     INTEGER,
    label        INTEGER DEFAULT 0, -- 0=skip, 1=click, 2=read_full, 3=save, -1=not_interested
    features_json TEXT,             -- score breakdown dict
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Table: `search_events`

```sql
CREATE TABLE search_events (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          TEXT NOT NULL,
    query_text       TEXT,
    normalized_query TEXT,
    intent_json      TEXT,          -- {"embedding": [...], "entities": [...]}
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Table: `user_recent_state`

Snapshot of session state for warm restarts (when Redis is unavailable).

```sql
CREATE TABLE user_recent_state (
    user_id                    TEXT PRIMARY KEY,
    recent_clicks_json         TEXT,   -- JSON array, max 25
    recent_skips_json          TEXT,   -- JSON array, max 25
    recent_negative_actions_json TEXT,
    recent_queries_json        TEXT,   -- JSON array, max 10
    recent_entities_json       TEXT,
    recent_sources_json        TEXT,
    updated_at                 TIMESTAMP
);
```

---

## UserProfile (In-Memory Object)

```python
@dataclass
class UserProfile:
    # Identity
    user_id: str
    display_name: str = ""
    email: str = ""

    # Interests (EMA-updated)
    interests: Dict[str, float] = field(default_factory=dict)
    # e.g. {"sports": 0.42, "technology": 0.28, "entertainment": 0.15}

    # History
    reading_history: List[str] = field(default_factory=list)  # all-time, max 500
    recent_clicks: List[str] = field(default_factory=list)    # last 25
    recent_skips: List[str] = field(default_factory=list)     # last 25
    recent_negative_actions: List[str] = field(default_factory=list)
    session_topics: List[str] = field(default_factory=list)   # categories this session
    recent_queries: List[str] = field(default_factory=list)   # search queries
    recent_entities: List[str] = field(default_factory=list)  # NER entities clicked
    recent_sources: List[str] = field(default_factory=list)   # news sources clicked

    # Context
    mood: str = "neutral"
    time_of_day: str = "morning"
    avg_dwell_time: float = 30.0

    # Bio
    age_bucket: str = "unknown"
    gender: str = "unknown"
    occupation: str = "unknown"
    location_region: str = "unknown"
    location_country: str = ""
    interest_text: str = ""
    top_categories: List[str] = field(default_factory=list)
    affect_consent: bool = False

    # Embeddings
    bio_embedding: List[float] = field(default_factory=list)       # 64-d
    bio_text_embedding: List[float] = field(default_factory=list)  # 384-d

    # Stats
    total_positive_interactions: int = 0
    interest_update_count: int = 0
    onboarding_completed: bool = False
```

---

## Interest Update Algorithm

Interests are updated via Exponential Moving Average (EMA) to prevent any one category from dominating permanently.

### On Positive Feedback (click, read_full, save)

```python
INTEREST_EMA_ALPHA = 0.05
INTEREST_NUDGE_FACTOR = 0.25

# Nudge upward
user.interests[category] = (
    (1 - INTEREST_EMA_ALPHA) * user.interests.get(category, 0)
    + INTEREST_EMA_ALPHA * (user.interests.get(category, 0) + reward * INTEREST_NUDGE_FACTOR)
)

# Decay all other categories toward zero
for cat in user.interests:
    if cat != category:
        user.interests[cat] *= (1 - INTEREST_EMA_ALPHA)

user.total_positive_interactions += 1
```

### Concentration Penalty

If one category accumulates > 50% of total interest weight, it gets penalised:

```python
total = sum(max(v, 0) for v in user.interests.values())
for cat, weight in user.interests.items():
    share = weight / total
    if share > INTEREST_DOMINANCE_THRESHOLD (0.50):
        user.interests[cat] *= (1 - share)  # penalty proportional to dominance
```

This prevents a user who clicks 20 sports articles from getting 100% sports forever.

### Warm-up Ramp

For users with < 5 positive interactions, category interest scores are linearly ramped:

```python
if total_positive_interactions < INTEREST_WARMUP_INTERACTIONS (5):
    ramp = total_positive_interactions / INTEREST_WARMUP_INTERACTIONS
    effective_interest = raw_interest * ramp
```

This reduces noise in early sessions where one accidental click would dominate.

---

## Redis Session Schema

```
Key:   history:{user_id}
Type:  Sorted Set
Score: Unix timestamp
Value: article_id

TTL:   3600 seconds

Commands:
  ZADD history:{user_id} {timestamp} {article_id}
  ZREVRANGE history:{user_id} 0 24   → last 25 clicks
  ZCARD history:{user_id}            → total history length
  EXPIRE history:{user_id} 3600      → refresh TTL
```

Redis is optional. If unavailable, the system falls back to a Python `TTLCache` dict and PostgreSQL.

---

## Authentication

### Password Hashing (scrypt)

```python
# Hash
N, r, p = 2**14, 8, 1
salt = os.urandom(16)
dk = hashlib.scrypt(password.encode(), salt=salt, n=N, r=r, p=p, dklen=32)
stored = f"scrypt${N}${r}${p}${salt.hex()}${dk.hex()}"

# Verify (timing-safe)
hmac.compare_digest(stored_hash, computed_hash)
```

scrypt is memory-hard (uses 16MB RAM per hash), making it resistant to GPU-based brute-force attacks.

### NextAuth Session

```typescript
// auth.ts
export const { handlers, auth, signIn, signOut } = NextAuth({
  secret: process.env.AUTH_SECRET,
  session: { strategy: "jwt" },
  providers: [Credentials({ ... })],
  callbacks: {
    jwt({ token, user }) {
      if (user) token.id = user.id;
      return token;
    },
    session({ session, token }) {
      session.user.id = token.id;
      return session;
    }
  }
});
```

Session is a JWT stored in an HTTP-only cookie. The user_id from the JWT is injected into every proxied backend request.

---

## Guest User Flow

Unauthenticated users get a guest session:

```typescript
// localStorage
const guestId = `guest_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`
// e.g. "guest_lq8k2j_x4f9m2"
localStorage.setItem("hypernews_guest_user_id", guestId)
```

Requests go to `/public/recommend` and `/public/feedback` endpoints, which accept `guest_id` instead of a session token.

Guest profiles are stored in the same `users` table and participate in the full ranking pipeline. The only difference: no auth, no cross-device persistence, no email.

---

## Frontend State (RecommendationSurface)

All recommendation state lives in `RecommendationSurface.tsx`:

```typescript
// Core
const [mood, setMood] = useState("neutral")
const [exploreFocus, setExploreFocus] = useState(55)
const [articles, setArticles] = useState<Article[]>([])
const [explanation, setExplanation] = useState("")
const [mode, setMode] = useState("")
const [loading, setLoading] = useState(false)
const [loadingMore, setLoadingMore] = useState(false)
const [hasMore, setHasMore] = useState(true)

// Feedback
const [feedbackMap, setFeedbackMap] = useState<Record<string, string>>({})

// Camera/affect
const [affectEnabled, setAffectEnabled] = useState(false)
const [affectConsented, setAffectConsented] = useState(false)
const [showConsentModal, setShowConsentModal] = useState(false)

// Refs (for use inside callbacks without stale closures)
const moodRef = useRef(mood)
const exploreFocusRef = useRef(exploreFocus)
const activeQueryRef = useRef(activeQuery)
const articlesRef = useRef<Article[]>([])
```

`moodRef` is used instead of `mood` state inside `fetchRecommendations` because React closures capture the stale state value at callback creation time. The ref is always current.

### LocalStorage Keys

```
hypernews_mood              → "neutral" | "happy" | "curious" | "stressed" | "tired"
hypernews_explore_focus     → "55" (string, parsed to number)
hypernews_guest_user_id     → "guest_lq8k2j_x4f9m2"
hypernews_affect_consent    → "true" (string)
```
