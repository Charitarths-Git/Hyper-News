# HyperNews — Affect Sensing & Mood System

## Overview

HyperNews personalises the feed along two mood dimensions:
1. **Manual mood** — user clicks a mood pill (Neutral / Curious / Happy / Stressed / Tired)
2. **Auto mood** — webcam reads facial expressions every 5 seconds and updates mood automatically

Both feed into the same backend mood pipeline. The affect sensor just automates step 1.

---

## Manual Mood System

### Frontend

Mood pills in `ContextBar.tsx`:

```tsx
const MOODS = [
  { key: "neutral",  emoji: "Calm",   label: "Neutral"  },
  { key: "curious",  emoji: "Explore",label: "Curious"  },
  { key: "happy",    emoji: "Bright", label: "Happy"    },
  { key: "stressed", emoji: "Light",  label: "Stressed" },
  { key: "tired",    emoji: "Easy",   label: "Tired"    },
];
```

Clicking a pill calls `setMood(key)` which:
1. Stores mood in `localStorage` (persists across refresh)
2. Updates `moodRef.current` immediately (so next fetch uses new mood)
3. Triggers `fetchRecommendations()` on the feed surface

### Backend Encoding

```python
# In ranker.py
mood_map = {"neutral": 0, "happy": 1, "curious": 2, "stressed": 3, "tired": 4}
mood_val = mood_map.get(user.mood, 0) / 4.0   # Normalised to [0, 1]
```

This scalar is position 385 in the 391-d bandit context vector.

---

## Auto Mood: Affect Sensor Pipeline

### Component: `AffectSensor.tsx`

Runs entirely in the browser. No video or image data ever leaves the device.

```
User enables camera → consent modal (first time)
          ↓
navigator.mediaDevices.getUserMedia({ video: 640×480 })
          ↓
    ┌─────────────────────────────────────────┐
    │  Every 5 seconds:                       │
    │                                         │
    │  faceapi.detectSingleFace(video,        │
    │    TinyFaceDetectorOptions({            │
    │      inputSize: 224,                    │
    │      scoreThreshold: 0.3               │
    │    }))                                  │
    │  .withFaceExpressions()                 │
    │          ↓                              │
    │  result.expressions = {                 │
    │    angry:     0.01,                     │
    │    disgusted: 0.00,                     │
    │    fearful:   0.02,                     │
    │    happy:     0.89,   ← argmax          │
    │    neutral:   0.05,                     │
    │    sad:       0.02,                     │
    │    surprised: 0.01,                     │
    │  }                                      │
    │          ↓                              │
    │  expression = "happy" (0.89)            │
    │  mood = EXPR_TO_MOOD["happy"] = "happy" │
    └─────────────────────────────────────────┘
          ↓
  if mood !== currentMood:
    moodRef.current = mood   ← synchronous, before fetch
    setMood(mood)            ← React state update
    fetchRecommendations()   ← full fresh feed + new explanation
```

### Expression → Mood Mapping

| Model Output | Confidence Range | App Mood | Reasoning |
|---|---|---|---|
| happy | 0–1 | happy | Direct match |
| surprised | 0–1 | curious | Raised brows = engaged/curious |
| neutral | 0–1 | neutral | Direct match |
| angry | 0–1 | stressed | Tense facial muscles |
| fearful | 0–1 | stressed | Stress/anxiety |
| sad | 0–1 | tired | Low energy, downcast |
| disgusted | 0–1 | tired | Low engagement, disinterested |

### No-Face Handling

If `detectSingleFace()` returns null (no face detected), the sensor:
- Shows "No face" in the status badge (orange dot)
- Does NOT update the mood
- Tries again at the next 5-second interval

This prevents background-only frames from corrupting the mood.

---

## Models Used for Affect

### TinyFaceDetector

```
Type:       CNN (MobileNet-based)
Size:       ~190 KB
Framework:  TF.js WebGL
Input:      Full video frame (any size, internally resized)
Output:     Bounding box [x, y, width, height] + confidence score
Threshold:  scoreThreshold = 0.3 (low false-negative rate)
Speed:      ~5–10ms on integrated GPU
```

The detector internally:
1. Resizes input to `inputSize × inputSize` (224×224)
2. Runs 6-layer MobileNet backbone
3. Outputs anchor-based bounding box predictions
4. Applies NMS (Non-Maximum Suppression)
5. Returns highest-confidence face box

### FaceExpressionNet

```
Type:       CNN (MobileNet-based)  
Size:       ~310 KB
Framework:  TF.js WebGL
Input:      Face crop (extracted from bounding box), 224×224, RGB normalised
Output:     7 class probabilities (softmax)
Speed:      ~5–10ms on integrated GPU
```

Total affect inference: ~15–20ms per cycle. Zero network overhead.

---

## Consent System (GDPR Article 9)

Facial expression data is classified as **biometric data** under GDPR Article 9 — a special category requiring explicit consent.

### Consent Flow

```
User clicks "Auto Mood" (first time)
          ↓
Consent modal appears:
  "HyperNews can read your facial expression using your webcam..."
  ✓ All inference runs ON YOUR DEVICE — no images sent to server
  ✓ Only the mood label (e.g. "happy") is used
  ✓ Turn off anytime with the Camera On button
          ↓
  [Not Now]  or  [Allow Camera]
          ↓
If Allow:
  1. localStorage.setItem("hypernews_affect_consent", "true")
  2. If authenticated: POST /api/me/profile { affect_consent: true }
  3. Camera starts, AffectSensor activates
          ↓
Subsequent clicks on "Auto Mood":
  → Skips modal (consent already stored)
  → Directly toggles camera on/off
```

### Data Minimisation

```
What is processed on-device:     640×480 video frames
What leaves the device:           "happy" (string, 5–7 bytes)
What is stored in backend:        user.mood (overwritten each request)
What is never transmitted:        video frames, pixel data, face crops
```

---

## Mood Influence on the Feed

Mood affects the recommendation pipeline at three levels:

### Level 1: Category Score Multiplier (Immediate)

```python
# Applied before LTR/bandit scoring
context_score = MOOD_WEIGHTS[mood][category] * TIME_WEIGHTS[time][category]
```

Example: Stressed user at evening

```
entertainment: 1.5 × 1.3 = 1.95  ← boosted
sports:        1.3 × 1.0 = 1.30  ← boosted
politics:      0.4 × 1.0 = 0.40  ← suppressed
health:        0.5 × 1.0 = 0.50  ← suppressed
```

### Level 2: Category Avoidance (Filter)

```python
avoid_cats = {
    "stressed": ["politics", "health"],
    "tired":    ["politics", "business", "finance"],
}.get(user.mood, [])

mood_boosts = {
    "curious":  ["technology", "science", "business"],
    "happy":    ["sports", "entertainment", "lifestyle"],
    "stressed": ["entertainment", "lifestyle", "sports"],
    "tired":    ["entertainment", "lifestyle"],
}.get(user.mood, [])

# Articles in avoid_cats get -0.35 penalty
# Articles in mood_boosts get +0.45 bonus
```

### Level 3: Bandit Context Vector (Learned)

The LinUCB bandit receives `mood_val` as position 385 of the 391-d context vector. Over time it learns:

```
"When mood=stressed AND article category=entertainment → historically higher rewards"
"When mood=tired AND article is long-form science → historically lower rewards"
```

The bandit's A matrix captures the covariance between mood and reward, allowing mood-conditional personalisation that is different for each user.

### Level 4: Explanation Update

When mood changes (either manual or via affect sensor), `fetchRecommendations()` is called without `excludeIds` — a full fresh fetch. The `ExplanationBanner` in the UI receives the new explanation immediately:

```
"For your happy morning mood, showing sports and technology stories 
 based on your reading history and current upbeat context."
```

---

## Status Badge States

```
[Auto Mood]           → Camera off, consent not yet given (or given but off)
[Camera On] • Reading…→ Analyzing current frame (yellow dot)
[Camera On] • 😊 happy · happy 89%  → Detection successful (green dot)
[Camera On] • No face → No face detected in frame (orange dot)
[Camera On] • Camera denied  → getUserMedia() rejected (red)
Loading models…       → TF.js + model weights loading (purple dot)
```

The raw expression label and confidence percentage are shown alongside the mood label, so users can see exactly what the model detected.

---

## Why Face Detection Must Come Before Expression Classification

The `facial_expression_recognition_mobilefacenet_2022july.onnx` model (the original ONNX model) requires a **pre-cropped 112×112 face patch** as input. If you feed the full webcam frame (640×480 of a person sitting at a desk with background visible), the model:

1. Receives background pixels as "face" features
2. Produces high logits for "sad" consistently (it's what uniform/noisy input activates)
3. Maps "sad" → "tired" via `EXPR_TO_MOOD`
4. User always sees "tired" regardless of actual expression

`face-api.js` solves this by:
1. Running TinyFaceDetector to find the bounding box
2. Cropping and normalising the face region internally
3. Feeding the clean face crop to FaceExpressionNet

The full pipeline is: detect → crop → classify. Skipping detection makes classification useless.
