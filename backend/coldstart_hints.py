from __future__ import annotations

import re
from collections import Counter


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]+")
_IMPORTANT_SHORT_TOKENS = {"ai", "tv", "vr", "ml", "uk", "us"}
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "about",
    "just",
    "like",
    "love",
    "want",
    "prefer",
    "follow",
    "stories",
    "story",
    "news",
}

_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "entertainment": ("thriller", "action", "drama", "anime", "cinema", "celeb", "celebrity", "showbiz", "series", "comedy", "romance", "horror"),
    "movies": ("movie", "movies", "film", "films", "cinema", "screenplay", "thriller", "action", "horror", "scifi", "sci", "fiction"),
    "tv": ("tv", "television", "series", "episode", "show", "streaming", "binge"),
    "music": ("music", "song", "songs", "album", "albums", "artist", "artists", "band", "concert", "playlist"),
    "sports": ("sports", "sport", "football", "soccer", "cricket", "tennis", "basketball", "nba", "nfl", "f1", "formula", "olympic", "badminton"),
    "technology": ("ai", "ml", "robot", "robotics", "startup", "startups", "software", "coding", "chip", "chips", "semiconductor", "gadget", "device", "tech"),
    "science": ("science", "scientific", "research", "lab", "physics", "biology", "biotech", "climate", "space", "future", "sci", "discovery"),
    "finance": ("finance", "stock", "stocks", "market", "markets", "economy", "crypto", "bitcoin", "invest", "investing", "money", "bank"),
    "politics": ("politics", "policy", "election", "government", "parliament", "president", "minister", "senate", "congress", "vote"),
    "health": ("health", "fitness", "wellness", "mental", "medicine", "medical", "disease", "nutrition", "workout"),
    "travel": ("travel", "trip", "trips", "flight", "flights", "airline", "destination", "vacation", "tourism", "hotel"),
    "food": ("food", "recipe", "recipes", "restaurant", "restaurants", "cooking", "cuisine", "chef"),
    "lifestyle": ("lifestyle", "fashion", "beauty", "home", "design", "relationship", "relationships", "style"),
    "education": ("education", "learning", "school", "college", "university", "student", "students", "teaching"),
    "games": ("game", "games", "gaming", "esports", "console", "xbox", "playstation"),
    "autos": ("auto", "autos", "car", "cars", "automobile", "automotive", "ev", "tesla"),
    "weather": ("weather", "forecast", "storm", "rain", "snow", "temperature"),
    "video": ("video", "videos", "youtube", "creator", "creators", "streamer"),
}

_OCCUPATION_CATEGORY_WEIGHTS: dict[str, dict[str, float]] = {
    "student": {"education": 1.0, "science": 0.5, "technology": 0.35},
    "engineer": {"technology": 1.0, "science": 0.7, "finance": 0.25},
    "teacher": {"education": 1.0, "science": 0.45, "news": 0.2},
    "doctor": {"health": 1.0, "science": 0.8},
    "lawyer": {"politics": 0.85, "news": 0.55, "finance": 0.2},
    "journalist": {"news": 1.0, "politics": 0.65, "entertainment": 0.2},
    "artist": {"entertainment": 0.8, "fashion": 0.55, "music": 0.45},
    "finance": {"finance": 1.0, "news": 0.45, "technology": 0.2},
    "government": {"politics": 1.0, "news": 0.7, "finance": 0.2},
    "retail": {"finance": 0.55, "lifestyle": 0.35, "news": 0.2},
}

_REGION_TERMS: dict[str, tuple[str, ...]] = {
    "north_america": ("north america", "american", "canada", "canadian", "usa", "u.s.", "us"),
    "europe": ("europe", "european", "uk", "britain", "british", "france", "germany", "italy", "spain"),
    "asia": ("asia", "asian", "india", "indian", "china", "chinese", "japan", "japanese", "korea", "singapore"),
    "latin_america": ("latin america", "latam", "mexico", "mexican", "brazil", "brazilian", "argentina"),
    "africa": ("africa", "african", "nigeria", "kenya", "south africa"),
    "oceania": ("oceania", "australia", "australian", "new zealand"),
}

_COUNTRY_DEMONYMS: dict[str, tuple[str, ...]] = {
    "india": ("india", "indian"),
    "united states": ("united states", "u.s.", "usa", "us", "american"),
    "usa": ("usa", "us", "u.s.", "american", "united states"),
    "uk": ("uk", "british", "britain", "united kingdom"),
    "united kingdom": ("united kingdom", "uk", "britain", "british"),
    "china": ("china", "chinese"),
    "japan": ("japan", "japanese"),
    "france": ("france", "french"),
    "germany": ("germany", "german"),
    "italy": ("italy", "italian"),
    "spain": ("spain", "spanish"),
    "brazil": ("brazil", "brazilian"),
    "mexico": ("mexico", "mexican"),
    "canada": ("canada", "canadian"),
    "australia": ("australia", "australian"),
    "korea": ("korea", "korean", "south korea"),
}

_REGION_LABELS = {
    "north_america": "North America",
    "europe": "Europe",
    "asia": "Asia",
    "latin_america": "Latin America",
    "africa": "Africa",
    "oceania": "Oceania",
}


def _normalize_text(value: str) -> str:
    return str(value or "").strip().lower()


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _normalize_text(value))


def extract_interest_terms(text: str, limit: int = 16) -> list[str]:
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(_normalize_text(text)):
        normalized = raw.strip("-")
        compact = _normalize_token(normalized)
        if not compact:
            continue
        if compact in _STOPWORDS:
            continue
        if len(compact) <= 2 and compact not in _IMPORTANT_SHORT_TOKENS:
            continue
        if compact in tokens:
            continue
        tokens.append(compact)
        if len(tokens) >= limit:
            break
    return tokens


def _token_matches_keyword(token: str, keyword: str) -> bool:
    normalized_token = _normalize_token(token)
    normalized_keyword = _normalize_token(keyword)
    if not normalized_token or not normalized_keyword:
        return False
    if normalized_token == normalized_keyword:
        return True
    if len(normalized_keyword) >= 3 and normalized_token.startswith(normalized_keyword):
        return True
    if len(normalized_token) >= 4 and normalized_keyword.startswith(normalized_token):
        return True
    if len(normalized_keyword) >= 4 and normalized_keyword in normalized_token:
        return True
    return False


def infer_interest_category_weights(text: str, limit: int = 6) -> dict[str, float]:
    tokens = extract_interest_terms(text)
    scores: Counter[str] = Counter()
    for token in tokens:
        for category, keywords in _CATEGORY_KEYWORDS.items():
            matched = False
            for keyword in keywords:
                if _token_matches_keyword(token, keyword):
                    matched = True
                    break
            if matched:
                scores[category] += 1.0

    if not scores:
        return {}

    top_categories = scores.most_common(max(int(limit), 0))
    peak = float(top_categories[0][1]) if top_categories else 1.0
    return {
        category: float(score / max(peak, 1.0))
        for category, score in top_categories
        if float(score) > 0.0
    }


def infer_occupation_category_weights(occupation: str) -> dict[str, float]:
    normalized = _normalize_text(occupation)
    return dict(_OCCUPATION_CATEGORY_WEIGHTS.get(normalized, {}))


def location_terms(location_region: str = "", location_country: str = "") -> list[str]:
    terms: list[str] = []
    normalized_region = _normalize_text(location_region)
    normalized_country = _normalize_text(location_country)

    for value in _REGION_TERMS.get(normalized_region, ()):
        normalized = _normalize_text(value)
        if normalized and normalized not in terms:
            terms.append(normalized)

    if normalized_country:
        for value in _COUNTRY_DEMONYMS.get(normalized_country, (normalized_country,)):
            normalized = _normalize_text(value)
            if normalized and normalized not in terms:
                terms.append(normalized)

        for part in re.split(r"[\s,;/\-]+", normalized_country):
            normalized = _normalize_text(part)
            if len(normalized) >= 3 and normalized not in terms:
                terms.append(normalized)

    return terms[:10]


def humanize_location(location_region: str = "", location_country: str = "") -> str:
    country = str(location_country or "").strip()
    if country:
        return country
    region_key = _normalize_text(location_region)
    return _REGION_LABELS.get(region_key, "")


def has_location_preference(location_region: str = "", location_country: str = "") -> bool:
    return bool(str(location_region or "").strip() or str(location_country or "").strip())
