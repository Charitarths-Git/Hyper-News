BIO_CATEGORY_ORDER = [
    "sports",
    "news",
    "finance",
    "entertainment",
    "technology",
    "lifestyle",
    "health",
    "travel",
    "food",
    "politics",
    "science",
    "education",
    "fashion",
    "autos",
    "weather",
    "games",
    "movies",
    "music",
    "tv",
    "video",
]


def normalize_bio_category_key(value: str) -> str:
    return str(value or "").strip().lower()
