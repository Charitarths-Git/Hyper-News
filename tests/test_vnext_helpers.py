import asyncio
import importlib
import json

import numpy as np
import pandas as pd

from backend.bio_categories import BIO_CATEGORY_ORDER
from backend.hybrid_search import _tokenize, build_hybrid_candidates, reciprocal_rank_fusion, suggest_queries
from backend.ranker import cold_start_recommendations, rank_search_articles
from backend.user_profile import UserProfile, update_user_state


def test_recent_state_caps_at_25():
    user = UserProfile(user_id="demo")
    for index in range(40):
        update_user_state(user, article_id=f"A{index}", category="tech", negative=False)
        update_user_state(user, article_id=f"N{index}", category="tech", negative=True)
        update_user_state(user, query=f"query {index}")

    assert len(user.recent_clicks) == 25
    assert user.recent_clicks[0] == "A15"
    assert len(user.recent_negative_actions) == 25
    assert user.recent_negative_actions[0] == "N15"
    assert len(user.recent_queries) == 10
    assert user.recent_queries[0] == "query 30"


def test_rrf_merges_sources_and_orders_high_consensus_first():
    ranking, scores, source_map = reciprocal_rank_fusion(
        {
            "lexical": [1, 2, 3],
            "dense": [2, 1, 4],
            "memory": [2, 5],
        },
        k=60,
        limit=5,
    )

    assert ranking[0] == 2
    assert scores[2] > scores[1]
    assert set(source_map[2]) == {"lexical", "dense", "memory"}


def test_rrf_source_weights_can_favor_query_driven_sources():
    ranking, _, _ = reciprocal_rank_fusion(
        {
            "lexical": [1, 2, 3],
            "session_memory": [2, 3, 4],
        },
        k=60,
        limit=4,
        source_weights={"lexical": 1.5, "session_memory": 0.5},
    )

    assert ranking[0] == 2
    assert ranking.index(1) < ranking.index(4)


def test_tokenize_keeps_important_short_query_tokens():
    assert "ai" in _tokenize("AI singer")


def test_suggest_queries_prefers_search_backend_results():
    class DummySearchBackend:
        def suggest(self, prefix: str, limit: int = 10):
            return ["Sports", "Spotify", "SpaceX"][:limit]

    user = UserProfile(user_id="demo", recent_queries=["sports today"])
    suggestions = suggest_queries(
        "sp",
        df=__import__("pandas").DataFrame({"category": ["Science"], "source": ["Reuters"]}),
        user=user,
        limit=5,
        search_backend=DummySearchBackend(),
    )

    assert suggestions[0] == "Sports"
    assert "sports today" in suggestions


def test_suggest_queries_can_use_recent_search_history():
    suggestions = suggest_queries(
        "",
        df=__import__("pandas").DataFrame({"category": ["Science"], "source": ["Reuters"]}),
        recent_queries=["search cats", "sports today"],
        limit=5,
        search_backend=None,
    )

    assert suggestions[0] == "sports today"


def test_build_hybrid_candidates_keeps_explicit_search_query_led():
    df = pd.DataFrame(
        {
            "news_id": ["N1", "N2", "N3"],
            "title": ["Singer wins award", "Football final preview", "Singer releases album"],
            "abstract": ["A singer tops the charts", "Sports coverage only", "New album from singer"],
            "category": ["entertainment", "sports", "music"],
            "subcategory": ["awards", "football", "albums"],
            "source": ["Reuters", "ESPN", "Billboard"],
            "entity_labels": [["Taylor Swift"], [], ["Adele"]],
            "popularity": [0.9, 0.6, 0.8],
        }
    )
    news_id_to_idx = {row.news_id: idx for idx, row in df.iterrows()}
    user = UserProfile(
        user_id="query_only_demo",
        recent_clicks=["N2"],
        recent_entities=["NFL"],
        recent_sources=["ESPN"],
    )

    candidates, diagnostics = build_hybrid_candidates(
        user,
        "singer",
        df,
        index=None,
        model=None,
        graph=None,
        news_id_to_idx=news_id_to_idx,
        reranker=None,
        search_backend=None,
        vector_backend=None,
        limit=5,
        rerank_k=3,
    )

    assert "session_memory" not in (diagnostics.get("source_counts") or {})
    assert all("session_memory" not in str(article.get("candidate_source", "")) for article in candidates)
    assert candidates
    assert candidates[0]["news_id"] in {"N1", "N3"}


def test_rank_search_articles_uses_action_history_lightly():
    df = pd.DataFrame(
        {
            "news_id": ["N1", "N2"],
            "title": ["Singer interviews on stage", "Singer sports crossover"],
            "abstract": ["Entertainment focus", "Sports angle"],
            "category": ["entertainment", "sports"],
            "subcategory": ["music", "football"],
            "source": ["Reuters", "ESPN"],
            "entity_labels": [["Taylor Swift"], []],
            "popularity": [0.5, 0.7],
        }
    )
    user = UserProfile(
        user_id="search_personalization_demo",
        interests={"entertainment": 4.0},
        recent_clicks=["N1"],
        reading_history=["N1"],
    )
    articles = [
        {
            "news_id": "N1",
            "title": "Singer interviews on stage",
            "abstract": "Entertainment focus",
            "category": "entertainment",
            "subcategory": "music",
            "source": "Reuters",
            "entity_labels": ["Taylor Swift"],
            "popularity": 0.5,
            "retrieval_score": 0.5,
            "lexical_score": 0.5,
            "dense_score": 0.2,
            "candidate_source": "lexical",
        },
        {
            "news_id": "N2",
            "title": "Singer sports crossover",
            "abstract": "Sports angle",
            "category": "sports",
            "subcategory": "football",
            "source": "ESPN",
            "entity_labels": [],
            "popularity": 0.7,
            "retrieval_score": 0.55,
            "lexical_score": 0.55,
            "dense_score": 0.25,
            "candidate_source": "dense",
        },
    ]
    embeddings = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]]).to_numpy()

    ranked = rank_search_articles(
        user,
        articles,
        embeddings,
        df,
        query_intent={
            "query": "singer",
            "normalized_query": "singer",
            "tokens": ["singer"],
            "matched_entities": [],
            "matched_categories": [],
            "matched_sources": [],
        },
        n=2,
    )

    assert ranked[0]["news_id"] == "N1"


def _reload_backend_for_temp_db(monkeypatch, tmp_path):
    db_path = tmp_path / "hypernews-test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    import backend.db as db_module
    import backend.app as app_module

    db_module = importlib.reload(db_module)
    app_module = importlib.reload(app_module)
    db_module.init_db()
    return db_module, app_module


def test_registration_defaults_onboarding_incomplete(monkeypatch, tmp_path):
    db_module, app_module = _reload_backend_for_temp_db(monkeypatch, tmp_path)

    response = asyncio.run(
        app_module.auth_register(
            app_module.RegisterRequest(
                email="new-user@example.com",
                password="password123",
                display_name="New User",
            )
        )
    )

    assert response["status"] == "registered"
    assert response["user"]["onboarding_completed"] is False

    stored = app_module.load_auth_user_by_email("new-user@example.com")
    assert stored is not None
    assert bool(stored["onboarding_completed"]) is False


def test_float_list_accepts_numpy_arrays(monkeypatch, tmp_path):
    db_module, app_module = _reload_backend_for_temp_db(monkeypatch, tmp_path)

    converted = app_module._float_list(np.array([1.0, 2.5, 3.25], dtype=np.float32))

    assert converted == [1.0, 2.5, 3.25]


def test_onboarding_completion_persists_bio_fields(monkeypatch, tmp_path):
    db_module, app_module = _reload_backend_for_temp_db(monkeypatch, tmp_path)

    registration = asyncio.run(
        app_module.auth_register(
            app_module.RegisterRequest(
                email="complete-me@example.com",
                password="password123",
                display_name="Complete Me",
            )
        )
    )
    user_id = registration["user"]["user_id"]

    response = asyncio.run(
        app_module.complete_me_onboarding(
            app_module.OnboardingRequest(
                age_bucket="25-34",
                gender="female",
                occupation="engineer",
                location_region="asia",
                location_country="India",
                interest_text="ai startups robotics",
                top_categories=["technology", "science"],
                affect_consent=True,
            ),
            user_id=user_id,
        )
    )

    assert response["status"] == "completed"
    assert response["profile"]["onboarding_completed"] is True

    stored = app_module.load_auth_user_by_id(user_id)
    assert stored is not None
    assert bool(stored["onboarding_completed"]) is True
    assert stored["location_country"] == "India"
    assert stored["interest_text"] == "ai startups robotics"
    assert json.loads(stored["top_categories_json"])[:2] == ["technology", "science"]
    assert isinstance(json.loads(stored["bio_embedding_json"]), list)
    assert isinstance(json.loads(stored["bio_text_embedding_json"]), list)


def test_profile_edits_keep_onboarding_complete(monkeypatch, tmp_path):
    db_module, app_module = _reload_backend_for_temp_db(monkeypatch, tmp_path)

    registration = asyncio.run(
        app_module.auth_register(
            app_module.RegisterRequest(
                email="edit-me@example.com",
                password="password123",
                display_name="Edit Me",
            )
        )
    )
    user_id = registration["user"]["user_id"]

    asyncio.run(
        app_module.complete_me_onboarding(
            app_module.OnboardingRequest(
                age_bucket="18-24",
                occupation="student",
                location_region="europe",
                location_country="France",
                interest_text="football transfers",
                top_categories=["sports"],
            ),
            user_id=user_id,
        )
    )

    response = asyncio.run(
        app_module.update_me_profile(
            app_module.ProfileUpdateRequest(
                display_name="Edited User",
                age_bucket="25-34",
                occupation="journalist",
                location_region="north_america",
                location_country="United States",
                interest_text="policy and media",
                top_categories=["news", "politics"],
            ),
            user_id=user_id,
        )
    )

    assert response["status"] == "updated"
    assert response["profile"]["onboarding_completed"] is True

    stored = app_module.load_auth_user_by_id(user_id)
    assert stored is not None
    assert bool(stored["onboarding_completed"]) is True
    assert stored["display_name"] == "Edited User"
    assert stored["location_country"] == "United States"


def test_cold_start_recommendations_follow_onboarding_preferences():
    df = pd.DataFrame(
        {
            "news_id": ["T1", "T2", "T3", "S1", "S2", "S3"],
            "title": [
                "AI startup raises funding",
                "Robotics lab unveils new model",
                "Chip industry expands compute capacity",
                "Football final preview",
                "Cricket captain returns to squad",
                "Tennis upset in semifinals",
            ],
            "abstract": [
                "Technology investors back artificial intelligence",
                "Science and technology crossover story",
                "Technology supply chain and semiconductors",
                "Sports coverage and match tactics",
                "Sports fans track the comeback",
                "Sports roundup from the tour",
            ],
            "category": ["technology", "technology", "technology", "sports", "sports", "sports"],
            "subcategory": ["ai", "robotics", "business", "football", "cricket", "tennis"],
            "popularity": [0.8, 0.76, 0.73, 0.8, 0.77, 0.74],
        }
    )
    embeddings = pd.DataFrame(
        [
            [1.0, 0.0],
            [0.98, 0.02],
            [0.96, 0.04],
            [0.0, 1.0],
            [0.02, 0.98],
            [0.04, 0.96],
        ]
    ).to_numpy()

    tech_bio = [0.0] * 64
    tech_bio[BIO_CATEGORY_ORDER.index("technology")] = 1.0
    sports_bio = [0.0] * 64
    sports_bio[BIO_CATEGORY_ORDER.index("sports")] = 1.0

    tech_user = UserProfile(
        user_id="tech-user",
        time_of_day="afternoon",
        top_categories=["technology"],
        interest_text="ai robotics startups",
        bio_embedding=tech_bio,
        bio_text_embedding=[1.0, 0.0],
    )
    sports_user = UserProfile(
        user_id="sports-user",
        time_of_day="afternoon",
        top_categories=["sports"],
        interest_text="football cricket sports",
        bio_embedding=sports_bio,
        bio_text_embedding=[0.0, 1.0],
    )

    tech_feed = cold_start_recommendations(tech_user, df, embeddings, n=4)
    sports_feed = cold_start_recommendations(sports_user, df, embeddings, n=4)

    tech_categories = [item["category"] for item in tech_feed[:3]]
    sports_categories = [item["category"] for item in sports_feed[:3]]

    assert tech_categories.count("technology") >= 2
    assert sports_categories.count("sports") >= 2
    assert tech_feed[0]["category"] != sports_feed[0]["category"]


def test_cold_start_recommendations_use_interest_notes_and_location():
    df = pd.DataFrame(
        {
            "news_id": ["E1", "E2", "N1", "S1"],
            "title": [
                "Thriller series dominates streaming charts",
                "Action film sequel lands global release",
                "India policy update reshapes technology imports",
                "Football preview ahead of the weekend final",
            ],
            "abstract": [
                "Entertainment story for thriller fans",
                "Cinema and action coverage for movie fans",
                "News coverage focused on India and business policy",
                "Sports coverage only",
            ],
            "category": ["entertainment", "movies", "news", "sports"],
            "subcategory": ["tv", "movies", "politics", "football"],
            "popularity": [0.75, 0.74, 0.73, 0.72],
        }
    )
    embeddings = pd.DataFrame(
        [
            [1.0, 0.0, 0.0],
            [0.98, 0.02, 0.0],
            [0.10, 0.10, 0.95],
            [0.0, 1.0, 0.0],
        ]
    ).to_numpy()

    user = UserProfile(
        user_id="notes-location-user",
        time_of_day="evening",
        top_categories=["sports", "news"],
        interest_text="thriller action sci-fi",
        location_region="asia",
        location_country="India",
    )

    feed = cold_start_recommendations(user, df, embeddings, n=3)
    top_ids = [item["news_id"] for item in feed]
    reasons = {
        item["news_id"]: item.get("reasons", [])
        for item in feed
    }

    assert "E1" in top_ids[:2] or "E2" in top_ids[:2]
    assert "N1" in top_ids
    assert any("notes" in reason.lower() or "selected" in reason.lower() for reason in reasons[top_ids[0]])
