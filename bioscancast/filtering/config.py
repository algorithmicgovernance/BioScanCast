FILTER_CONFIG = {
    "blocked_domains": {
        "facebook.com",
        "instagram.com",
        "tiktok.com",
        "pinterest.com",
    },
    "low_value_url_keywords": {
        "/login",
        "/signup",
        "/register",
        "/account",
        "/privacy",
        "/terms",
        "/contact",
        "/about",
        "/careers",
        "/advertise",
    },
    "low_value_title_keywords": {
        "sign in",
        "login",
        "register",
        "cookie policy",
        "privacy policy",
        "terms of use",
    },
    "source_tier_scores": {
        "official": 1.0,
        "academic": 0.9,
        "trusted_media": 0.65,
        "ngo": 0.6,
        "unknown": 0.35,
    },
    "heuristic_weights": {
        "keyword_overlap": 0.40,
        "freshness": 0.20,
        "domain": 0.20,
        "official_bonus": 0.20,
    },
    "heuristic_keep_threshold": 0.72,
    "heuristic_borderline_threshold": 0.45,

    "reranker_weights": {
        "heuristic_priority": 0.6,
        "reranker_score": 0.4,
    },
    "auto_keep_after_rerank": 0.82,
    "auto_reject_after_rerank": 0.30,
    "max_llm_filter_candidates": 10,

    "max_docs_per_domain": 2,
    "max_docs_per_type": 5,

    "near_duplicate_similarity_threshold": 0.92,
}