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
    # keyword_overlap (topical relevance) carries the most weight so a
    # reputable-but-off-topic source can't clear the keep threshold on domain
    # authority alone (#44). domain (generic tier authority) was lowered from
    # 0.20 -> 0.10 and folded into keyword_overlap (0.40 -> 0.50); official
    # authority is preserved via official_bonus (unchanged) plus the
    # unconditional official keep in heuristics.compute_heuristic_decisions.
    "heuristic_weights": {
        "keyword_overlap": 0.50,
        "freshness": 0.20,
        "domain": 0.10,
        "official_bonus": 0.20,
    },
    # Lowered from 0.72 to 0.65 after q7/q12 live runs showed filter
    # survival of 4.7% / 13.5% — the threshold was tighter than the
    # heuristic's actual signal supports. Borderline candidates that
    # cross the new threshold still go to the LLM rescue path; this
    # change just stops dropping high-credibility-but-low-keyword-overlap
    # results pre-LLM (e.g. apnews/theguardian/washingtonpost in q7).
    # See issue #13.
    "heuristic_keep_threshold": 0.65,
    "heuristic_borderline_threshold": 0.45,

    "reranker_weights": {
        "heuristic_priority": 0.6,
        "reranker_score": 0.4,
    },
    "auto_keep_after_rerank": 0.82,
    "auto_reject_after_rerank": 0.30,
    "max_llm_filter_candidates": 10,

    # When no LLM client is configured, the ambiguous "llm_needed" band
    # (reranked priority between auto_reject and auto_keep) is normally
    # rejected outright (fail-closed). With this flag enabled — for dev /
    # offline / no-API-key runs — a borderline candidate is instead KEPT if it
    # is an official domain OR its keyword-overlap relevance clears
    # ``no_llm_fallback_relevance_threshold``. This approximates the LLM-rescue
    # path without an API call, recovering the on-topic / authoritative tail
    # without admitting the generic-news mass. Default OFF so production (which
    # always has an LLM client) is unchanged. See issue #13.
    "no_llm_soft_fallback": False,
    "no_llm_fallback_relevance_threshold": 0.5,

    "max_docs_per_domain": 2,
    "max_docs_per_type": 5,

    "near_duplicate_similarity_threshold": 0.92,
}