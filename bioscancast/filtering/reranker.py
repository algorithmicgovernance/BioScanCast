from __future__ import annotations

from typing import Dict, List

from .config import FILTER_CONFIG
from .models import FilterDecision, ForecastQuestion, SearchResult
from .utils import keyword_overlap_score, weighted_sum


def lightweight_reranker_score(question: ForecastQuestion, result: SearchResult) -> float:
    """
    Placeholder cheap scorer.
    Replace later with:
    - BM25 score
    - embedding similarity
    - cross-encoder
    """
    text = f"{result.title} {result.snippet} {result.domain}"
    terms = [question.text]
    if question.pathogen:
        terms.append(question.pathogen)
    if question.region:
        terms.append(question.region)
    if question.event_type:
        terms.append(question.event_type)

    score = keyword_overlap_score(text, terms)

    if result.is_official_domain:
        score += 0.10
    if result.source_tier == "academic":
        score += 0.05

    return min(score, 1.0)


def rerank_borderline_candidates(
    borderline: List[FilterDecision],
    result_map: Dict[str, SearchResult],
    question: ForecastQuestion,
) -> List[FilterDecision]:
    reranked: list[FilterDecision] = []

    for decision in borderline:
        result = result_map[decision.result_id]
        rr_score = lightweight_reranker_score(question, result)

        updated_priority = weighted_sum(
            {
                "heuristic_priority": decision.priority_score,
                "reranker_score": rr_score,
            },
            FILTER_CONFIG["reranker_weights"],
        )

        decision.stage = "reranker"
        decision.relevance_score = max(decision.relevance_score, rr_score)
        decision.priority_score = updated_priority
        reranked.append(decision)

    reranked.sort(key=lambda d: d.priority_score, reverse=True)
    return reranked


def split_for_llm_review(
    reranked: List[FilterDecision],
) -> tuple[List[FilterDecision], List[FilterDecision]]:
    llm_needed: list[FilterDecision] = []
    llm_not_needed: list[FilterDecision] = []

    for decision in reranked:
        if decision.priority_score >= FILTER_CONFIG["auto_keep_after_rerank"]:
            decision.keep = True
            decision.reason_codes.append("auto_keep_after_rerank")
            llm_not_needed.append(decision)
        elif decision.priority_score <= FILTER_CONFIG["auto_reject_after_rerank"]:
            decision.keep = False
            decision.reason_codes.append("auto_reject_after_rerank")
            llm_not_needed.append(decision)
        else:
            llm_needed.append(decision)

    llm_needed = llm_needed[: FILTER_CONFIG["max_llm_filter_candidates"]]
    return llm_needed, llm_not_needed