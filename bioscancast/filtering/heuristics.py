from __future__ import annotations

from typing import List, Tuple

from .config import FILTER_CONFIG
from .models import FilterDecision, ForecastQuestion, SearchResult
from .utils import keyword_overlap_score, normalize_text, weighted_sum


def build_query_terms(question: ForecastQuestion) -> list[str]:
    terms = [question.text]
    if question.pathogen:
        terms.append(question.pathogen)
    if question.region:
        terms.append(question.region)
    if question.event_type:
        terms.append(question.event_type)
    if question.resolution_criteria:
        terms.append(question.resolution_criteria)
    return terms


def is_low_value_page(result: SearchResult) -> bool:
    url_lower = result.url.lower()
    title_lower = result.title.lower()

    if result.domain.lower() in FILTER_CONFIG["blocked_domains"]:
        return True

    if any(key in url_lower for key in FILTER_CONFIG["low_value_url_keywords"]):
        return True

    if any(key in title_lower for key in FILTER_CONFIG["low_value_title_keywords"]):
        return True

    if len((result.title or "").strip()) == 0 and len((result.snippet or "").strip()) == 0:
        return True

    return False


def compute_heuristic_relevance(result: SearchResult, question: ForecastQuestion) -> float:
    terms = build_query_terms(question)
    text = f"{result.title} {result.snippet} {result.domain}"
    return keyword_overlap_score(text, terms)


def compute_heuristic_credibility(result: SearchResult) -> float:
    base = FILTER_CONFIG["source_tier_scores"].get(result.source_tier, 0.35)

    if result.is_official_domain:
        base = max(base, 0.95)

    if result.file_type == "pdf":
        base += 0.03

    return min(base, 1.0)


def compute_priority_score(
    result: SearchResult,
    relevance_score: float,
    credibility_score: float,
) -> float:
    score = weighted_sum(
        {
            "keyword_overlap": relevance_score,
            "freshness": result.freshness_score,
            "domain": result.domain_score,
            "official_bonus": 1.0 if result.is_official_domain else 0.0,
        },
        FILTER_CONFIG["heuristic_weights"],
    )

    # Blend in credibility as an additional stabilizer.
    return 0.75 * score + 0.25 * credibility_score


def make_decision(
    result: SearchResult,
    keep: bool | None,
    stage: str,
    relevance_score: float,
    credibility_score: float,
    priority_score: float,
    reason_codes: list[str],
    notes: str | None = None,
) -> FilterDecision:
    return FilterDecision(
        result_id=result.id,
        keep=keep,
        stage=stage,
        relevance_score=relevance_score,
        credibility_score=credibility_score,
        priority_score=priority_score,
        reason_codes=reason_codes,
        notes=notes,
    )


def heuristic_filter(
    search_results: List[SearchResult],
    question: ForecastQuestion,
) -> Tuple[List[FilterDecision], List[FilterDecision], List[FilterDecision]]:
    keep_list: list[FilterDecision] = []
    borderline_list: list[FilterDecision] = []
    reject_list: list[FilterDecision] = []

    for result in search_results:
        if is_low_value_page(result):
            reject_list.append(
                make_decision(
                    result=result,
                    keep=False,
                    stage="heuristic",
                    relevance_score=0.0,
                    credibility_score=0.0,
                    priority_score=0.0,
                    reason_codes=["low_value_page"],
                )
            )
            continue

        relevance_score = compute_heuristic_relevance(result, question)
        credibility_score = compute_heuristic_credibility(result)
        priority_score = compute_priority_score(result, relevance_score, credibility_score)

        reason_codes = []
        if result.is_official_domain:
            reason_codes.append("official_domain")
        if result.source_tier == "academic":
            reason_codes.append("academic_source")
        if result.published_date is not None:
            reason_codes.append("has_publication_date")
        if result.file_type == "pdf":
            reason_codes.append("pdf_candidate")

        if priority_score >= FILTER_CONFIG["heuristic_keep_threshold"]:
            keep_list.append(
                make_decision(
                    result=result,
                    keep=True,
                    stage="heuristic",
                    relevance_score=relevance_score,
                    credibility_score=credibility_score,
                    priority_score=priority_score,
                    reason_codes=reason_codes + ["passed_heuristic_threshold"],
                )
            )
        elif priority_score >= FILTER_CONFIG["heuristic_borderline_threshold"]:
            borderline_list.append(
                make_decision(
                    result=result,
                    keep=None,
                    stage="heuristic",
                    relevance_score=relevance_score,
                    credibility_score=credibility_score,
                    priority_score=priority_score,
                    reason_codes=reason_codes + ["borderline_candidate"],
                )
            )
        else:
            reject_list.append(
                make_decision(
                    result=result,
                    keep=False,
                    stage="heuristic",
                    relevance_score=relevance_score,
                    credibility_score=credibility_score,
                    priority_score=priority_score,
                    reason_codes=reason_codes + ["below_threshold"],
                )
            )

    return keep_list, borderline_list, reject_list