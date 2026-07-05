from datetime import datetime

import pytest

from bioscancast.stages.filtering.heuristics import (
    compute_priority_score,
    heuristic_filter,
    is_low_value_page,
)
from bioscancast.stages.filtering.models import ForecastQuestion, SearchResult


def _mk_result(**overrides) -> SearchResult:
    base = dict(
        id="1",
        question_id="q1",
        query_id="s1",
        engine="google",
        url="https://example.com/article",
        canonical_url=None,
        domain="example.com",
        title="Some headline",
        snippet="Some body text",
        rank=1,
        retrieved_at=datetime.utcnow(),
    )
    base.update(overrides)
    return SearchResult(**base)


def test_low_value_page_detected():
    result = _mk_result(url="https://example.com/login", title="Login", snippet="Sign in here")
    assert is_low_value_page(result) is True


# ---------------------------------------------------------------------------
# Relevance-vs-credibility keep decision (#44)
# ---------------------------------------------------------------------------


def test_priority_credibility_weight_is_low():
    """The credibility blend is 0.15, not the old 0.25 — two results identical
    except credibility differ by exactly 0.15 * delta (#44)."""
    result = _mk_result(freshness_score=0.5, domain_score=0.5, is_official_domain=False)
    low = compute_priority_score(result, relevance_score=0.4, credibility_score=0.2)
    high = compute_priority_score(result, relevance_score=0.4, credibility_score=0.8)
    assert high - low == pytest.approx(0.15 * (0.8 - 0.2))


def test_topical_relevance_outranks_generic_authority():
    """A non-official on-topic source outscores a non-official off-topic one
    that only has domain/credibility authority (#44). Under the old weights
    (keyword_overlap 0.40, domain 0.20, credibility 0.25) the ranking was the
    other way around."""
    on_topic = _mk_result(freshness_score=0.5, domain_score=0.35, is_official_domain=False)
    off_topic_authority = _mk_result(
        freshness_score=0.5, domain_score=0.9, is_official_domain=False
    )
    p_on_topic = compute_priority_score(
        on_topic, relevance_score=0.8, credibility_score=0.35
    )
    p_authority = compute_priority_score(
        off_topic_authority, relevance_score=0.2, credibility_score=0.9
    )
    assert p_on_topic > p_authority


def test_official_source_kept_even_when_priority_below_threshold():
    """Official-source recall stays 1.0: an official document with no topical
    overlap and no freshness (priority well below the keep threshold) is still
    kept, tagged ``official_recall_guarantee`` (#44 guardrail / #13)."""
    question = ForecastQuestion(
        id="q1",
        text="How many cases of pathogen Y in country X?",
        created_at=datetime.utcnow(),
        pathogen="pathogen Y",
        region="country X",
    )
    official = _mk_result(
        url="https://who.int/emergencies/don-456",
        domain="who.int",
        title="Weekly bulletin",
        snippet="Routine administrative note.",
        source_tier="official",
        is_official_domain=True,
        domain_score=0.0,
        freshness_score=0.0,
    )
    keep_list, borderline_list, reject_list = heuristic_filter([official], question)
    kept_ids = {d.result_id for d in keep_list}
    assert official.id in kept_ids
    kept = next(d for d in keep_list if d.result_id == official.id)
    assert "official_recall_guarantee" in kept.reason_codes
    assert kept.priority_score < 0.65  # kept despite being below threshold


def test_dashboard_lookup_survives_low_value_url():
    """A curated dashboard whose URL trips the low-value screen (an ``/about``
    path like GPEI's ``about-polio/polio-this-week``) must still be kept: the
    ``dashboard_lookup`` bypass runs *before* the low-value-page check, so
    injected sources are never dropped by URL-shape heuristics."""
    question = ForecastQuestion(
        id="q1",
        text="How many wild poliovirus type 1 cases in 2026?",
        created_at=datetime.utcnow(),
        pathogen="poliovirus",
    )
    dashboard = _mk_result(
        url="https://polioeradication.org/about-polio/polio-this-week/",
        domain="polioeradication.org",
        retrieval_reason="dashboard_lookup",
    )
    # Sanity: on its own this URL would be dropped by the low-value screen.
    assert is_low_value_page(dashboard) is True

    keep_list, _borderline, reject_list = heuristic_filter([dashboard], question)
    assert dashboard.id in {d.result_id for d in keep_list}
    assert dashboard.id not in {d.result_id for d in reject_list}
    kept = next(d for d in keep_list if d.result_id == dashboard.id)
    assert "dashboard_lookup_bypass" in kept.reason_codes