from datetime import datetime

from bioscancast.stages.filtering.config import FILTER_CONFIG
from bioscancast.stages.filtering.models import ForecastQuestion, SearchResult
from bioscancast.stages.filtering.pipeline import FilteringPipeline


def test_pipeline_keeps_official_result():
    question = ForecastQuestion(
        id="q1",
        text="Will there be more than 50 cases?",
        created_at=datetime.utcnow(),
        pathogen="pathogen Y",
        region="country X",
    )

    result = SearchResult(
        id="r1",
        question_id="q1",
        query_id="sq1",
        engine="google",
        url="https://who.int/update",
        canonical_url="https://who.int/update",
        domain="who.int",
        title="Country X pathogen Y outbreak update",
        snippet="Confirmed human cases reported.",
        rank=1,
        retrieved_at=datetime.utcnow(),
        source_tier="official",
        is_official_domain=True,
        domain_score=1.0,
        freshness_score=1.0,
        search_stage_score=0.95,
    )

    pipeline = FilteringPipeline(llm_client=None)
    docs = pipeline.run(question, [result])

    assert len(docs) == 1
    assert docs[0].domain == "who.int"


def _borderline_question():
    return ForecastQuestion(
        id="q1",
        text="How many confirmed Ebola cases in the DRC outbreak?",
        created_at=datetime(2026, 5, 1),
        pathogen="ebola",
        region="DRC",
    )


def _borderline_result():
    # trusted_media (domain_score 0.6, non-official) with partial term overlap →
    # lands in the heuristic borderline band, then the no-LLM "llm_needed" band.
    return SearchResult(
        id="r-border",
        question_id="q1",
        query_id="sq1",
        engine="google",
        url="https://www.cnn.com/ebola-drc",
        canonical_url="https://www.cnn.com/ebola-drc",
        domain="cnn.com",
        title="Ebola cases climb in the DRC outbreak",
        snippet="Confirmed Ebola cases reported in the Democratic Republic of the Congo outbreak.",
        rank=2,
        retrieved_at=datetime(2026, 5, 1),
        source_tier="trusted_media",
        is_official_domain=False,
        domain_score=0.6,
        freshness_score=1.0,
        search_stage_score=0.6,
    )


def test_no_llm_soft_fallback_flag_changes_borderline_outcome():
    question = _borderline_question()
    result = _borderline_result()

    saved = dict(FILTER_CONFIG)
    try:
        # Flag OFF (default): fail closed → borderline candidate dropped.
        FILTER_CONFIG["no_llm_soft_fallback"] = False
        docs_off = FilteringPipeline(llm_client=None).run(question, [result])

        # Flag ON: relevant borderline candidate kept without an LLM call.
        FILTER_CONFIG["no_llm_soft_fallback"] = True
        FILTER_CONFIG["no_llm_fallback_relevance_threshold"] = 0.0
        docs_on = FilteringPipeline(llm_client=None).run(question, [result])
    finally:
        FILTER_CONFIG.clear()
        FILTER_CONFIG.update(saved)

    assert {d.result_id for d in docs_off} == set()
    assert {d.result_id for d in docs_on} == {"r-border"}