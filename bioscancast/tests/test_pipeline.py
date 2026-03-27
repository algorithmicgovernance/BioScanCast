from datetime import datetime

from bioscancast.filtering.models import ForecastQuestion, SearchResult
from bioscancast.filtering.pipeline import FilteringPipeline


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