"""End-to-end tests for the insight pipeline.

Uses synthetic documents and scripted fake LLM responses.
No network calls, no real OpenAI imports.
"""

from bioscancast.llm.fake_client import FakeLLMClient
from bioscancast.llm.base import LLMResponse
from bioscancast.insight.pipeline import InsightPipeline, InsightRunResult
from bioscancast.insight.config import InsightConfig

from bioscancast.tests.fixtures.insight.synthetic_documents import (
    DOC_WHO_SUDAN,
    DOC_CDC_H5N1,
    DOC_REUTERS_MPOX,
    DOC_FAILED,
    QUESTION_SUDAN,
    QUESTION_H5N1,
    QUESTION_MPOX,
)
from bioscancast.tests.fixtures.insight.fake_llm_responses import (
    SUDAN_PROSE_RESPONSE,
    SUDAN_TABLE_RESPONSE,
    SUDAN_RESPONSE_CHUNK,
    EMPTY_RESPONSE,
    RISK_ASSESSMENT_RESPONSE,
    H5N1_PROSE_RESPONSE,
    H5N1_TABLE_RESPONSE,
    H5N1_HUMAN_CASES_RESPONSE,
    MPOX_PHEIC_RESPONSE,
    MPOX_DETAILS_RESPONSE,
    DUPLICATE_SUDAN_CASE_COUNT,
)


def _make_pipeline_client(responses: list[LLMResponse]) -> FakeLLMClient:
    """Create a FakeLLMClient with responses for a full pipeline run.

    The client needs responses for both embedding calls and extraction calls.
    Embedding calls use embed() which doesn't consume the response queue.
    """
    return FakeLLMClient(responses, embedding_dim=32)


# ---------------------------------------------------------------------------
# Single-document pipeline tests
# ---------------------------------------------------------------------------


def test_pipeline_single_document():
    """Pipeline should extract facts from a single document."""
    # WHO Sudan doc has 5 chunks. After hybrid retrieval, up to 5 are
    # sent for extraction. We need 5 extraction responses.
    client = _make_pipeline_client([
        SUDAN_PROSE_RESPONSE,     # chunk p1 (case count + deaths)
        SUDAN_TABLE_RESPONSE,     # chunk t2 (no facts)
        SUDAN_RESPONSE_CHUNK,     # chunk p3 (intervention)
        EMPTY_RESPONSE,           # chunk h0 (heading, no facts)
        RISK_ASSESSMENT_RESPONSE, # chunk p4 (no facts)
    ])

    config = InsightConfig(retrieval_top_k=5, max_chunks_per_document=5)
    pipeline = InsightPipeline(llm_client=client, config=config)

    result = pipeline.run(QUESTION_SUDAN, [DOC_WHO_SUDAN])

    assert isinstance(result, InsightRunResult)
    assert result.documents_processed == 1
    assert result.documents_skipped == 0
    assert len(result.records) >= 2  # At least case_count + death_count

    # Every record should have provenance
    for record in result.records:
        assert len(record.sources) >= 1
        assert record.question_id == QUESTION_SUDAN.id

    # Budget should be tracked
    assert result.budget_summary["total_input_tokens"] > 0
    assert result.budget_summary["total_output_tokens"] > 0


# ---------------------------------------------------------------------------
# Failed document skipping
# ---------------------------------------------------------------------------


def test_pipeline_skips_failed_documents():
    """Failed documents should be skipped, not cause errors."""
    # Only need responses for the successful document's chunks
    client = _make_pipeline_client([
        EMPTY_RESPONSE,  # For the one chunk that gets extracted
    ])

    config = InsightConfig(retrieval_top_k=1, max_chunks_per_document=1)
    pipeline = InsightPipeline(llm_client=client, config=config)

    # Include a failed document alongside a successful one
    result = pipeline.run(QUESTION_SUDAN, [DOC_FAILED, DOC_WHO_SUDAN])

    assert result.documents_skipped == 1
    assert result.documents_processed == 1
    assert any("Skipped" in note for note in result.notes)


# ---------------------------------------------------------------------------
# Budget tracking and early stopping
# ---------------------------------------------------------------------------


def test_pipeline_budget_tracking():
    """Every LLM call should be reflected in the budget summary."""
    client = _make_pipeline_client([
        SUDAN_PROSE_RESPONSE,
        SUDAN_TABLE_RESPONSE,
    ])

    config = InsightConfig(retrieval_top_k=2, max_chunks_per_document=2)
    pipeline = InsightPipeline(llm_client=client, config=config)

    result = pipeline.run(QUESTION_SUDAN, [DOC_WHO_SUDAN])

    summary = result.budget_summary
    assert summary["total_input_tokens"] > 0
    assert "per_model" in summary
    assert "gpt-4o-mini" in summary["per_model"]


def test_pipeline_stops_on_budget_exceeded():
    """Pipeline should stop gracefully when budget is exceeded."""
    # Set a very low budget so it triggers after the first document
    client = _make_pipeline_client([
        SUDAN_PROSE_RESPONSE,     # doc 1 chunk 1
        SUDAN_TABLE_RESPONSE,     # doc 1 chunk 2
        # No more responses needed — budget should stop before doc 2
    ])

    config = InsightConfig(
        retrieval_top_k=2,
        max_chunks_per_document=2,
        max_input_tokens_per_run=1,  # Absurdly low -> triggers immediately
    )
    pipeline = InsightPipeline(llm_client=client, config=config)

    result = pipeline.run(QUESTION_SUDAN, [DOC_WHO_SUDAN, DOC_CDC_H5N1])

    # Should have processed 0 documents (budget exceeded before first doc)
    # or at most 1 if check is after processing
    assert result.documents_processed <= 1
    assert any("Budget exceeded" in note for note in result.notes)


# ---------------------------------------------------------------------------
# Cross-document deduplication
# ---------------------------------------------------------------------------


def test_pipeline_deduplication():
    """Duplicate facts across documents should be merged."""
    from dataclasses import replace
    from copy import deepcopy

    # Create a second copy of the WHO Sudan document with a different ID
    # so that both documents contain the prose chunk with the same quote.
    doc2 = deepcopy(DOC_WHO_SUDAN)
    doc2 = replace(doc2, id="doc-who-sudan-002", result_id="r-who-sudan-002")

    # Two documents, each producing the same Sudan case count fact
    client = _make_pipeline_client([
        SUDAN_PROSE_RESPONSE,        # doc 1 -> 2 facts (case + death)
        DUPLICATE_SUDAN_CASE_COUNT,  # doc 2 -> 1 fact (duplicate case)
    ])

    config = InsightConfig(retrieval_top_k=1, max_chunks_per_document=1)
    pipeline = InsightPipeline(llm_client=client, config=config)

    result = pipeline.run(QUESTION_SUDAN, [DOC_WHO_SUDAN, doc2])

    # Find the case_count records
    case_records = [
        r for r in result.records
        if r.event_type == "case_count"
        and r.metric_name == "confirmed_cases"
    ]

    # Should be deduplicated to 1 record
    assert len(case_records) == 1

    # But with provenance from both documents (sources merged)
    assert len(case_records[0].sources) >= 2


# ---------------------------------------------------------------------------
# Multi-document end-to-end
# ---------------------------------------------------------------------------


def test_pipeline_multi_document():
    """Full pipeline run across multiple documents."""
    # Need extraction responses for each document's chunks
    # WHO Sudan: 2 chunks (retrieval_top_k=2)
    # CDC H5N1: 2 chunks
    client = _make_pipeline_client([
        # WHO Sudan
        SUDAN_PROSE_RESPONSE,
        SUDAN_TABLE_RESPONSE,
        # CDC H5N1
        H5N1_PROSE_RESPONSE,
        H5N1_TABLE_RESPONSE,
    ])

    config = InsightConfig(retrieval_top_k=2, max_chunks_per_document=2)
    pipeline = InsightPipeline(llm_client=client, config=config)

    result = pipeline.run(QUESTION_H5N1, [DOC_WHO_SUDAN, DOC_CDC_H5N1])

    assert result.documents_processed == 2
    assert result.documents_skipped == 0
    assert len(result.records) >= 1

    # Budget should include calls for both documents
    assert result.budget_summary["total_input_tokens"] > 0


# ---------------------------------------------------------------------------
# Schema validation on output records
# ---------------------------------------------------------------------------


def test_pipeline_output_records_valid():
    """Every InsightRecord should have required fields populated."""
    client = _make_pipeline_client([
        SUDAN_PROSE_RESPONSE,
        SUDAN_TABLE_RESPONSE,
    ])

    config = InsightConfig(retrieval_top_k=2, max_chunks_per_document=2)
    pipeline = InsightPipeline(llm_client=client, config=config)

    result = pipeline.run(QUESTION_SUDAN, [DOC_WHO_SUDAN])

    for record in result.records:
        assert record.id is not None
        assert record.question_id == QUESTION_SUDAN.id
        assert record.event_type in {
            "case_count", "death_count", "outbreak_declared",
            "intervention", "policy_change", "other",
        }
        assert 0.0 <= record.confidence <= 1.0
        assert len(record.sources) >= 1
        for src in record.sources:
            assert src.document_id
            assert src.chunk_id
            assert src.source_url
            assert src.quote


# ---------------------------------------------------------------------------
# Empty document list
# ---------------------------------------------------------------------------


def test_pipeline_no_documents():
    """Pipeline should handle an empty document list gracefully."""
    client = _make_pipeline_client([])
    pipeline = InsightPipeline(llm_client=client)

    result = pipeline.run(QUESTION_SUDAN, [])

    assert result.records == []
    assert result.documents_processed == 0
    assert result.documents_skipped == 0
