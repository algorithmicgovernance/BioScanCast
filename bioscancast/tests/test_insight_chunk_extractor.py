"""Tests for insight stage chunk extraction.

All tests use FakeLLMClient — no network calls, no real OpenAI imports.
"""

from bioscancast.llm.fake_client import FakeLLMClient
from bioscancast.insight.extraction.chunk_extractor import (
    extract_facts_from_chunk,
    _resolve_country_code,
    _normalize_whitespace,
)

from bioscancast.tests.fixtures.insight.synthetic_documents import (
    DOC_WHO_SUDAN,
    DOC_CDC_H5N1,
    QUESTION_SUDAN,
    QUESTION_H5N1,
)
from bioscancast.tests.fixtures.insight.fake_llm_responses import (
    SUDAN_PROSE_RESPONSE,
    HALLUCINATED_QUOTE_RESPONSE,
    H5N1_PROSE_RESPONSE,
    EMPTY_RESPONSE,
)


# ---------------------------------------------------------------------------
# Basic extraction
# ---------------------------------------------------------------------------


def test_extract_facts_from_sudan_prose():
    """Extracting from the Sudan virus prose chunk should produce valid
    InsightRecords with correct provenance."""
    client = FakeLLMClient([SUDAN_PROSE_RESPONSE])
    chunk = DOC_WHO_SUDAN.chunks[1]  # prose chunk with case count

    records, response = extract_facts_from_chunk(
        chunk, DOC_WHO_SUDAN, QUESTION_SUDAN, client, model="gpt-4o-mini"
    )

    assert len(records) == 2  # case_count + death_count

    case_record = records[0]
    assert case_record.event_type == "case_count"
    assert case_record.metric_value == 9.0
    assert case_record.question_id == QUESTION_SUDAN.id

    # Provenance
    assert len(case_record.sources) == 1
    assert case_record.sources[0].document_id == DOC_WHO_SUDAN.id
    assert case_record.sources[0].chunk_id == chunk.chunk_id
    assert case_record.sources[0].source_url == DOC_WHO_SUDAN.source_url

    # Quote should be a substring of the chunk text (hallucination guard passed)
    quote = case_record.sources[0].quote
    assert _normalize_whitespace(quote) in _normalize_whitespace(chunk.text)


def test_extract_facts_country_normalization():
    """Country names should be normalized to ISO codes."""
    client = FakeLLMClient([SUDAN_PROSE_RESPONSE])
    chunk = DOC_WHO_SUDAN.chunks[1]

    records, _ = extract_facts_from_chunk(
        chunk, DOC_WHO_SUDAN, QUESTION_SUDAN, client, model="gpt-4o-mini"
    )

    # "Mubende district, Uganda" -> "UG"
    assert records[0].iso_country_code == "UG"


def test_extract_facts_from_h5n1():
    """H5N1 extraction should produce correct metric and country code."""
    client = FakeLLMClient([H5N1_PROSE_RESPONSE])
    chunk = DOC_CDC_H5N1.chunks[0]

    records, _ = extract_facts_from_chunk(
        chunk, DOC_CDC_H5N1, QUESTION_H5N1, client, model="gpt-4o-mini"
    )

    assert len(records) == 1
    assert records[0].metric_value == 1043.0
    assert records[0].metric_name == "affected_herds"
    assert records[0].iso_country_code == "US"


def test_extract_empty_response():
    """A chunk with no relevant facts should return an empty list."""
    client = FakeLLMClient([EMPTY_RESPONSE])
    chunk = DOC_WHO_SUDAN.chunks[0]  # heading chunk

    records, response = extract_facts_from_chunk(
        chunk, DOC_WHO_SUDAN, QUESTION_SUDAN, client, model="gpt-4o-mini"
    )

    assert records == []
    assert response.input_tokens > 0  # Budget still tracked


# ---------------------------------------------------------------------------
# Hallucination guard
# ---------------------------------------------------------------------------


def test_hallucination_guard_drops_fake_quote():
    """The hallucination guard must drop facts whose quote does not
    appear in the chunk text.  This test should FAIL if the guard
    is removed."""
    client = FakeLLMClient([HALLUCINATED_QUOTE_RESPONSE])
    chunk = DOC_WHO_SUDAN.chunks[1]  # prose chunk

    records, response = extract_facts_from_chunk(
        chunk, DOC_WHO_SUDAN, QUESTION_SUDAN, client, model="gpt-4o-mini"
    )

    # The hallucinated quote "As of 20 February 2025, 50 confirmed cases
    # have been identified" does NOT appear in the chunk text.
    # The guard should drop it.
    assert len(records) == 0

    # But the LLM response was still consumed (budget tracking)
    assert response.input_tokens == 300


def test_hallucination_guard_passes_valid_quote():
    """Valid quotes should pass the hallucination guard."""
    client = FakeLLMClient([SUDAN_PROSE_RESPONSE])
    chunk = DOC_WHO_SUDAN.chunks[1]

    records, _ = extract_facts_from_chunk(
        chunk, DOC_WHO_SUDAN, QUESTION_SUDAN, client, model="gpt-4o-mini"
    )

    # Both facts have valid quotes
    assert len(records) == 2


# ---------------------------------------------------------------------------
# Country code resolution
# ---------------------------------------------------------------------------


def test_resolve_country_code_simple():
    assert _resolve_country_code("Uganda") == "UG"
    assert _resolve_country_code("United States") == "US"
    assert _resolve_country_code("usa") == "US"


def test_resolve_country_code_with_region():
    assert _resolve_country_code("Mubende district, Uganda") == "UG"
    assert _resolve_country_code("Texas") == "US"


def test_resolve_country_code_unknown():
    assert _resolve_country_code("Unknown Planet") is None
    assert _resolve_country_code(None) is None
    assert _resolve_country_code("") is None


# ---------------------------------------------------------------------------
# Token tracking
# ---------------------------------------------------------------------------


def test_response_returned_for_budget_tracking():
    """extract_facts_from_chunk must always return the LLMResponse
    so the pipeline can record it in the budget tracker."""
    client = FakeLLMClient([EMPTY_RESPONSE])
    chunk = DOC_WHO_SUDAN.chunks[0]

    _, response = extract_facts_from_chunk(
        chunk, DOC_WHO_SUDAN, QUESTION_SUDAN, client, model="gpt-4o-mini"
    )

    assert response is not None
    assert response.input_tokens == 150
    assert response.output_tokens == 15
    assert response.model == "gpt-4o-mini"
