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
# Date-precision-aware dedup
# ---------------------------------------------------------------------------


def _record(
    record_id: str,
    *,
    event_date: "datetime | None" = None,
    event_date_precision: "str | None" = None,
    confidence: float = 0.8,
    location: "str | None" = "United States",
    metric_name: "str | None" = "confirmed_cases",
    event_type: str = "case_count",
    document_id: str = "doc-test",
):
    """Build an InsightRecord with a single ChunkReference for dedup tests."""
    from bioscancast.schemas import InsightRecord, ChunkReference
    return InsightRecord(
        id=record_id,
        question_id="q-test",
        event_type=event_type,
        confidence=confidence,
        location=location,
        metric_name=metric_name,
        metric_value=1.0,
        event_date=event_date,
        event_date_precision=event_date_precision,
        sources=[ChunkReference(
            document_id=document_id,
            chunk_id=f"chunk-{record_id}",
            source_url=f"https://example.com/{document_id}",
            quote="some quote",
        )],
    )


def test_dedup_merges_day_precision_with_month_precision_in_same_month():
    """A day-precision fact (2026-01-25) should merge with a month-precision
    fact (2026-01) and the merged record should keep the finer precision."""
    from datetime import datetime
    from bioscancast.insight.pipeline import _deduplicate_records

    day_record = _record(
        "day", event_date=datetime(2026, 1, 25), event_date_precision="day",
    )
    month_record = _record(
        "month", event_date=datetime(2026, 1, 1), event_date_precision="month",
        document_id="doc-other",
    )
    result = _deduplicate_records([day_record, month_record])
    assert len(result) == 1
    merged = result[0]
    assert merged.event_date_precision == "day"
    assert merged.event_date == datetime(2026, 1, 25)
    assert len(merged.sources) == 2


def test_dedup_keeps_month_precision_when_order_reversed():
    """Order of presentation must not change the outcome: month-then-day
    must also merge to the day-precision form."""
    from datetime import datetime
    from bioscancast.insight.pipeline import _deduplicate_records

    month_record = _record(
        "month", event_date=datetime(2026, 1, 1), event_date_precision="month",
    )
    day_record = _record(
        "day", event_date=datetime(2026, 1, 25), event_date_precision="day",
        document_id="doc-other",
    )
    result = _deduplicate_records([month_record, day_record])
    assert len(result) == 1
    merged = result[0]
    assert merged.event_date_precision == "day"
    assert merged.event_date == datetime(2026, 1, 25)


def test_dedup_does_not_merge_different_months_at_month_precision():
    """Two month-precision records in different months must stay separate."""
    from datetime import datetime
    from bioscancast.insight.pipeline import _deduplicate_records

    jan = _record(
        "jan", event_date=datetime(2026, 1, 1), event_date_precision="month",
    )
    feb = _record(
        "feb", event_date=datetime(2026, 2, 1), event_date_precision="month",
        document_id="doc-other",
    )
    result = _deduplicate_records([jan, feb])
    assert len(result) == 2


def test_dedup_does_not_merge_different_days():
    """Two day-precision records on different days must stay separate even
    when they share the same month."""
    from datetime import datetime
    from bioscancast.insight.pipeline import _deduplicate_records

    day5 = _record(
        "day5", event_date=datetime(2026, 1, 5), event_date_precision="day",
    )
    day6 = _record(
        "day6", event_date=datetime(2026, 1, 6), event_date_precision="day",
        document_id="doc-other",
    )
    result = _deduplicate_records([day5, day6])
    assert len(result) == 2


def test_dedup_merges_year_precision_with_day_precision_in_same_year():
    """A year-only fact (2026) should merge with a day-precision fact
    inside that year."""
    from datetime import datetime
    from bioscancast.insight.pipeline import _deduplicate_records

    year = _record(
        "year", event_date=datetime(2026, 1, 1), event_date_precision="year",
    )
    day = _record(
        "day", event_date=datetime(2026, 3, 15), event_date_precision="day",
        document_id="doc-other",
    )
    result = _deduplicate_records([year, day])
    assert len(result) == 1
    assert result[0].event_date_precision == "day"
    assert result[0].event_date == datetime(2026, 3, 15)


def test_dedup_does_not_merge_different_years():
    """Year-only facts in different years must stay separate."""
    from datetime import datetime
    from bioscancast.insight.pipeline import _deduplicate_records

    y2025 = _record(
        "2025", event_date=datetime(2025, 1, 1), event_date_precision="year",
    )
    y2026 = _record(
        "2026", event_date=datetime(2026, 1, 1), event_date_precision="year",
        document_id="doc-other",
    )
    result = _deduplicate_records([y2025, y2026])
    assert len(result) == 2


def test_dedup_three_way_merge_with_mixed_precisions():
    """Three records, one each at year/month/day precision, all in the
    same time range, should collapse to one with day precision."""
    from datetime import datetime
    from bioscancast.insight.pipeline import _deduplicate_records

    year = _record(
        "year", event_date=datetime(2026, 1, 1), event_date_precision="year",
        document_id="doc-a",
    )
    month = _record(
        "month", event_date=datetime(2026, 1, 1), event_date_precision="month",
        document_id="doc-b",
    )
    day = _record(
        "day", event_date=datetime(2026, 1, 25), event_date_precision="day",
        document_id="doc-c",
    )
    result = _deduplicate_records([year, month, day])
    assert len(result) == 1
    merged = result[0]
    assert merged.event_date_precision == "day"
    assert merged.event_date == datetime(2026, 1, 25)
    assert len(merged.sources) == 3


def test_dedup_does_not_merge_dated_with_undated():
    """A record with no date and a record with a date should stay
    separate — we don't claim they're about the same event."""
    from datetime import datetime
    from bioscancast.insight.pipeline import _deduplicate_records

    undated = _record("undated")
    dated = _record(
        "dated", event_date=datetime(2026, 1, 25), event_date_precision="day",
        document_id="doc-other",
    )
    result = _deduplicate_records([undated, dated])
    assert len(result) == 2


def test_dedup_merges_undated_records():
    """Two records with no date in the same group still merge."""
    from bioscancast.insight.pipeline import _deduplicate_records

    a = _record("a")
    b = _record("b", document_id="doc-other")
    result = _deduplicate_records([a, b])
    assert len(result) == 1
    assert len(result[0].sources) == 2


def test_dedup_does_not_merge_across_different_locations():
    """Different normalized locations stay in separate groups."""
    from datetime import datetime
    from bioscancast.insight.pipeline import _deduplicate_records

    us = _record(
        "us", location="United States",
        event_date=datetime(2026, 1, 25), event_date_precision="day",
    )
    uk = _record(
        "uk", location="United Kingdom", document_id="doc-other",
        event_date=datetime(2026, 1, 25), event_date_precision="day",
    )
    result = _deduplicate_records([us, uk])
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Value-aware dedup (catches model location-attribution errors)
# ---------------------------------------------------------------------------


def _record_with_value(
    record_id: str,
    metric_value: float,
    *,
    event_date=None,
    event_date_precision=None,
    document_id: str = "doc-test",
):
    """Build a record with a specific metric_value for value-conflict tests."""
    from bioscancast.schemas import InsightRecord, ChunkReference
    from datetime import datetime
    if event_date is None:
        event_date = datetime(2026, 1, 1)
    return InsightRecord(
        id=record_id,
        question_id="q-test",
        event_type="case_count",
        confidence=0.8,
        location="Democratic Republic of the Congo",
        metric_name="confirmed_cases",
        metric_value=metric_value,
        event_date=event_date,
        event_date_precision=event_date_precision or "month",
        sources=[ChunkReference(
            document_id=document_id,
            chunk_id=f"chunk-{record_id}",
            source_url=f"https://example.com/{document_id}",
            quote="some quote",
        )],
    )


def test_dedup_does_not_merge_when_metric_values_differ_significantly():
    """Two records sharing event_type, metric_name, location and date
    bucket should still NOT merge when their metric_values disagree —
    this is almost always a model location-attribution error (e.g.
    regional 9782 cases mistakenly labelled with the DRC's location).
    Live tests on the WHO cholera doc exposed this exact case."""
    from bioscancast.insight.pipeline import _deduplicate_records

    drc_real = _record_with_value("drc", 6543.0)
    africa_misattributed = _record_with_value(
        "africa_misattributed", 9782.0, document_id="doc-other",
    )
    result = _deduplicate_records([drc_real, africa_misattributed])
    assert len(result) == 2, (
        f"Expected separate records for different values, got {len(result)}"
    )


def test_dedup_merges_when_metric_values_are_close():
    """Rounding differences within 1% should still merge — one source
    rounding to 6500 while another reports 6543 doesn't mean they're
    different facts."""
    from bioscancast.insight.pipeline import _deduplicate_records

    exact = _record_with_value("exact", 6543.0)
    rounded = _record_with_value(
        "rounded", 6540.0, document_id="doc-other",  # within 1%
    )
    result = _deduplicate_records([exact, rounded])
    assert len(result) == 1


def test_dedup_merges_when_one_value_is_none():
    """When one record has no metric_value (qualitative claim) and
    another has a value, they should still merge if dedup keys and
    dates align — the value-aware check treats None as compatible."""
    from bioscancast.insight.pipeline import _deduplicate_records

    valued = _record_with_value("valued", 6543.0)
    no_value = _record_with_value("no_value", None, document_id="doc-other")
    result = _deduplicate_records([valued, no_value])
    assert len(result) == 1
    # The merged record retains the actual value
    assert result[0].metric_value == 6543.0


# ---------------------------------------------------------------------------
# Per-chunk parallelism (ThreadPoolExecutor)
# ---------------------------------------------------------------------------


def _content_keyed_response(chunk_text_marker: str, quote: str) -> LLMResponse:
    """Build an LLMResponse whose `quote` matches a chunk that contains
    `chunk_text_marker`. Used to test that parallel chunk extraction
    pairs responses with the right chunks regardless of which worker
    processes which chunk first.
    """
    return LLMResponse(
        content={"facts": [{
            "event_type": "case_count",
            "confidence": 0.8,
            "location": None,
            "pathogen": None,
            "metric_name": "marker",
            "metric_value": 1.0,
            "metric_unit": "events",
            "event_date": None,
            "summary": chunk_text_marker,
            "quote": quote,
        }]},
        input_tokens=100,
        output_tokens=20,
        model="gpt-4o-mini",
        raw_text='{"facts": [...]}',
    )


class _ContentKeyedFakeLLM:
    """Smart fake that returns a different response per chunk based on
    which chunk text appears in the user prompt. Order-independent — so
    concurrent worker threads can hit any chunk in any order and still
    get the right quote-matching response.
    """

    def __init__(self) -> None:
        import threading
        self._lock = threading.Lock()
        self.calls = 0

    def generate_json(self, *, system, user, schema, model, max_tokens=1024):
        import re as _re
        with self._lock:
            self.calls += 1
        # Find the chunk text inside the prompt
        marker_match = _re.search(r"CHUNK TEXT:\n(.+?)$", user, _re.DOTALL)
        if not marker_match:
            return LLMResponse(
                content={"facts": []}, input_tokens=50, output_tokens=5,
                model=model, raw_text="{}",
            )
        chunk_text = marker_match.group(1)
        # Pull the first sentence as a verbatim quote
        sentence = _re.search(r"[A-Z][^.\n]{20,150}\.", chunk_text)
        if not sentence:
            return LLMResponse(
                content={"facts": []}, input_tokens=50, output_tokens=5,
                model=model, raw_text="{}",
            )
        quote = sentence.group(0)
        return LLMResponse(
            content={"facts": [{
                "event_type": "case_count",
                "confidence": 0.7,
                "location": None,
                "pathogen": None,
                "metric_name": "concurrent_marker",
                "metric_value": 1.0,
                "metric_unit": "events",
                "event_date": None,
                "summary": None,
                "quote": quote,
            }]},
            input_tokens=120, output_tokens=25,
            model=model, raw_text='{"facts": [...]}',
        )

    def embed(self, texts, *, model):
        # Reuse FakeLLMClient's hash-based embeddings
        return FakeLLMClient(embedding_dim=32).embed(texts, model=model)


def test_pipeline_parallel_chunk_extraction_produces_all_records():
    """With chunk_workers > 1, every retrieved chunk should still be
    processed and every accepted quote should produce a record. Uses a
    content-keyed fake so response/chunk pairing is order-independent."""
    fake = _ContentKeyedFakeLLM()
    config = InsightConfig(
        retrieval_top_k=4,
        max_chunks_per_document=4,
        chunk_workers=4,
    )
    pipeline = InsightPipeline(llm_client=fake, config=config)
    result = pipeline.run(QUESTION_SUDAN, [DOC_WHO_SUDAN])

    # Every chunk in the top-k should have been called
    assert fake.calls == 4
    # The pipeline-level dedup may merge records, but at least the
    # ones whose chunks contain a quotable sentence should produce one
    # record each (mod dedup).
    assert len(result.records) >= 1
    # All records must have valid provenance
    for rec in result.records:
        assert rec.sources
        for s in rec.sources:
            assert s.quote


def test_pipeline_sequential_and_parallel_produce_same_record_count():
    """chunk_workers=1 and chunk_workers=4 must produce the same number
    of records when the fake LLM is content-keyed (so result depends on
    chunk content, not worker order)."""
    config_seq = InsightConfig(
        retrieval_top_k=4, max_chunks_per_document=4, chunk_workers=1,
    )
    config_par = InsightConfig(
        retrieval_top_k=4, max_chunks_per_document=4, chunk_workers=4,
    )

    seq_pipeline = InsightPipeline(
        llm_client=_ContentKeyedFakeLLM(), config=config_seq,
    )
    par_pipeline = InsightPipeline(
        llm_client=_ContentKeyedFakeLLM(), config=config_par,
    )

    seq_result = seq_pipeline.run(QUESTION_SUDAN, [DOC_WHO_SUDAN])
    par_result = par_pipeline.run(QUESTION_SUDAN, [DOC_WHO_SUDAN])

    assert len(seq_result.records) == len(par_result.records)
    assert (
        seq_result.budget_summary["total_input_tokens"]
        == par_result.budget_summary["total_input_tokens"]
    )


def test_pipeline_parallel_isolates_chunk_failures():
    """If extract_facts_from_chunk raises on one chunk, the others
    should still complete and the doc shouldn't blow up."""
    import threading

    class _IntermittentFake:
        def __init__(self) -> None:
            self._lock = threading.Lock()
            self.calls = 0

        def generate_json(self, *, system, user, schema, model, max_tokens=1024):
            with self._lock:
                self.calls += 1
                n = self.calls
            if n == 2:
                raise RuntimeError("simulated failure on second chunk")
            # Return a benign empty response for others
            return LLMResponse(
                content={"facts": []}, input_tokens=80, output_tokens=10,
                model=model, raw_text="{}",
            )

        def embed(self, texts, *, model):
            return FakeLLMClient(embedding_dim=32).embed(texts, model=model)

    fake = _IntermittentFake()
    config = InsightConfig(
        retrieval_top_k=4, max_chunks_per_document=4, chunk_workers=4,
    )
    pipeline = InsightPipeline(llm_client=fake, config=config)
    # Must not raise — failed chunk is logged and skipped
    result = pipeline.run(QUESTION_SUDAN, [DOC_WHO_SUDAN])

    # The doc still counts as processed
    assert result.documents_processed == 1
    # All four chunks were attempted (the failure didn't abort the rest)
    assert fake.calls == 4


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
