"""End-to-end integration tests against 7 real biosecurity documents.

Wires the real ExtractionPipeline (fetcher monkey-patched to read on-disk
bytes) into the insight pipeline with deterministic fake LLMs. Uses the
documents already committed under ``data/docling_eval/sources/``.

Assertions use ``>=`` thresholds so subsequent insight-stage refactors
(items 5–7 of the hardening plan) that legitimately add records don't
break this test. The numbers are floors, calibrated from the observed
behaviour with the layered hallucination guard.

No live LLM calls. Runs in <10 seconds.
"""

from __future__ import annotations

import pytest

from bioscancast.filtering.models import ForecastQuestion
from bioscancast.insight.config import InsightConfig
from bioscancast.insight.pipeline import InsightPipeline
from bioscancast.schemas import Document
from bioscancast.tests.fixtures.insight.real_doc_extracts import (
    HallucinatingFakeLLM,
    QuoteEchoingFakeLLM,
    SOURCES,
    extract_real_documents,
    get_source,
)


# ---------------------------------------------------------------------------
# Module-scope fixture — extract once, reuse across tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_docs() -> dict[str, Document]:
    """Extract all 7 real source documents once for the whole module."""
    return extract_real_documents()


@pytest.fixture
def insight_config() -> InsightConfig:
    """Config matching the parameters used in the live evaluation."""
    return InsightConfig(
        retrieval_top_k=5,
        max_chunks_per_document=5,
        max_input_tokens_per_run=10_000_000,
    )


# ---------------------------------------------------------------------------
# Extraction sanity
# ---------------------------------------------------------------------------


def test_extraction_produces_a_document_for_every_source(real_docs):
    """Every source file in SOURCES must yield exactly one Document."""
    assert set(real_docs.keys()) == {src["name"] for src in SOURCES}


def test_extraction_africa_cdc_fails_with_requires_ocr(real_docs):
    """Africa CDC's PDF is image-only — the in-tree parser must flag
    this as ``requires_ocr`` rather than silently producing no chunks."""
    doc = real_docs["africa_cdc_weekly_apr2026"]
    assert doc.status == "failed"
    assert doc.error_message == "requires_ocr"
    assert doc.chunks == []


@pytest.mark.parametrize(
    # (name, min_chunks). CIDRAP has a lower floor because trafilatura's
    # main-content extraction correctly isolates only the actual article
    # body (~2 chunks) rather than the surrounding navigation, sidebars,
    # and three additional unrelated articles that the raw DOM contains.
    # Other sources are calibrated to today's behaviour.
    "name,min_chunks",
    [
        ("who_mpox_sitrep64", 5),
        ("who_cholera_epi34", 5),
        ("cdc_mmwr_nm_measles", 5),
        ("ecdc_cdtr_week16", 5),
        ("cidrap_utah_measles", 1),
        ("promed_latest", 5),
    ],
)
def test_extraction_produces_chunks_for_text_extractable_sources(
    real_docs, name, min_chunks
):
    """Every source except Africa CDC must extract at least a few chunks."""
    doc = real_docs[name]
    assert doc.status == "success", (
        f"{name}: expected status=success, got {doc.status}"
    )
    assert len(doc.chunks) >= min_chunks, (
        f"{name}: expected >= {min_chunks} chunks, got {len(doc.chunks)}"
    )


def test_who_mpox_publication_date_extracted(real_docs):
    """WHO PDFs carry usable /CreationDate metadata."""
    doc = real_docs["who_mpox_sitrep64"]
    assert doc.published_date is not None
    assert doc.published_date.year == 2026


# ---------------------------------------------------------------------------
# Insight pipeline happy path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    # (name, min_records). Floors are calibrated with the layered
    # hallucination guard; future items that lift extraction quality may
    # legitimately push records higher.
    "name,min_records",
    [
        ("who_mpox_sitrep64", 1),
        ("who_cholera_epi34", 1),
        ("cdc_mmwr_nm_measles", 1),
        ("ecdc_cdtr_week16", 1),
        ("cidrap_utah_measles", 1),
    ],
)
def test_pipeline_produces_at_least_one_record_per_text_doc(
    real_docs, insight_config, name, min_records
):
    """With a quote-echoing fake LLM, every text-extractable doc should
    yield at least one record (the fake picks a number-bearing sentence
    from each retrieved chunk; the layered guard accepts the picked
    quote as long as it's a real verbatim substring of the chunk)."""
    doc = real_docs[name]
    src = get_source(name)
    fake = QuoteEchoingFakeLLM()
    pipe = InsightPipeline(llm_client=fake, config=insight_config)
    result = pipe.run(src["question"], [doc])
    assert len(result.records) >= min_records, (
        f"{name}: expected >= {min_records} records, got {len(result.records)}"
    )
    # Every record must carry valid provenance
    for rec in result.records:
        assert rec.sources, f"{name}: record has no sources"
        for s in rec.sources:
            assert s.chunk_id.startswith(f"{doc.id}-")
            assert s.source_url == doc.source_url
            assert s.quote, f"{name}: record has empty quote"


def test_cidrap_pipeline_captures_602_utah_cases(real_docs, insight_config):
    """The CIDRAP article's headline fact ("602 measles cases in Utah")
    should appear in the quote of at least one record."""
    doc = real_docs["cidrap_utah_measles"]
    src = get_source("cidrap_utah_measles")
    fake = QuoteEchoingFakeLLM()
    pipe = InsightPipeline(llm_client=fake, config=insight_config)
    result = pipe.run(src["question"], [doc])
    assert any(
        "602" in s.quote
        for rec in result.records
        for s in rec.sources
    ), "CIDRAP: expected at least one record citing '602' (measles cases)"


def test_failed_doc_is_skipped(real_docs, insight_config):
    """Africa CDC (status=failed) must take the skip path and never
    cause a per-chunk LLM call."""
    doc = real_docs["africa_cdc_weekly_apr2026"]
    src = get_source("africa_cdc_weekly_apr2026")
    fake = QuoteEchoingFakeLLM()
    pipe = InsightPipeline(llm_client=fake, config=insight_config)
    result = pipe.run(src["question"], [doc])
    assert result.documents_skipped == 1
    assert result.documents_processed == 0
    assert len(fake.calls) == 0
    assert any("Skipped" in note for note in result.notes)


# ---------------------------------------------------------------------------
# Hallucination guard end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "who_mpox_sitrep64",
        "who_cholera_epi34",
        "cdc_mmwr_nm_measles",
        "ecdc_cdtr_week16",
        "cidrap_utah_measles",
        "promed_latest",
    ],
)
def test_hallucination_guard_drops_every_fact_with_fabricated_quote(
    real_docs, insight_config, name
):
    """A HallucinatingFakeLLM that always emits a quote that doesn't
    appear in any chunk must produce zero records on every doc."""
    doc = real_docs[name]
    src = get_source(name)
    fake = HallucinatingFakeLLM()
    pipe = InsightPipeline(llm_client=fake, config=insight_config)
    result = pipe.run(src["question"], [doc])
    assert fake.calls > 0, f"{name}: fake was never called"
    assert result.records == [], (
        f"{name}: hallucination guard let through {len(result.records)} fact(s)"
    )


# ---------------------------------------------------------------------------
# Cross-document deduplication
# ---------------------------------------------------------------------------


def test_cross_doc_dedup_merges_identical_facts(real_docs, insight_config):
    """Two docs producing facts with the same dedup key should merge
    into a single record whose sources span both docs."""
    cidrap = real_docs["cidrap_utah_measles"]
    mmwr = real_docs["cdc_mmwr_nm_measles"]
    assert cidrap.status == "success"
    assert mmwr.status == "success"

    # Custom fake that returns an identically-structured fact for every
    # chunk it sees, but uses a real quote pulled from the chunk text
    # (so the hallucination guard accepts it).
    class _TwinFactFake(QuoteEchoingFakeLLM):
        def generate_json(self, *, system, user, schema, model, max_tokens=1024):
            response = super().generate_json(
                system=system, user=user, schema=schema, model=model,
                max_tokens=max_tokens,
            )
            # Force every fact to dedup-collide on the same key
            response.content["facts"][0].update({
                "event_type": "case_count",
                "location": "United States",
                "pathogen": "measles",
                "metric_name": "confirmed_cases",
                "metric_value": 42.0,
                "metric_unit": "cases",
                "event_date": "2026-03-01",
            })
            return response

    question = ForecastQuestion(
        id="q-cross-doc-measles",
        text="How many measles cases reported in the US?",
        created_at=cidrap.fetched_at.replace(tzinfo=None),
        region="United States",
        pathogen="measles",
        event_type="case_count",
    )
    fake = _TwinFactFake()
    pipe = InsightPipeline(llm_client=fake, config=insight_config)
    result = pipe.run(question, [cidrap, mmwr])

    assert len(result.records) == 1, (
        f"Expected 1 merged record, got {len(result.records)}"
    )
    record = result.records[0]
    source_doc_ids = {s.document_id for s in record.sources}
    assert source_doc_ids == {cidrap.id, mmwr.id}, (
        f"Expected sources from both docs, got {source_doc_ids}"
    )
