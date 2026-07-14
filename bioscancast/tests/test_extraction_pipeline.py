"""End-to-end tests for bioscancast.stages.extraction.pipeline with monkeypatched fetcher."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from bioscancast.stages.extraction.config import ExtractionConfig
from bioscancast.stages.extraction.fetcher import FetchResult
from bioscancast.stages.extraction.pipeline import ExtractionPipeline
from bioscancast.stages.filtering.models import FilteredDocument
from bioscancast.schemas.document import Document

FIXTURES = Path(__file__).parent / "fixtures" / "extraction"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_filtered_doc(
    result_id: str = "r1",
    url: str = "https://example.com/page",
    domain: str = "example.com",
    title: str = "Test Doc",
    extraction_priority: int = 1,
    extraction_mode: str = "html",
    expected_value: str = "medium",
) -> FilteredDocument:
    return FilteredDocument(
        result_id=result_id,
        question_id="q1",
        url=url,
        canonical_url=url,
        domain=domain,
        title=title,
        snippet="A test snippet.",
        published_date=None,
        file_type=None,
        relevance_score=0.8,
        credibility_score=0.7,
        final_score=0.75,
        source_tier="trusted_media",
        is_official_domain=False,
        selection_reasons=["test"],
        extraction_priority=extraction_priority,
        extraction_mode=extraction_mode,
        expected_value=expected_value,
    )


def _make_fetch_result(
    url: str,
    content: bytes,
    content_type: str = "text/html",
    error: str | None = None,
) -> FetchResult:
    return FetchResult(
        url=url,
        final_url=url,
        status_code=200 if not error else None,
        content_type=content_type,
        content_bytes=content if not error else None,
        fetched_at=datetime.now(timezone.utc),
        error=error,
    )


def _fake_fetch_factory(mapping: dict[str, FetchResult]):
    """Return a fetch function that looks up results by URL."""

    def fake_fetch(
        url,
        *,
        config=None,
        as_of_date=None,
        source_id=None,
        region=None,
        question_text=None,
    ):
        if url in mapping:
            return mapping[url]
        return _make_fetch_result(url, b"", error="not_found")

    return fake_fetch


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------

class TestExtractionPipeline:
    def test_html_extraction_end_to_end(self):
        html_bytes = (FIXTURES / "reuters_article.html").read_bytes()
        fdoc = _make_filtered_doc(
            url="https://reuters.com/mpox",
            domain="reuters.com",
        )
        fetch_map = {
            "https://reuters.com/mpox": _make_fetch_result(
                "https://reuters.com/mpox", html_bytes
            )
        }

        with patch(
            "bioscancast.stages.extraction.pipeline.fetch",
            side_effect=_fake_fetch_factory(fetch_map),
        ):
            pipeline = ExtractionPipeline()
            results = pipeline.run([fdoc])

        assert len(results) == 1
        doc = results[0]
        assert doc.status == "success"
        assert len(doc.chunks) > 0
        assert doc.char_count > 0
        assert doc.token_count > 0
        assert doc.document_type == "html"

    def test_pdf_extraction_end_to_end(self):
        pdf_bytes = (FIXTURES / "who_don_sample.pdf").read_bytes()
        fdoc = _make_filtered_doc(
            result_id="r_pdf",
            url="https://who.int/don.pdf",
            domain="who.int",
            extraction_mode="pdf",
        )
        fetch_map = {
            "https://who.int/don.pdf": _make_fetch_result(
                "https://who.int/don.pdf", pdf_bytes, content_type="application/pdf"
            )
        }

        with patch(
            "bioscancast.stages.extraction.pipeline.fetch",
            side_effect=_fake_fetch_factory(fetch_map),
        ):
            pipeline = ExtractionPipeline()
            results = pipeline.run([fdoc])

        assert len(results) == 1
        doc = results[0]
        assert doc.status == "success"
        assert doc.document_type == "pdf"
        assert doc.page_count is not None
        assert len(doc.chunks) > 0

    def test_refiner_receives_resolved_pdf_url_not_hub(self):
        # A hub scraper (who_cholera, who_h5_hai, ...) resolves a landing page
        # who.int/<hub> to a cdn.who.int/.../*.pdf. The Docling allowlist targets
        # the resolved PDF path, so the refiner must be handed
        # ``fetch_result.final_url`` — matching on the hub ``filtered_doc.url``
        # would never fire the allowlist for exactly those scraper PDFs.
        pdf_bytes = (FIXTURES / "who_don_sample.pdf").read_bytes()
        hub_url = "https://www.who.int/emergencies/situations/cholera-upsurge"
        pdf_url = (
            "https://cdn.who.int/media/docs/default-source/documents/"
            "emergencies/situation-reports/cholera_update.pdf"
        )
        fdoc = _make_filtered_doc(
            result_id="r_hub", url=hub_url, domain="who.int", extraction_mode="pdf"
        )
        fetch_map = {
            hub_url: FetchResult(
                url=pdf_url,
                final_url=pdf_url,
                status_code=200,
                content_type="application/pdf",
                content_bytes=pdf_bytes,
                fetched_at=datetime.now(timezone.utc),
                error=None,
            )
        }

        seen: dict[str, str] = {}

        class _SpyRefiner:
            def refine(self, parsed, *, source_url, content):
                seen["source_url"] = source_url
                return parsed

        with patch(
            "bioscancast.stages.extraction.pipeline.fetch",
            side_effect=_fake_fetch_factory(fetch_map),
        ), patch.object(
            ExtractionPipeline, "_get_docling_refiner", return_value=_SpyRefiner()
        ):
            pipeline = ExtractionPipeline(
                config=ExtractionConfig(enable_docling_refiner=True)
            )
            pipeline.run([fdoc])

        assert seen.get("source_url") == pdf_url

    def test_ordering_by_extraction_priority(self):
        html_bytes = b"<html><body><p>Test content</p></body></html>"
        fdoc_low = _make_filtered_doc(
            result_id="r_low",
            url="https://a.com/low",
            extraction_priority=3,
        )
        fdoc_high = _make_filtered_doc(
            result_id="r_high",
            url="https://b.com/high",
            extraction_priority=1,
        )

        fetch_map = {
            "https://a.com/low": _make_fetch_result("https://a.com/low", html_bytes),
            "https://b.com/high": _make_fetch_result("https://b.com/high", html_bytes),
        }

        with patch(
            "bioscancast.stages.extraction.pipeline.fetch",
            side_effect=_fake_fetch_factory(fetch_map),
        ):
            pipeline = ExtractionPipeline()
            results = pipeline.run([fdoc_low, fdoc_high])

        assert len(results) == 2
        # Higher priority (lower number) should be processed first
        assert results[0].result_id == "r_high"
        assert results[1].result_id == "r_low"

    def test_failure_isolation(self):
        html_bytes = b"<html><body><p>Good content</p></body></html>"
        fdoc_good = _make_filtered_doc(
            result_id="r_good",
            url="https://good.com/page",
            extraction_priority=2,
        )
        fdoc_bad = _make_filtered_doc(
            result_id="r_bad",
            url="https://bad.com/broken",
            extraction_priority=1,
        )

        fetch_map = {
            "https://good.com/page": _make_fetch_result(
                "https://good.com/page", html_bytes
            ),
            "https://bad.com/broken": _make_fetch_result(
                "https://bad.com/broken", b"", error="Connection timeout"
            ),
        }

        with patch(
            "bioscancast.stages.extraction.pipeline.fetch",
            side_effect=_fake_fetch_factory(fetch_map),
        ):
            pipeline = ExtractionPipeline()
            results = pipeline.run([fdoc_good, fdoc_bad])

        assert len(results) == 2
        # Bad doc should be failed but not crash the pipeline
        bad_doc = [r for r in results if r.result_id == "r_bad"][0]
        good_doc = [r for r in results if r.result_id == "r_good"][0]
        assert bad_doc.status == "failed"
        assert bad_doc.error_message is not None
        assert good_doc.status == "success"

    def test_no_parser_available(self):
        fdoc = _make_filtered_doc(
            url="https://example.com/data.bin",
        )
        fetch_map = {
            "https://example.com/data.bin": _make_fetch_result(
                "https://example.com/data.bin",
                b"\x00\x01\x02\x03binary",
                content_type="application/octet-stream",
            )
        }

        with patch(
            "bioscancast.stages.extraction.pipeline.fetch",
            side_effect=_fake_fetch_factory(fetch_map),
        ):
            pipeline = ExtractionPipeline()
            results = pipeline.run([fdoc])

        # PlaintextParser is catch-all, so it will parse anything
        # But if the content is binary garbage, it should still produce a Document
        assert len(results) == 1

    def test_chunk_ids_are_stable(self):
        html_bytes = (FIXTURES / "cdc_dashboard.html").read_bytes()
        fdoc = _make_filtered_doc(
            result_id="r_stable",
            url="https://cdc.gov/dashboard",
        )
        fetch_map = {
            "https://cdc.gov/dashboard": _make_fetch_result(
                "https://cdc.gov/dashboard", html_bytes
            )
        }

        with patch(
            "bioscancast.stages.extraction.pipeline.fetch",
            side_effect=_fake_fetch_factory(fetch_map),
        ):
            pipeline = ExtractionPipeline()
            results = pipeline.run([fdoc])

        doc = results[0]
        assert doc.id == "doc-r_stable"
        for chunk in doc.chunks:
            assert chunk.chunk_id.startswith("doc-r_stable-c")

    def test_document_json_serializable(self):
        html_bytes = b"<html><body><p>Serialization test</p></body></html>"
        fdoc = _make_filtered_doc(url="https://example.com/serial")
        fetch_map = {
            "https://example.com/serial": _make_fetch_result(
                "https://example.com/serial", html_bytes
            )
        }

        with patch(
            "bioscancast.stages.extraction.pipeline.fetch",
            side_effect=_fake_fetch_factory(fetch_map),
        ):
            pipeline = ExtractionPipeline()
            results = pipeline.run([fdoc])

        doc = results[0]
        # Must be JSON-serializable (datetime needs default=str)
        serialized = json.dumps(asdict(doc), default=str)
        assert isinstance(serialized, str)
        parsed = json.loads(serialized)
        assert parsed["id"] == doc.id

    def test_failed_document_json_serializable(self):
        fdoc = _make_filtered_doc(url="https://example.com/fail")
        fetch_map = {
            "https://example.com/fail": _make_fetch_result(
                "https://example.com/fail", b"", error="DNS failure"
            )
        }

        with patch(
            "bioscancast.stages.extraction.pipeline.fetch",
            side_effect=_fake_fetch_factory(fetch_map),
        ):
            pipeline = ExtractionPipeline()
            results = pipeline.run([fdoc])

        doc = results[0]
        assert doc.status == "failed"
        serialized = json.dumps(asdict(doc), default=str)
        assert isinstance(serialized, str)

    def test_extract_one(self):
        html_bytes = b"<html><body><h1>Title</h1><p>Content here.</p></body></html>"
        fdoc = _make_filtered_doc(url="https://example.com/one")
        fetch_map = {
            "https://example.com/one": _make_fetch_result(
                "https://example.com/one", html_bytes
            )
        }

        with patch(
            "bioscancast.stages.extraction.pipeline.fetch",
            side_effect=_fake_fetch_factory(fetch_map),
        ):
            pipeline = ExtractionPipeline()
            doc = pipeline.extract_one(fdoc)

        assert isinstance(doc, Document)
        assert doc.status == "success"


# ---------------------------------------------------------------------------
# Empty chunk handling — _drop_or_repair_empty_chunks
# ---------------------------------------------------------------------------

class TestEmptyChunkHandling:
    """The extraction pipeline must drop empty prose chunks and render
    table chunks whose text is empty (but whose table_data is populated)
    into a flat text representation. See pipeline._drop_or_repair_empty_chunks."""

    def test_empty_prose_chunk_is_dropped(self):
        from bioscancast.stages.extraction.pipeline import _drop_or_repair_empty_chunks
        from bioscancast.schemas.document import DocumentChunk

        chunks = [
            DocumentChunk(
                chunk_id="c0", chunk_index=0, text="Real content.",
                chunk_type="prose",
            ),
            DocumentChunk(
                chunk_id="c1", chunk_index=1, text="",
                chunk_type="prose",
            ),
            DocumentChunk(
                chunk_id="c2", chunk_index=2, text="   \n\t  ",
                chunk_type="prose",
            ),
            DocumentChunk(
                chunk_id="c3", chunk_index=3, text="More content.",
                chunk_type="prose",
            ),
        ]
        out = _drop_or_repair_empty_chunks(chunks)
        assert [c.chunk_id for c in out] == ["c0", "c3"]

    def test_empty_text_table_chunk_with_table_data_is_repaired(self):
        from bioscancast.stages.extraction.pipeline import _drop_or_repair_empty_chunks
        from bioscancast.schemas.document import DocumentChunk

        table_data = [
            ["Country", "Cases", "Deaths"],
            ["Uganda", "9", "3"],
            ["DRC", "6543", "148"],
        ]
        chunks = [
            DocumentChunk(
                chunk_id="t0", chunk_index=0, text="",
                chunk_type="table", table_data=table_data,
            ),
        ]
        out = _drop_or_repair_empty_chunks(chunks)
        assert len(out) == 1
        assert "Uganda" in out[0].text
        assert "6543" in out[0].text
        assert "Deaths" in out[0].text
        # Original structured data is preserved unchanged
        assert out[0].table_data == table_data
        # Token count is recomputed for the rendered text
        assert out[0].token_count is not None
        assert out[0].token_count > 0

    def test_empty_table_chunk_without_table_data_is_dropped(self):
        from bioscancast.stages.extraction.pipeline import _drop_or_repair_empty_chunks
        from bioscancast.schemas.document import DocumentChunk

        chunks = [
            DocumentChunk(
                chunk_id="t0", chunk_index=0, text="",
                chunk_type="table", table_data=None,
            ),
            DocumentChunk(
                chunk_id="t1", chunk_index=1, text="",
                chunk_type="table", table_data=[],
            ),
        ]
        out = _drop_or_repair_empty_chunks(chunks)
        assert out == []

    def test_empty_table_chunk_with_only_empty_rows_is_dropped(self):
        from bioscancast.stages.extraction.pipeline import _drop_or_repair_empty_chunks
        from bioscancast.schemas.document import DocumentChunk

        chunks = [
            DocumentChunk(
                chunk_id="t0", chunk_index=0, text="",
                chunk_type="table",
                table_data=[["", "", ""], ["", ""]],
            ),
        ]
        out = _drop_or_repair_empty_chunks(chunks)
        assert out == []

    def test_table_chunk_with_text_passes_through_unchanged(self):
        """A table chunk that already has text is left alone."""
        from bioscancast.stages.extraction.pipeline import _drop_or_repair_empty_chunks
        from bioscancast.schemas.document import DocumentChunk

        chunks = [
            DocumentChunk(
                chunk_id="t0", chunk_index=0,
                text="Existing flowed text",
                chunk_type="table",
                table_data=[["a", "b"]],
                token_count=4,
            ),
        ]
        out = _drop_or_repair_empty_chunks(chunks)
        assert len(out) == 1
        assert out[0].text == "Existing flowed text"  # unchanged
        assert out[0].token_count == 4  # unchanged
