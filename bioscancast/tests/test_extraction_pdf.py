"""Tests for bioscancast.extraction.parsers.pdf_parser using fixture files."""

from __future__ import annotations

from pathlib import Path

import pytest

from bioscancast.extraction.parsers.pdf_parser import PdfParser

FIXTURES = Path(__file__).parent / "fixtures" / "extraction"


@pytest.fixture
def pdf_parser():
    return PdfParser(max_pages=100)


@pytest.fixture
def who_pdf():
    return (FIXTURES / "who_don_sample.pdf").read_bytes()


# ---------------------------------------------------------------------------
# can_parse
# ---------------------------------------------------------------------------

class TestCanParse:
    def test_pdf_content_type(self, pdf_parser):
        assert pdf_parser.can_parse("application/pdf", b"")

    def test_pdf_magic_bytes(self, pdf_parser):
        assert pdf_parser.can_parse("", b"%PDF-1.7 rest")

    def test_rejects_html(self, pdf_parser):
        assert not pdf_parser.can_parse("text/html", b"<html>")


# ---------------------------------------------------------------------------
# WHO DON PDF fixture
# ---------------------------------------------------------------------------

class TestWhoDonPdf:
    def test_page_count(self, pdf_parser, who_pdf):
        result = pdf_parser.parse(who_pdf, source_url="https://who.int/don")
        assert result.page_count is not None
        assert result.page_count == 3

    def test_title_from_metadata(self, pdf_parser, who_pdf):
        result = pdf_parser.parse(who_pdf, source_url="https://who.int/don")
        assert result.title is not None
        assert "Sudan" in result.title or "Disease" in result.title

    def test_published_date_from_metadata(self, pdf_parser, who_pdf):
        result = pdf_parser.parse(who_pdf, source_url="https://who.int/don")
        assert result.published_date is not None
        assert result.published_date.year == 2024

    def test_sections_have_page_numbers(self, pdf_parser, who_pdf):
        result = pdf_parser.parse(who_pdf, source_url="https://who.int/don")
        pages = {s.page_number for s in result.sections if s.page_number}
        assert len(pages) > 0
        assert all(1 <= p <= 3 for p in pages)

    def test_heading_derived_sections(self, pdf_parser, who_pdf):
        result = pdf_parser.parse(who_pdf, source_url="https://who.int/don")
        paths = [s.section_path for s in result.sections if s.section_path]
        assert len(paths) > 0

    def test_table_chunk_found(self, pdf_parser, who_pdf):
        result = pdf_parser.parse(who_pdf, source_url="https://who.int/don")
        table_sections = [s for s in result.sections if s.chunk_type == "table"]
        # The fixture has a table on page 2
        assert len(table_sections) >= 1

    def test_table_has_data(self, pdf_parser, who_pdf):
        result = pdf_parser.parse(who_pdf, source_url="https://who.int/don")
        table_sections = [s for s in result.sections if s.chunk_type == "table"]
        if table_sections:
            table = table_sections[0]
            assert table.table_rows is not None
            assert len(table.table_rows) > 1

    def test_raw_text_not_empty(self, pdf_parser, who_pdf):
        result = pdf_parser.parse(who_pdf, source_url="https://who.int/don")
        assert len(result.raw_text) > 100

    def test_not_partial(self, pdf_parser, who_pdf):
        result = pdf_parser.parse(who_pdf, source_url="https://who.int/don")
        assert not result.is_partial

    def test_page_cap_triggers_partial(self, who_pdf):
        parser = PdfParser(max_pages=2)
        result = parser.parse(who_pdf, source_url="https://who.int/don")
        assert result.is_partial
        assert result.partial_reason is not None
        assert "Truncated" in result.partial_reason
