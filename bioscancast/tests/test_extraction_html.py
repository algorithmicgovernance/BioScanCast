"""Tests for bioscancast.extraction.parsers.html_parser using fixture files."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from bioscancast.extraction.parsers.html_parser import HtmlParser

FIXTURES = Path(__file__).parent / "fixtures" / "extraction"


@pytest.fixture
def html_parser():
    return HtmlParser()


@pytest.fixture
def cdc_html():
    return (FIXTURES / "cdc_dashboard.html").read_bytes()


@pytest.fixture
def reuters_html():
    return (FIXTURES / "reuters_article.html").read_bytes()


# ---------------------------------------------------------------------------
# can_parse
# ---------------------------------------------------------------------------

class TestCanParse:
    def test_text_html_content_type(self, html_parser):
        assert html_parser.can_parse("text/html", b"")

    def test_html_magic_bytes(self, html_parser):
        assert html_parser.can_parse("", b"<!DOCTYPE html><html>")

    def test_rejects_pdf(self, html_parser):
        assert not html_parser.can_parse("application/pdf", b"%PDF-1.7")


# ---------------------------------------------------------------------------
# CDC dashboard fixture
# ---------------------------------------------------------------------------

class TestCdcDashboard:
    def test_title_extracted(self, html_parser, cdc_html):
        result = html_parser.parse(cdc_html, source_url="https://cdc.gov/bird-flu")
        assert result.title is not None
        assert "H5N1" in result.title

    def test_published_date(self, html_parser, cdc_html):
        result = html_parser.parse(cdc_html, source_url="https://cdc.gov/bird-flu")
        assert result.published_date is not None
        assert result.published_date.year == 2024
        assert result.published_date.month == 12

    def test_language(self, html_parser, cdc_html):
        result = html_parser.parse(cdc_html, source_url="https://cdc.gov/bird-flu")
        assert result.language == "en"

    def test_sections_recovered(self, html_parser, cdc_html):
        result = html_parser.parse(cdc_html, source_url="https://cdc.gov/bird-flu")
        assert len(result.sections) > 0

    def test_section_paths_contain_headings(self, html_parser, cdc_html):
        result = html_parser.parse(cdc_html, source_url="https://cdc.gov/bird-flu")
        paths = [s.section_path for s in result.sections if s.section_path]
        assert any("Epidemiological Summary" in p for p in paths)

    def test_table_chunks_found(self, html_parser, cdc_html):
        result = html_parser.parse(cdc_html, source_url="https://cdc.gov/bird-flu")
        table_sections = [s for s in result.sections if s.chunk_type == "table"]
        assert len(table_sections) >= 1

    def test_table_has_rows(self, html_parser, cdc_html):
        result = html_parser.parse(cdc_html, source_url="https://cdc.gov/bird-flu")
        table_sections = [s for s in result.sections if s.chunk_type == "table"]
        assert table_sections[0].table_rows is not None
        assert len(table_sections[0].table_rows) > 1  # header + data rows

    def test_raw_text_not_empty(self, html_parser, cdc_html):
        result = html_parser.parse(cdc_html, source_url="https://cdc.gov/bird-flu")
        assert len(result.raw_text) > 100


# ---------------------------------------------------------------------------
# Reuters article fixture
# ---------------------------------------------------------------------------

class TestReutersArticle:
    def test_title_extracted(self, html_parser, reuters_html):
        result = html_parser.parse(reuters_html, source_url="https://reuters.com/mpox")
        assert result.title is not None
        assert "mpox" in result.title.lower()

    def test_published_date(self, html_parser, reuters_html):
        result = html_parser.parse(reuters_html, source_url="https://reuters.com/mpox")
        assert result.published_date is not None
        assert result.published_date.year == 2024
        assert result.published_date.month == 8

    def test_prose_sections_only(self, html_parser, reuters_html):
        result = html_parser.parse(reuters_html, source_url="https://reuters.com/mpox")
        table_sections = [s for s in result.sections if s.chunk_type == "table"]
        assert len(table_sections) == 0

    def test_heading_sections_present(self, html_parser, reuters_html):
        result = html_parser.parse(reuters_html, source_url="https://reuters.com/mpox")
        paths = [s.section_path for s in result.sections if s.section_path]
        # Should have sections under the article headings
        assert len(paths) > 0

    def test_section_paths_contain_article_headings(self, html_parser, reuters_html):
        result = html_parser.parse(reuters_html, source_url="https://reuters.com/mpox")
        all_paths = " ".join(s.section_path or "" for s in result.sections)
        assert "Outbreak Spread" in all_paths or "International Response" in all_paths
