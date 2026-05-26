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


# ---------------------------------------------------------------------------
# Trafilatura-XML extraction path
# ---------------------------------------------------------------------------

class TestTrafilaturaXmlExtraction:
    """The HTML parser prefers trafilatura's structural XML output (which
    strips navigation, sidebars, "related articles" lists, and other
    site boilerplate) and falls back to the DOM walker only when
    trafilatura returns too little content to be trustworthy.
    """

    def test_xml_path_strips_unrelated_sibling_articles(self, html_parser):
        """A page whose <body> contains the target article AND multiple
        unrelated articles should yield only the target's content via
        the trafilatura path."""
        html = b"""
        <html><head><title>Target Article</title></head>
        <body>
        <main>
            <h1>Target Article Title</h1>
            <p>The target article body says fact one with enough text to
            survive trafilatura's content-density heuristic and clear the
            minimum-body-chars threshold easily.</p>
            <p>A second paragraph of the target article continues
            describing fact two in detail with concrete numbers like 602
            and 405 and other figures.</p>
        </main>
        <aside>
            <h2>Top reads</h2>
            <article><h3>Unrelated A</h3><p>Sidebar article A</p></article>
            <article><h3>Unrelated B</h3><p>Sidebar article B</p></article>
            <article><h3>Unrelated C</h3><p>Sidebar article C</p></article>
        </aside>
        </body></html>
        """
        result = html_parser.parse(html, source_url="https://example.com/a")
        all_text = " ".join(s.text or "" for s in result.sections)
        assert "fact one" in all_text
        assert "fact two" in all_text
        # The aside content must not appear in any extracted section
        assert "Sidebar article A" not in all_text
        assert "Sidebar article B" not in all_text
        assert "Sidebar article C" not in all_text
        # Every section should come from the trafilatura extractor
        for s in result.sections:
            assert s.extractor == "trafilatura"

    def test_xml_path_preserves_tables(self, html_parser):
        """Tables inside the main content should make it through as
        chunk_type=table with table_rows populated."""
        html = b"""
        <html><head><title>Table Page</title></head>
        <body>
        <main>
            <h1>Disease Surveillance Weekly</h1>
            <p>The following table summarises this week's outbreak alerts
            with one row per event, including pathogen and country, in
            enough text to exceed the minimum-body-chars threshold for
            the trafilatura extraction path to be selected.</p>
            <table>
                <thead><tr><th>Pathogen</th><th>Country</th></tr></thead>
                <tbody>
                    <tr><td>H5N1</td><td>United States</td></tr>
                    <tr><td>Measles</td><td>Utah</td></tr>
                    <tr><td>Mpox</td><td>Comoros</td></tr>
                </tbody>
            </table>
        </main>
        </body></html>
        """
        result = html_parser.parse(html, source_url="https://example.com/b")
        table_sections = [s for s in result.sections if s.chunk_type == "table"]
        assert len(table_sections) == 1
        table = table_sections[0]
        assert table.table_rows is not None
        # Header + 3 data rows
        assert len(table.table_rows) >= 3
        # The headers and at least one cell should appear in the rows
        flat = [c for row in table.table_rows for c in row]
        assert "H5N1" in flat
        assert "Utah" in flat

    def test_falls_back_to_dom_walker_on_thin_xml(self, html_parser):
        """When trafilatura can extract very little, the parser falls
        back to the DOM walker so navigation-heavy pages still produce
        SOMETHING rather than silently extracting nothing.

        The fallback path is also what keeps the parser working when
        trafilatura is not installed at all.
        """
        # Tiny page where trafilatura will find almost no content.
        html = b"""
        <html><head><title>Tiny</title></head>
        <body>
        <nav>Menu Home About Search</nav>
        <p>Hello.</p>
        </body></html>
        """
        result = html_parser.parse(html, source_url="https://example.com/c")
        # The result is allowed to be empty here (no good content
        # anywhere), but the call must not raise.
        assert isinstance(result.sections, list)

    def test_metadata_extracted_independently_of_body_path(self, html_parser):
        """Title, published_date, and language must come from the raw
        DOM head/meta regardless of which body extraction path runs."""
        html = b"""
        <html lang="en">
        <head>
            <title>Real Title From Meta</title>
            <meta property="og:title" content="Real Title From Meta">
            <meta property="article:published_time" content="2026-03-15T00:00:00">
        </head>
        <body>
        <main>
            <p>Body content long enough to satisfy the trafilatura path
            threshold by clearing the minimum number of characters required
            for structured extraction to be preferred over the fallback.</p>
            <p>And a second paragraph for good measure.</p>
        </main>
        </body></html>
        """
        result = html_parser.parse(html, source_url="https://example.com/d")
        assert result.title == "Real Title From Meta"
        assert result.published_date is not None
        assert result.published_date.year == 2026
        assert result.language == "en"
