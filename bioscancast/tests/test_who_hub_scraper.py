"""Tests for the shared WHO situation-hub scraper (``_who_hub_common``).

All offline: the network boundary ``_get`` is monkeypatched to return canned
hub / item / PDF responses.
"""

from __future__ import annotations

from datetime import datetime, timezone

from bs4 import BeautifulSoup

from bioscancast.stages.extraction.custom_scrapers import _who_hub_common
from bioscancast.stages.extraction.custom_scrapers._who_hub_common import (
    _select_item_pdf,
    fetch_who_hub_latest_pdf,
)


class FakeResponse:
    """Minimal stand-in for a curl_cffi Response used by ``_get``."""

    def __init__(self, *, text: str = "", content: bytes = b"", status_code: int = 200,
                 url: str = "https://example.com"):
        self.text = text
        self.content = content
        self.status_code = status_code
        self.url = url


# ---------------------------------------------------------------------------
# _select_item_pdf — the issue #59 regression surface
# ---------------------------------------------------------------------------

class TestSelectItemPdf:
    def _soup(self, html: str) -> BeautifulSoup:
        return BeautifulSoup(html, "html.parser")

    def test_prefers_situation_report_over_first_pdf(self):
        # An annex PDF appears first in document order; the situation report
        # must still be chosen (issue #59).
        html = """
        <a href="https://cdn.who.int/media/docs/annex-map.pdf">Annex: map</a>
        <a href="https://cdn.who.int/media/docs/situation-reports/cholera-update.pdf">
            Epidemiological update
        </a>
        """
        chosen = _select_item_pdf(self._soup(html), "https://who.int/item", "cholera")
        assert chosen.endswith("situation-reports/cholera-update.pdf")

    def test_prefers_keyword_pdf_over_translation(self):
        html = """
        <a href="https://cdn.who.int/media/docs/brief-fr.pdf">Version française</a>
        <a href="https://cdn.who.int/media/docs/cholera-epi-update.pdf">Report</a>
        """
        chosen = _select_item_pdf(self._soup(html), "https://who.int/item", "cholera")
        assert chosen.endswith("cholera-epi-update.pdf")

    def test_falls_back_to_first_pdf_when_indistinguishable(self):
        # No situation-report / keyword signal on either link: keep the first,
        # preserving the pre-fix behaviour.
        html = """
        <a href="https://cdn.who.int/media/docs/doc-a.pdf">Document A</a>
        <a href="https://cdn.who.int/media/docs/doc-b.pdf">Document B</a>
        """
        chosen = _select_item_pdf(self._soup(html), "https://who.int/item", "cholera")
        assert chosen.endswith("doc-a.pdf")

    def test_returns_none_when_no_pdf(self):
        html = '<a href="https://who.int/item/other">Related item</a>'
        assert _select_item_pdf(self._soup(html), "https://who.int/item", "cholera") is None


# ---------------------------------------------------------------------------
# fetch_who_hub_latest_pdf — end-to-end with the wrong-PDF trap
# ---------------------------------------------------------------------------

class TestFetchWhoHubLatestPdf:
    def test_selects_situation_report_pdf_end_to_end(self, monkeypatch):
        hub_html = """
        <a href="/publications/m/item/cholera-update-30-June-2026">
            Cholera epidemiological update, 30 June 2026
        </a>
        """
        item_html = """
        <a href="https://cdn.who.int/media/docs/annex-map.pdf">Annex: map</a>
        <a href="https://cdn.who.int/media/docs/situation-reports/cholera.pdf">
            Epidemiological update
        </a>
        """
        pdf_bytes = b"%PDF-1.7 fake"

        def fake_get(url, cfg):
            if url.endswith(".pdf"):
                return FakeResponse(content=pdf_bytes, url=url)
            if "/publications/" in url:
                return FakeResponse(text=item_html, url=url)
            return FakeResponse(text=hub_html, url=url)

        monkeypatch.setattr(_who_hub_common, "_get", fake_get)

        result = fetch_who_hub_latest_pdf(
            "https://who.int/emergencies/cholera",
            "cholera",
            as_of_date=datetime(2026, 7, 1, tzinfo=timezone.utc),
        )
        assert result is not None
        assert result.content_type == "application/pdf"
        assert result.content_bytes == pdf_bytes
        assert result.url.endswith("situation-reports/cholera.pdf")
