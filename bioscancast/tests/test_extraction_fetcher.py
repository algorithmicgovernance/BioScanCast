"""Tests for bioscancast.stages.extraction.fetcher — all offline via monkeypatching."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from bioscancast.stages.extraction.config import ExtractionConfig
from bioscancast.stages.extraction.fetcher import fetch, _sniff_content_type


# ---------------------------------------------------------------------------
# Content-type sniffing
# ---------------------------------------------------------------------------

class TestSniffContentType:
    def test_pdf_magic(self):
        assert _sniff_content_type(b"%PDF-1.7 rest of header") == "application/pdf"

    def test_html_doctype(self):
        assert _sniff_content_type(b"<!DOCTYPE html><html>") == "text/html"

    def test_html_tag(self):
        assert _sniff_content_type(b"  <html lang='en'>") == "text/html"

    def test_unknown(self):
        assert _sniff_content_type(b"Just some random text") is None

    def test_empty(self):
        assert _sniff_content_type(b"") is None


# ---------------------------------------------------------------------------
# Helpers: fake curl_cffi responses
# ---------------------------------------------------------------------------

class FakeResponse:
    """Minimal stand-in for a curl_cffi.requests streaming Response."""

    def __init__(
        self,
        *,
        status_code: int = 200,
        headers: dict | None = None,
        chunks: list[bytes] | None = None,
        url: str = "https://example.com/page",
    ):
        self.status_code = status_code
        # curl_cffi headers behave like a case-insensitive dict; a plain dict
        # with lowercase keys is sufficient for the fetcher's lookups.
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}
        self.url = url
        self._chunks = chunks or [b"<html><body>Hello</body></html>"]

    def iter_content(self):
        yield from self._chunks

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Fetch tests
# ---------------------------------------------------------------------------

class TestFetch:
    def _patch_get(self, response: FakeResponse):
        return patch(
            "bioscancast.stages.extraction.fetcher.curl_requests.get",
            return_value=response,
        )

    def test_successful_html_fetch(self):
        resp = FakeResponse(
            status_code=200,
            headers={"content-type": "text/html; charset=utf-8"},
            chunks=[b"<html><body>Hello world</body></html>"],
            url="https://example.com/page",
        )
        with self._patch_get(resp):
            result = fetch("https://example.com/page")

        assert result.error is None
        assert result.status_code == 200
        assert result.content_type == "text/html"
        assert result.content_bytes == b"<html><body>Hello world</body></html>"
        assert result.final_url == "https://example.com/page"

    def test_content_type_from_sniffing_when_header_missing(self):
        resp = FakeResponse(
            status_code=200,
            headers={},
            chunks=[b"%PDF-1.7 fake pdf content"],
        )
        with self._patch_get(resp):
            result = fetch("https://example.com/report")

        assert result.content_type == "application/pdf"

    def test_content_type_sniff_octet_stream(self):
        resp = FakeResponse(
            status_code=200,
            headers={"content-type": "application/octet-stream"},
            chunks=[b"<!DOCTYPE html><html><body>Hi</body></html>"],
        )
        with self._patch_get(resp):
            result = fetch("https://example.com/page")

        assert result.content_type == "text/html"

    def test_oversized_content_length_header(self):
        resp = FakeResponse(
            status_code=200,
            headers={
                "content-type": "application/pdf",
                "content-length": "999999999",
            },
            chunks=[b"small"],
        )
        config = ExtractionConfig(fetch_max_bytes=1000)
        with self._patch_get(resp):
            result = fetch("https://example.com/big.pdf", config=config)

        assert result.error is not None
        assert "exceeds" in result.error
        assert result.content_bytes is None

    def test_oversized_during_streaming(self):
        resp = FakeResponse(
            status_code=200,
            headers={"content-type": "text/html"},
            chunks=[b"a" * 600, b"b" * 600],
        )
        config = ExtractionConfig(fetch_max_bytes=1000)
        with self._patch_get(resp):
            result = fetch("https://example.com/page", config=config)

        assert result.error is not None
        assert "exceeded" in result.error
        assert result.content_bytes is None

    def test_network_error_returns_fetch_result(self):
        with patch(
            "bioscancast.stages.extraction.fetcher.curl_requests.get",
            side_effect=ConnectionError("Connection refused"),
        ):
            result = fetch("https://unreachable.example.com")

        assert result.error is not None
        assert "Connection refused" in result.error
        assert result.status_code is None
        assert result.content_bytes is None

    def test_redirect_captures_final_url(self):
        resp = FakeResponse(
            status_code=200,
            headers={"content-type": "text/html"},
            chunks=[b"<html>redirected</html>"],
            url="https://example.com/final-page",
        )
        with self._patch_get(resp):
            result = fetch("https://example.com/old-page")

        assert result.final_url == "https://example.com/final-page"
        assert result.url == "https://example.com/old-page"

    def test_fetched_at_is_utc(self):
        resp = FakeResponse(
            status_code=200,
            headers={"content-type": "text/html"},
            chunks=[b"<html></html>"],
        )
        with self._patch_get(resp):
            result = fetch("https://example.com/page")

        assert result.fetched_at.tzinfo is not None

    def test_impersonate_passed_to_curl_cffi(self):
        """The configured impersonation profile reaches curl_cffi.get."""
        resp = FakeResponse(
            status_code=200,
            headers={"content-type": "text/html"},
            chunks=[b"<html></html>"],
        )
        config = ExtractionConfig(impersonate="firefox")
        with patch(
            "bioscancast.stages.extraction.fetcher.curl_requests.get",
            return_value=resp,
        ) as mock_get:
            fetch("https://example.com/page", config=config)

        kwargs = mock_get.call_args.kwargs
        assert kwargs["impersonate"] == "firefox"
        assert kwargs["stream"] is True
        assert kwargs["allow_redirects"] is True


# ---------------------------------------------------------------------------
# Live integration test: opt-in with `pytest -m live`.
# Confirms curl_cffi successfully fetches a Cloudflare-fronted URL that
# httpx/requests would 401/403 on. Skipped by default to keep CI offline.
# ---------------------------------------------------------------------------

@pytest.mark.live
def test_live_cloudflare_fronted_fetch():
    # Reuters is the canonical case from the docling eval (see issue #18).
    # Pick a stable landing page rather than a dated article.
    url = "https://www.reuters.com/"
    result = fetch(url)
    assert result.error is None, f"fetch errored: {result.error}"
    assert result.status_code == 200, (
        f"expected 200 from Cloudflare-fronted URL, got {result.status_code}"
    )
    assert result.content_bytes is not None
    assert result.content_type == "text/html"
