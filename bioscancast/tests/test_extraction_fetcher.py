"""Tests for bioscancast.extraction.fetcher — all offline via monkeypatching."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import httpx
import pytest

from bioscancast.extraction.config import ExtractionConfig
from bioscancast.extraction.fetcher import FetchResult, fetch, _sniff_content_type


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
# Helpers: fake httpx responses
# ---------------------------------------------------------------------------

class FakeResponse:
    """Minimal stand-in for httpx.Response used in stream context."""

    def __init__(
        self,
        *,
        status_code: int = 200,
        headers: dict | None = None,
        chunks: list[bytes] | None = None,
        url: str = "https://example.com/page",
    ):
        self.status_code = status_code
        self.headers = httpx.Headers(headers or {})
        self.url = httpx.URL(url)
        self._chunks = chunks or [b"<html><body>Hello</body></html>"]

    def iter_bytes(self, chunk_size: int = 65536):
        yield from self._chunks

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class FakeClient:
    """Minimal stand-in for httpx.Client."""

    def __init__(self, response: FakeResponse):
        self._response = response

    def stream(self, method, url, **kwargs):
        return self._response

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# ---------------------------------------------------------------------------
# Fetch tests
# ---------------------------------------------------------------------------

class TestFetch:
    def _patch_client(self, response: FakeResponse):
        client = FakeClient(response)
        return patch("bioscancast.extraction.fetcher.httpx.Client", return_value=client)

    def test_successful_html_fetch(self):
        resp = FakeResponse(
            status_code=200,
            headers={"content-type": "text/html; charset=utf-8"},
            chunks=[b"<html><body>Hello world</body></html>"],
            url="https://example.com/page",
        )
        with self._patch_client(resp):
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
        with self._patch_client(resp):
            result = fetch("https://example.com/report")

        assert result.content_type == "application/pdf"

    def test_content_type_sniff_octet_stream(self):
        resp = FakeResponse(
            status_code=200,
            headers={"content-type": "application/octet-stream"},
            chunks=[b"<!DOCTYPE html><html><body>Hi</body></html>"],
        )
        with self._patch_client(resp):
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
        with self._patch_client(resp):
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
        with self._patch_client(resp):
            result = fetch("https://example.com/page", config=config)

        assert result.error is not None
        assert "exceeded" in result.error
        assert result.content_bytes is None

    def test_network_error_returns_fetch_result(self):
        with patch(
            "bioscancast.extraction.fetcher.httpx.Client",
            side_effect=httpx.ConnectError("Connection refused"),
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
        with self._patch_client(resp):
            result = fetch("https://example.com/old-page")

        assert result.final_url == "https://example.com/final-page"
        assert result.url == "https://example.com/old-page"

    def test_fetched_at_is_utc(self):
        resp = FakeResponse(
            status_code=200,
            headers={"content-type": "text/html"},
            chunks=[b"<html></html>"],
        )
        with self._patch_client(resp):
            result = fetch("https://example.com/page")

        assert result.fetched_at.tzinfo is not None
