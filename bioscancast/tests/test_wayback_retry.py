"""Retry/backoff behavior for the Wayback CDX client."""

from __future__ import annotations

import socket
import urllib.error
from io import BytesIO
from unittest.mock import patch

from bioscancast.stages.search_stage import wayback


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://web.archive.org/cdx/search/cdx",
        code=code,
        msg=str(code),
        hdrs=None,  # type: ignore[arg-type]
        fp=None,
    )


def _ok_response(payload: bytes):
    """A minimal stand-in for the context manager returned by urlopen."""

    class _CM:
        def __enter__(self):
            return BytesIO(payload)

        def __exit__(self, *a):
            return False

    return _CM()


class TestCdxRetry:
    def _no_sleep(self):
        return patch.object(wayback, "_sleep", lambda _s: None)

    def _short_schedule(self):
        # 3 attempts max so tests are predictable; all delays are no-ops.
        return patch.object(wayback, "RETRY_BACKOFF_SECONDS", (0, 0, 0))

    def test_retries_then_succeeds_on_503(self):
        # First two calls 503, third returns valid JSON.
        seq = [
            _http_error(503),
            _http_error(503),
            _ok_response(b'[["urlkey","timestamp","original"],["a","20240101120000","b"]]'),
        ]
        with self._short_schedule(), self._no_sleep(), patch(
            "bioscancast.stages.search_stage.wayback.urllib.request.urlopen",
            side_effect=seq,
        ):
            data = wayback._cdx_query({"url": "https://example.com/"})
        assert data is not None
        assert data == [["a", "20240101120000", "b"]]

    def test_gives_up_after_max_attempts_503(self):
        with self._short_schedule(), self._no_sleep(), patch(
            "bioscancast.stages.search_stage.wayback.urllib.request.urlopen",
            side_effect=[_http_error(503)] * 3,
        ):
            data = wayback._cdx_query({"url": "https://example.com/"})
        assert data is None

    def test_retries_on_timeout(self):
        seq = [
            socket.timeout("read timeout"),
            _ok_response(b'[["urlkey","timestamp","original"]]'),
        ]
        with self._short_schedule(), self._no_sleep(), patch(
            "bioscancast.stages.search_stage.wayback.urllib.request.urlopen",
            side_effect=seq,
        ):
            data = wayback._cdx_query({"url": "https://example.com/"})
        # Header-only payload → empty rows list
        assert data == []

    def test_non_recoverable_status_does_not_retry(self):
        # 404 should fail immediately with no retries
        with self._short_schedule(), self._no_sleep(), patch(
            "bioscancast.stages.search_stage.wayback.urllib.request.urlopen",
            side_effect=[_http_error(404)],
        ) as mock_open:
            data = wayback._cdx_query({"url": "https://example.com/"})
        assert data is None
        assert mock_open.call_count == 1

    def test_recoverable_statuses_cover_5xx_and_429(self):
        # 429 is rate-limit; should be treated as recoverable.
        seq = [
            _http_error(429),
            _ok_response(b'[["header"]]'),
        ]
        with self._short_schedule(), self._no_sleep(), patch(
            "bioscancast.stages.search_stage.wayback.urllib.request.urlopen",
            side_effect=seq,
        ):
            data = wayback._cdx_query({"url": "https://example.com/"})
        assert data == []
