"""Retry/backoff behavior for the Wayback CDX client."""

from __future__ import annotations

import socket
import urllib.error
from io import BytesIO
from unittest.mock import patch

from bioscancast.stages.searching import wayback


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
            "bioscancast.stages.searching.wayback.urllib.request.urlopen",
            side_effect=seq,
        ):
            data = wayback._cdx_query({"url": "https://example.com/"})
        assert data is not None
        assert data == [["a", "20240101120000", "b"]]

    def test_gives_up_after_max_attempts_503(self):
        with self._short_schedule(), self._no_sleep(), patch(
            "bioscancast.stages.searching.wayback.urllib.request.urlopen",
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
            "bioscancast.stages.searching.wayback.urllib.request.urlopen",
            side_effect=seq,
        ):
            data = wayback._cdx_query({"url": "https://example.com/"})
        # Header-only payload → empty rows list
        assert data == []

    def test_non_recoverable_status_does_not_retry(self):
        # 404 should fail immediately with no retries
        with self._short_schedule(), self._no_sleep(), patch(
            "bioscancast.stages.searching.wayback.urllib.request.urlopen",
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
            "bioscancast.stages.searching.wayback.urllib.request.urlopen",
            side_effect=seq,
        ):
            data = wayback._cdx_query({"url": "https://example.com/"})
        assert data == []


class TestCdxThrottle:
    """Proactive min-interval pacing in front of every urlopen."""

    def test_throttle_paces_successive_calls(self):
        sleep_calls: list[float] = []
        ok = b'[["urlkey","timestamp","original"],["a","20240101120000","b"]]'
        with patch.object(wayback, "_last_call_monotonic", 0.0), patch.object(
            wayback, "_min_interval_seconds", lambda: 5.0
        ), patch.object(wayback, "_sleep", lambda s: sleep_calls.append(s)), patch.object(
            wayback, "RETRY_BACKOFF_SECONDS", (0,)
        ), patch(
            "bioscancast.stages.searching.wayback.urllib.request.urlopen",
            side_effect=[_ok_response(ok), _ok_response(ok)],
        ):
            wayback._cdx_query({"url": "https://example.com/a"})
            wayback._cdx_query({"url": "https://example.com/b"})
        positive_waits = [s for s in sleep_calls if s > 0]
        assert len(positive_waits) == 1
        assert 4.0 < positive_waits[0] <= 5.0

    def test_throttle_fires_before_each_retry(self):
        # Throttle paces before every urlopen — including retried ones — so a
        # 503 → OK sequence yields two _throttle() calls, the second of which
        # sleeps because the first urlopen just bumped _last_call_monotonic.
        sleep_calls: list[float] = []
        ok = b'[["urlkey","timestamp","original"]]'
        with patch.object(wayback, "_last_call_monotonic", 0.0), patch.object(
            wayback, "_min_interval_seconds", lambda: 3.0
        ), patch.object(wayback, "_sleep", lambda s: sleep_calls.append(s)), patch.object(
            wayback, "RETRY_BACKOFF_SECONDS", (0, 0, 0)
        ), patch(
            "bioscancast.stages.searching.wayback.urllib.request.urlopen",
            side_effect=[_http_error(503), _ok_response(ok)],
        ):
            data = wayback._cdx_query({"url": "https://example.com/"})
        assert data == []
        positive_waits = [s for s in sleep_calls if s > 0]
        assert len(positive_waits) == 1
        assert 2.0 < positive_waits[0] <= 3.0

    def test_min_interval_env_override(self, monkeypatch):
        monkeypatch.setenv("BIOSCANCAST_WAYBACK_MIN_INTERVAL_SECONDS", "1.5")
        assert wayback._min_interval_seconds() == 1.5
        monkeypatch.setenv("BIOSCANCAST_WAYBACK_MIN_INTERVAL_SECONDS", "not-a-number")
        assert wayback._min_interval_seconds() == wayback._DEFAULT_MIN_INTERVAL_SECONDS
        monkeypatch.delenv("BIOSCANCAST_WAYBACK_MIN_INTERVAL_SECONDS", raising=False)
        assert wayback._min_interval_seconds() == wayback._DEFAULT_MIN_INTERVAL_SECONDS
