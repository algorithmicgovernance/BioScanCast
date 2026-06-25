"""Offline tests for the Wayback rewrite in the extraction fetcher.

The patching reaches into ``bioscancast.extraction.fetcher.closest_snapshot_before``
(the symbol imported at module load) and ``curl_requests.get``, never touching
the network. There is also a ``@pytest.mark.live`` smoke test for hitting
Wayback for real.
"""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from bioscancast.stages.extraction.fetcher import fetch
from bioscancast.stages.searching.wayback import closest_snapshot_before


class _FakeResponse:
    def __init__(self, *, body: bytes, url: str, status: int = 200):
        self.status_code = status
        self.headers = {"content-type": "text/html"}
        self.url = url
        self._body = body

    def iter_content(self):
        yield self._body

    def close(self):
        pass


def _patch_curl(body: bytes, url: str = "https://example.com/page"):
    return patch(
        "bioscancast.extraction.fetcher.curl_requests.get",
        return_value=_FakeResponse(body=body, url=url),
    )


def _patch_snapshot(value):
    return patch(
        "bioscancast.extraction.fetcher.closest_snapshot_before",
        return_value=value,
    )


class TestWaybackRewrite:
    def test_live_mode_no_wayback_call(self):
        with _patch_curl(b"<html>live</html>") as mock_get, _patch_snapshot(None) as mock_snap:
            result = fetch("https://example.com/page", as_of_date=None)
        assert result.fetch_strategy == "live"
        assert result.snapshot_timestamp is None
        mock_snap.assert_not_called()
        mock_get.assert_called_once()

    def test_wayback_success(self):
        snap_dt = datetime(2024, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
        snap_url = "https://web.archive.org/web/20240301120000id_/https://example.com/page"
        with _patch_snapshot((snap_dt, snap_url)), _patch_curl(b"<html>snapshot</html>"):
            result = fetch(
                "https://example.com/page",
                as_of_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
            )
        assert result.fetch_strategy == "wayback"
        assert result.snapshot_timestamp == snap_dt
        assert result.url == "https://example.com/page"  # original, not archive.org
        assert result.content_bytes == b"<html>snapshot</html>"

    def test_no_snapshot_falls_back_to_live(self):
        with _patch_snapshot(None), _patch_curl(b"<html>live</html>"):
            result = fetch(
                "https://example.com/page",
                as_of_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
            )
        assert result.fetch_strategy == "wayback_fallback_to_live"
        assert result.snapshot_timestamp is None
        assert result.url == "https://example.com/page"

    def test_wayback_fetch_error_falls_back_to_live(self):
        snap_dt = datetime(2024, 3, 1, tzinfo=timezone.utc)
        snap_url = "https://web.archive.org/web/20240301120000id_/https://example.com/page"
        # First call (to Wayback) errors; second call (live) succeeds.
        responses = [
            ConnectionError("wayback down"),
            _FakeResponse(body=b"<html>live</html>", url="https://example.com/page"),
        ]

        def side_effect(*args, **kwargs):
            r = responses.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        with _patch_snapshot((snap_dt, snap_url)), patch(
            "bioscancast.extraction.fetcher.curl_requests.get", side_effect=side_effect
        ):
            result = fetch(
                "https://example.com/page",
                as_of_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
            )
        assert result.fetch_strategy == "wayback_fallback_to_live"
        assert result.content_bytes == b"<html>live</html>"


@pytest.mark.live
def test_live_wayback_lookup():
    """Smoke-test the real Wayback CDX endpoint. Skipped by default."""

    result = closest_snapshot_before(
        "https://www.cdc.gov/",
        datetime(2023, 1, 1, tzinfo=timezone.utc),
    )
    assert result is not None
    snap_dt, snap_url = result
    assert snap_dt < datetime(2023, 1, 2, tzinfo=timezone.utc)
    assert "web.archive.org/web/" in snap_url
