"""Unit tests for TavilyBackend's date-window forwarding.

The Tavily news endpoint silently ignores ``end_date`` unless ``start_date``
is also passed (verified 2026-05-20, see
``specs/tavily-investigation-findings.md``). The backend's job is to forward
the pair when both are present, drop ``end_date`` alone with a warning,
and call the SDK with no date params otherwise.
"""

from __future__ import annotations

from typing import Any

import pytest

from bioscancast.stages.searching.backends.tavily_backend import TavilyBackend


class _FakeTavilyClient:
    """Captures the kwargs of every ``search`` call so tests can assert on them."""

    def __init__(self, *_args, **_kwargs):
        self.calls: list[dict[str, Any]] = []
        _FakeTavilyClient.last_instance = self

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return {"results": []}


@pytest.fixture
def fake_tavily(monkeypatch):
    """Patch tavily.TavilyClient so no network call is made."""
    import tavily

    monkeypatch.setattr(tavily, "TavilyClient", _FakeTavilyClient)
    yield _FakeTavilyClient


def test_forwards_start_and_end_date_pair(fake_tavily):
    backend = TavilyBackend(api_key="test-key")
    backend.search(
        "H5N1 cases", max_results=5,
        start_date="2024-01-01", end_date="2025-02-17",
    )
    call = fake_tavily.last_instance.calls[-1]
    assert call["start_date"] == "2024-01-01"
    assert call["end_date"] == "2025-02-17"
    assert call["topic"] == "news"
    assert call["max_results"] == 5


def test_drops_end_date_when_start_date_missing(fake_tavily, caplog):
    """Tavily ignores end_date alone — sending it would mislead anyone reading
    the request log. The backend logs a warning and omits both."""
    backend = TavilyBackend(api_key="test-key")
    with caplog.at_level("WARNING"):
        backend.search("Mpox cases", end_date="2025-02-17")
    call = fake_tavily.last_instance.calls[-1]
    assert "end_date" not in call
    assert "start_date" not in call
    assert any("end_date" in rec.message and "start_date" in rec.message
               for rec in caplog.records), (
        "expected a warning when end_date is passed without start_date"
    )


def test_no_date_params_in_live_mode(fake_tavily):
    backend = TavilyBackend(api_key="test-key")
    backend.search("H5N1 cases")
    call = fake_tavily.last_instance.calls[-1]
    assert "start_date" not in call
    assert "end_date" not in call
    assert call["topic"] == "news"


def test_start_date_without_end_date_is_also_dropped(fake_tavily):
    """The pair must be complete; lone start_date is also ignored upstream."""
    backend = TavilyBackend(api_key="test-key")
    backend.search("H5N1 cases", start_date="2024-01-01")
    call = fake_tavily.last_instance.calls[-1]
    assert "start_date" not in call
    assert "end_date" not in call
