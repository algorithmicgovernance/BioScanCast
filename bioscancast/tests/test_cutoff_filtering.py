"""End-to-end-ish tests for historical-replay mode in SearchStagePipeline.

Uses the same FakeLLMClient/FakeSearchBackend pattern as test_search_pipeline.py
to keep the test layer hand-rolled and dependency-free.
"""

from datetime import datetime, timezone
from typing import List
from unittest.mock import patch

import pytest

from bioscancast.stages.filtering.models import ForecastQuestion, SearchResult
from bioscancast.stages.searching.backends.base import RawSearchResult
from bioscancast.stages.searching.pipeline import (
    SearchStagePipeline,
    _parse_published_date,
    _should_use_wayback_for_recovery,
)


@pytest.fixture(autouse=True)
def _stub_source_lookup_wayback(monkeypatch):
    """Injected YAML sources call closest_snapshot_before in historical mode;
    stub it so these tests never hit the live archive.org network. Returning
    None suppresses the injected source (the 'no pre-cutoff snapshot' path)."""
    monkeypatch.setattr(
        "bioscancast.stages.searching.source_lookup.closest_snapshot_before",
        lambda *args, **kwargs: None,
    )


class TestParsePublishedDate:
    def test_iso_with_offset(self):
        assert _parse_published_date("2025-02-17T13:00:00+00:00") == datetime(
            2025, 2, 17, 13, 0, 0, tzinfo=timezone.utc
        )

    def test_iso_date_only(self):
        assert _parse_published_date("2025-02-17") == datetime(
            2025, 2, 17, tzinfo=timezone.utc
        )

    def test_rfc2822_with_zone(self):
        # The format Tavily's news topic actually returns.
        result = _parse_published_date("Tue, 19 May 2026 13:00:00 GMT")
        assert result is not None
        assert result.year == 2026
        assert result.month == 5
        assert result.day == 19
        assert result.tzinfo is not None

    def test_rfc2822_with_offset(self):
        result = _parse_published_date("Tue, 19 May 2026 13:00:00 +0000")
        assert result is not None
        assert result.day == 19

    def test_none_and_empty(self):
        assert _parse_published_date(None) is None
        assert _parse_published_date("") is None

    def test_garbage_returns_none(self):
        assert _parse_published_date("not a date") is None


class _FakeLLM:
    def __init__(self):
        self._calls = 0

    def generate_json(self, prompt: str) -> dict:
        self._calls += 1
        if self._calls == 1:
            return {"question_type": "outbreak_count"}
        return {
            "sub_queries": [
                {"text": "H5N1 cases 2024", "axis": "latest_data"},
                {"text": "avian flu trend", "axis": "trend"},
                {"text": "bird flu policy", "axis": "policy"},
            ]
        }


class _FakeBackend:
    def __init__(self, results: List[RawSearchResult]):
        self._results = results
        self.end_dates_seen: list = []
        self.start_dates_seen: list = []

    def search(self, query, max_results=10, end_date=None, start_date=None):
        self.end_dates_seen.append(end_date)
        self.start_dates_seen.append(start_date)
        return list(self._results)


def _make_question(as_of: datetime | None) -> ForecastQuestion:
    return ForecastQuestion(
        id="Q-CUT",
        text="Will H5N1 exceed 100 cases by end of 2024?",
        created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        pathogen="nopathogen",  # avoid dashboard injection in this test
        as_of_date=as_of,
    )


def test_post_cutoff_results_are_dropped():
    cutoff = datetime(2024, 6, 1, tzinfo=timezone.utc)
    backend_results = [
        RawSearchResult(
            url="https://news.example.com/a",
            title="Pre-cutoff",
            snippet="",
            rank=1,
            published_date="2024-05-15",
        ),
        RawSearchResult(
            url="https://news.example.com/b",
            title="Post-cutoff",
            snippet="",
            rank=2,
            published_date="2024-08-15",
        ),
    ]
    pipeline = SearchStagePipeline(
        search_backend=_FakeBackend(backend_results),
        llm_client=_FakeLLM(),
        backend_name="fake",
    )
    results = pipeline.run(_make_question(cutoff))
    urls = {r.url for r in results}
    assert "https://news.example.com/a" in urls
    assert "https://news.example.com/b" not in urls


def test_undated_dropped_when_recovery_fails():
    cutoff = datetime(2024, 6, 1, tzinfo=timezone.utc)
    backend_results = [
        RawSearchResult(
            url="https://news.example.com/no-date",
            title="Undated",
            snippet="",
            rank=1,
            published_date=None,
        ),
    ]
    pipeline = SearchStagePipeline(
        search_backend=_FakeBackend(backend_results),
        llm_client=_FakeLLM(),
        backend_name="fake",
    )
    with patch(
        "bioscancast.stages.searching.pipeline.recover_published_date"
    ) as mock_rec:
        mock_rec.return_value = (None, None)
        results = pipeline.run(_make_question(cutoff))
    assert not any(r.url == "https://news.example.com/no-date" for r in results)


def test_undated_kept_when_recovery_succeeds_before_cutoff():
    cutoff = datetime(2024, 6, 1, tzinfo=timezone.utc)
    backend_results = [
        RawSearchResult(
            url="https://news.example.com/2024/03/15/article",
            title="Slug-dated",
            snippet="",
            rank=1,
            published_date=None,
        ),
    ]
    pipeline = SearchStagePipeline(
        search_backend=_FakeBackend(backend_results),
        llm_client=_FakeLLM(),
        backend_name="fake",
    )
    results = pipeline.run(_make_question(cutoff))
    matching = [r for r in results if "2024/03/15" in r.url]
    assert len(matching) == 1
    assert matching[0].published_date_source == "url_slug"


def test_end_date_forwarded_to_backend():
    cutoff = datetime(2024, 6, 1, tzinfo=timezone.utc)
    backend = _FakeBackend(
        [
            RawSearchResult(
                url="https://news.example.com/x",
                title="X",
                snippet="",
                rank=1,
                published_date="2024-01-01",
            )
        ]
    )
    pipeline = SearchStagePipeline(
        search_backend=backend, llm_client=_FakeLLM(), backend_name="fake"
    )
    pipeline.run(_make_question(cutoff))
    assert all(d == "2024-06-01" for d in backend.end_dates_seen if d is not None)
    assert any(d == "2024-06-01" for d in backend.end_dates_seen)


def test_live_mode_unchanged():
    backend = _FakeBackend(
        [
            RawSearchResult(
                url="https://news.example.com/x",
                title="X",
                snippet="",
                rank=1,
                published_date=None,
            )
        ]
    )
    pipeline = SearchStagePipeline(
        search_backend=backend, llm_client=_FakeLLM(), backend_name="fake"
    )
    results = pipeline.run(_make_question(as_of=None))
    # Undated result MUST be kept in live mode (the cutoff filter is off)
    assert any(r.url == "https://news.example.com/x" for r in results)
    # And backend received end_date=None AND start_date=None — Tavily ignores
    # end_date when start_date is missing, so the pipeline must keep them
    # both unset in live mode.
    assert all(d is None for d in backend.end_dates_seen)
    assert all(d is None for d in backend.start_dates_seen)


def test_historical_mode_forwards_start_and_end_date_pair():
    """Tavily honors end_date only when start_date is also set. The pipeline
    must synthesize start_date = as_of - historical_lookback_days and pass
    both to the backend on every search call."""
    cutoff = datetime(2024, 6, 1, tzinfo=timezone.utc)
    backend = _FakeBackend(
        [
            RawSearchResult(
                url="https://news.example.com/x",
                title="X",
                snippet="",
                rank=1,
                published_date="2024-01-01",
            )
        ]
    )
    pipeline = SearchStagePipeline(
        search_backend=backend,
        llm_client=_FakeLLM(),
        backend_name="fake",
        historical_lookback_days=365,
    )
    pipeline.run(_make_question(cutoff))
    # Every search call in historical mode must carry BOTH bounds.
    paired = [
        (s, e)
        for s, e in zip(backend.start_dates_seen, backend.end_dates_seen)
        if s is not None or e is not None
    ]
    assert paired, "expected at least one date-bounded search in historical mode"
    for start, end in paired:
        assert start is not None and end is not None, (
            "Tavily ignores end_date alone — pipeline must pass the pair"
        )
        assert end == "2024-06-01"
        assert start == "2023-06-02"  # 365 days before 2024-06-01


def test_historical_lookback_days_is_configurable():
    """Override the default 365-day lookback via the pipeline constructor."""
    cutoff = datetime(2024, 6, 1, tzinfo=timezone.utc)
    backend = _FakeBackend(
        [
            RawSearchResult(
                url="https://news.example.com/x",
                title="X",
                snippet="",
                rank=1,
                published_date="2024-01-01",
            )
        ]
    )
    pipeline = SearchStagePipeline(
        search_backend=backend,
        llm_client=_FakeLLM(),
        backend_name="fake",
        historical_lookback_days=30,
    )
    pipeline.run(_make_question(cutoff))
    starts = [s for s in backend.start_dates_seen if s is not None]
    assert starts and all(s == "2024-05-02" for s in starts)  # 30 days before


def _make_search_result(url: str, source_tier: str = "trusted_media") -> SearchResult:
    return SearchResult(
        id="r1",
        question_id="q1",
        query_id="sq1",
        engine="fake",
        url=url,
        canonical_url=None,
        domain="",
        title="t",
        snippet="",
        rank=1,
        retrieved_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        source_tier=source_tier,
    )


class TestSelectiveRecoveryGate:
    """The Wayback-leg gate on the date-recovery chain."""

    def test_official_tier_uses_wayback(self):
        r = _make_search_result(
            "https://www.cdc.gov/bird-flu/situation-summary/", source_tier="official"
        )
        assert _should_use_wayback_for_recovery(r) is True

    def test_academic_tier_uses_wayback(self):
        r = _make_search_result(
            "https://www.nature.com/articles/xyz", source_tier="academic"
        )
        assert _should_use_wayback_for_recovery(r) is True

    def test_unknown_tier_skips_wayback(self):
        r = _make_search_result(
            "https://obscure-site.example/article", source_tier="unknown"
        )
        assert _should_use_wayback_for_recovery(r) is False

    def test_aggregator_domain_skips_wayback(self):
        # metaculus.com is in AGGREGATOR_DOMAINS regardless of tier label.
        r = _make_search_result(
            "https://www.metaculus.com/questions/12345/",
            source_tier="trusted_media",
        )
        assert _should_use_wayback_for_recovery(r) is False


def test_aggregator_undated_recovery_skips_wayback():
    """End-to-end: an undated aggregator result with no slug date routes to
    recover_published_date with use_wayback=False, so the Wayback leg never
    fires for it."""
    cutoff = datetime(2024, 6, 1, tzinfo=timezone.utc)
    backend_results = [
        RawSearchResult(
            url="https://www.metaculus.com/questions/abc",  # known aggregator
            title="Aggregator forecast",
            snippet="",
            rank=1,
            published_date=None,
        ),
    ]
    pipeline = SearchStagePipeline(
        search_backend=_FakeBackend(backend_results),
        llm_client=_FakeLLM(),
        backend_name="fake",
    )
    with patch(
        "bioscancast.stages.searching.pipeline.recover_published_date",
        return_value=(None, None),
    ) as mock_rec:
        pipeline.run(_make_question(cutoff))
    # The recovery function was called, but with use_wayback=False.
    assert mock_rec.called
    # At least one of the calls was for the aggregator URL with use_wayback=False.
    aggregator_calls = [
        c for c in mock_rec.call_args_list
        if c.args and "metaculus.com" in c.args[0]
    ]
    assert aggregator_calls
    for call in aggregator_calls:
        assert call.kwargs.get("use_wayback") is False


def test_official_undated_recovery_still_tries_wayback():
    """A tier-1 official domain with no slug date should still hit the
    Wayback leg of recovery (i.e., use_wayback=True)."""
    cutoff = datetime(2024, 6, 1, tzinfo=timezone.utc)
    backend_results = [
        RawSearchResult(
            url="https://www.cdc.gov/some/article",  # tier 1 official
            title="CDC article",
            snippet="",
            rank=1,
            published_date=None,
        ),
    ]
    pipeline = SearchStagePipeline(
        search_backend=_FakeBackend(backend_results),
        llm_client=_FakeLLM(),
        backend_name="fake",
    )
    with patch(
        "bioscancast.stages.searching.pipeline.recover_published_date",
        return_value=(None, None),
    ) as mock_rec:
        pipeline.run(_make_question(cutoff))
    cdc_calls = [
        c for c in mock_rec.call_args_list
        if c.args and "cdc.gov" in c.args[0]
    ]
    assert cdc_calls
    for call in cdc_calls:
        assert call.kwargs.get("use_wayback") is True


def test_cutoff_applied_persisted_on_results():
    cutoff = datetime(2024, 6, 1, tzinfo=timezone.utc)
    backend = _FakeBackend(
        [
            RawSearchResult(
                url="https://news.example.com/x",
                title="X",
                snippet="",
                rank=1,
                published_date="2024-01-01",
            )
        ]
    )
    pipeline = SearchStagePipeline(
        search_backend=backend, llm_client=_FakeLLM(), backend_name="fake"
    )
    results = pipeline.run(_make_question(cutoff))
    assert results
    for r in results:
        assert r.cutoff_applied == cutoff
