"""End-to-end-ish tests for historical-replay mode in SearchStagePipeline.

Uses the same FakeLLMClient/FakeSearchBackend pattern as test_search_pipeline.py
to keep the test layer hand-rolled and dependency-free.
"""

from datetime import datetime, timezone
from typing import List
from unittest.mock import patch

from bioscancast.filtering.models import ForecastQuestion
from bioscancast.stages.search_stage.backends.base import RawSearchResult
from bioscancast.stages.search_stage.pipeline import (
    SearchStagePipeline,
    _parse_published_date,
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
        "bioscancast.stages.search_stage.pipeline.recover_published_date"
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
