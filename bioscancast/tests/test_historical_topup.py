"""Tests for the historical-mode year-hint and top-up behavior in
SearchStagePipeline.
"""

from datetime import datetime, timezone
from typing import List

from bioscancast.filtering.models import ForecastQuestion
from bioscancast.stages.search_stage.backends.base import RawSearchResult
from bioscancast.stages.search_stage.pipeline import SearchStagePipeline


class _FakeLLM:
    def __init__(self):
        self._calls = 0

    def generate_json(self, prompt: str) -> dict:
        self._calls += 1
        if self._calls == 1:
            return {"question_type": "outbreak_count"}
        return {
            "sub_queries": [
                {"text": "H5N1 cases", "axis": "latest_data"},
                {"text": "avian flu trend", "axis": "trend"},
                {"text": "bird flu policy", "axis": "policy"},
            ]
        }


class _RecordingBackend:
    """Records every (query, max_results) it was called with and returns a
    canned mapping. Same URL can appear across calls — pipeline dedup
    handles that."""

    def __init__(self, results_by_query: dict[tuple[str, int], List[RawSearchResult]] | None = None):
        self.calls: list[tuple[str, int]] = []
        self._results = results_by_query or {}
        # Fallback results for any query not explicitly mapped.
        self._fallback: List[RawSearchResult] = []

    def set_fallback(self, results: List[RawSearchResult]) -> None:
        self._fallback = results

    def search(self, query, max_results=10, end_date=None, start_date=None):
        self.calls.append((query, max_results))
        # Prefer exact match on (query, max_results); else any match on
        # query; else fallback.
        if (query, max_results) in self._results:
            return list(self._results[(query, max_results)])
        for (q, _), res in self._results.items():
            if q == query:
                return list(res)
        return list(self._fallback)


def _question(as_of: datetime | None) -> ForecastQuestion:
    return ForecastQuestion(
        id="Q-TU",
        text="H5N1 outbreak in 2024",
        created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        pathogen="nopathogen",  # skip dashboard lookup
        as_of_date=as_of,
    )


def test_year_hint_appended_in_historical_mode():
    backend = _RecordingBackend()
    backend.set_fallback(
        [
            RawSearchResult(
                url="https://news.example.com/a",
                title="A",
                snippet="",
                rank=1,
                published_date="2024-01-01",
            )
        ]
    )
    pipeline = SearchStagePipeline(
        search_backend=backend, llm_client=_FakeLLM(), backend_name="fake"
    )
    pipeline.run(_question(datetime(2024, 6, 1, tzinfo=timezone.utc)))
    # Every query the backend saw should end in " 2024"
    assert backend.calls, "backend should have been called"
    queries = [q for q, _ in backend.calls]
    assert all(q.endswith(" 2024") for q in queries), queries


def test_year_hint_skipped_in_live_mode():
    backend = _RecordingBackend()
    backend.set_fallback(
        [
            RawSearchResult(
                url="https://news.example.com/a",
                title="A",
                snippet="",
                rank=1,
                published_date=None,
            )
        ]
    )
    pipeline = SearchStagePipeline(
        search_backend=backend, llm_client=_FakeLLM(), backend_name="fake"
    )
    pipeline.run(_question(as_of=None))
    queries = [q for q, _ in backend.calls]
    assert not any(q.endswith(" 2024") for q in queries), queries


def test_year_hint_not_double_appended_if_already_present():
    # If the LLM's sub-query already mentions the year, don't append it again.
    class _LLM:
        def __init__(self):
            self._n = 0

        def generate_json(self, prompt: str) -> dict:
            self._n += 1
            if self._n == 1:
                return {"question_type": "outbreak_count"}
            return {
                "sub_queries": [
                    {"text": "H5N1 cases 2024", "axis": "latest_data"},
                ]
            }

    backend = _RecordingBackend()
    backend.set_fallback(
        [
            RawSearchResult(
                url="https://news.example.com/a",
                title="A",
                snippet="",
                rank=1,
                published_date="2024-01-01",
            )
        ]
    )
    pipeline = SearchStagePipeline(
        search_backend=backend, llm_client=_LLM(), backend_name="fake"
    )
    pipeline.run(_question(datetime(2024, 6, 1, tzinfo=timezone.utc)))
    queries = [q for q, _ in backend.calls]
    # Should NOT be "H5N1 cases 2024 2024"
    assert all(q.count("2024") == 1 for q in queries), queries


def test_top_up_fires_when_survivors_below_threshold():
    """First round returns mostly post-cutoff (so few survive); top-up
    round with bigger max_results returns extras that include pre-cutoff
    items. The backend should be called once per sub-query per round."""
    as_of = datetime(2024, 6, 1, tzinfo=timezone.utc)

    # Build the per-query result sets.
    round1 = [
        RawSearchResult(
            url=f"https://news.example.com/post-{i}",
            title="post",
            snippet="",
            rank=i,
            published_date="2024-09-01",  # post-cutoff
        )
        for i in range(3)
    ]
    round2 = round1 + [
        RawSearchResult(
            url=f"https://news.example.com/pre-{i}",
            title="pre",
            snippet="",
            rank=i + 10,
            published_date="2024-01-01",  # pre-cutoff
        )
        for i in range(20)
    ]

    backend = _RecordingBackend()
    # Three sub-queries from _FakeLLM each get year-hinted:
    for query in ("H5N1 cases 2024", "avian flu trend 2024", "bird flu policy 2024"):
        backend._results[(query, 10)] = round1
        backend._results[(query, 50)] = round2

    pipeline = SearchStagePipeline(
        search_backend=backend,
        llm_client=_FakeLLM(),
        backend_name="fake",
        min_post_filter_results=10,
        top_up_results_per_query=50,
        max_top_up_rounds=1,
    )
    results = pipeline.run(_question(as_of))

    # Each sub-query should have been called twice: once with max=10 and once
    # with max=50.
    max_results_seen = [m for _, m in backend.calls]
    assert 10 in max_results_seen
    assert 50 in max_results_seen
    # After top-up we should have well over the threshold of pre-cutoff results.
    assert len(results) >= 10


def test_top_up_skipped_when_survivors_meet_threshold():
    """If the initial round already returns enough survivors, no top-up."""
    as_of = datetime(2024, 6, 1, tzinfo=timezone.utc)
    plenty = [
        RawSearchResult(
            url=f"https://news.example.com/x-{i}",
            title=f"x{i}",
            snippet="",
            rank=i,
            published_date="2024-01-01",
        )
        for i in range(20)
    ]
    backend = _RecordingBackend()
    backend.set_fallback(plenty)
    pipeline = SearchStagePipeline(
        search_backend=backend,
        llm_client=_FakeLLM(),
        backend_name="fake",
        min_post_filter_results=10,
        top_up_results_per_query=50,
        max_top_up_rounds=1,
    )
    pipeline.run(_question(as_of))
    # Only the initial round (max=10) should have fired.
    max_results_seen = {m for _, m in backend.calls}
    assert max_results_seen == {10}


def test_top_up_skipped_in_live_mode():
    """Live mode never tops up, even when result count is low."""
    backend = _RecordingBackend()
    backend.set_fallback(
        [
            RawSearchResult(
                url="https://news.example.com/only",
                title="only",
                snippet="",
                rank=1,
                published_date=None,
            )
        ]
    )
    pipeline = SearchStagePipeline(
        search_backend=backend,
        llm_client=_FakeLLM(),
        backend_name="fake",
        min_post_filter_results=10,
        top_up_results_per_query=50,
        max_top_up_rounds=2,
    )
    pipeline.run(_question(as_of=None))
    max_results_seen = {m for _, m in backend.calls}
    assert max_results_seen == {10}
