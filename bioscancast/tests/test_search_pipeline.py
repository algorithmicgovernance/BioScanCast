from datetime import datetime, timezone
from typing import List

from bioscancast.filtering.models import ForecastQuestion, SearchResult
from bioscancast.stages.search_stage.backends.base import RawSearchResult
from bioscancast.stages.search_stage.pipeline import SearchStagePipeline


class FakeLLMClient:
    """Returns canned classification + decomposition responses."""

    def __init__(self):
        self._calls = 0

    def generate_json(self, prompt: str) -> dict:
        self._calls += 1
        if self._calls == 1:
            return {"question_type": "outbreak_count"}
        return {
            "sub_queries": [
                {"text": "H5N1 human cases 2025", "axis": "latest_data"},
                {"text": "avian influenza trend US", "axis": "trend"},
                {"text": "bird flu government policy", "axis": "policy"},
                {"text": "H5N1 historical outbreak data", "axis": "historical_analogy"},
                {"text": "avian flu latest CDC report", "axis": "latest_data"},
            ]
        }


class FakeSearchBackend:
    """Returns a fixed set of results for any query."""

    def __init__(self, results: List[RawSearchResult] | None = None):
        self._results = results or self._default_results()
        self.queries_received: list[str] = []

    @staticmethod
    def _default_results() -> List[RawSearchResult]:
        return [
            RawSearchResult(
                url="https://www.cdc.gov/bird-flu/situation-summary/",
                title="CDC Bird Flu Situation Summary",
                snippet="Current H5N1 situation in the US",
                rank=1,
                published_date="2025-06-01",
            ),
            RawSearchResult(
                url="https://www.who.int/news/h5n1-update",
                title="WHO H5N1 Update",
                snippet="Global avian influenza situation",
                rank=2,
                published_date="2025-05-15",
            ),
            RawSearchResult(
                url="https://www.nature.com/articles/h5n1-study",
                title="H5N1 Transmission Study",
                snippet="Research on avian influenza transmission",
                rank=3,
            ),
            RawSearchResult(
                url="https://metaculus.com/questions/h5n1",
                title="Metaculus H5N1 Forecast",
                snippet="Community forecast on H5N1 cases",
                rank=4,
            ),
            RawSearchResult(
                url="https://facebook.com/health-post",
                title="Some Health Post",
                snippet="Social media health content",
                rank=5,
            ),
            # Duplicate of CDC with tracking params
            RawSearchResult(
                url="https://www.cdc.gov/bird-flu/situation-summary/?utm_source=twitter",
                title="CDC Bird Flu Situation Summary",
                snippet="Current H5N1 situation in the US (shared)",
                rank=6,
            ),
        ]

    def search(self, query: str, max_results: int = 10) -> List[RawSearchResult]:
        self.queries_received.append(query)
        return self._results


def _make_question():
    return ForecastQuestion(
        id="Q001",
        text="Will H5N1 cause more than 100 human cases in the US by December 2026?",
        created_at=datetime(2025, 1, 15, tzinfo=timezone.utc),
        pathogen="H5N1",
        region="United States",
    )


class TestSearchStagePipeline:
    def _run_pipeline(self):
        pipeline = SearchStagePipeline(
            search_backend=FakeSearchBackend(),
            llm_client=FakeLLMClient(),
            backend_name="fake",
        )
        return pipeline.run(_make_question())

    def test_returns_list_of_search_results(self):
        results = self._run_pipeline()
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_output_contract_fields_populated(self):
        results = self._run_pipeline()
        for r in results:
            assert r.id is not None
            assert r.question_id == "Q001"
            assert r.query_id is not None
            assert r.engine is not None
            assert r.url is not None
            assert r.canonical_url is not None
            assert r.domain != ""
            assert r.title is not None
            assert r.source_tier in {"official", "academic", "trusted_media", "ngo", "unknown"}
            assert 0.0 <= r.domain_score <= 1.0
            assert 0.0 <= r.freshness_score <= 1.0
            assert 0.0 <= r.search_stage_score <= 1.0
            assert r.retrieval_reason is not None

    def test_blocked_domains_excluded(self):
        results = self._run_pipeline()
        domains = {r.domain for r in results}
        assert "facebook.com" not in domains

    def test_deduplication_by_canonical_url(self):
        results = self._run_pipeline()
        canonical_urls = [r.canonical_url for r in results]
        assert len(canonical_urls) == len(set(canonical_urls))

    def test_aggregator_flagged(self):
        results = self._run_pipeline()
        metaculus_results = [r for r in results if "metaculus" in r.domain]
        assert len(metaculus_results) > 0
        for r in metaculus_results:
            assert r.contains_aggregator_forecast is True

    def test_non_aggregator_not_flagged(self):
        results = self._run_pipeline()
        cdc_results = [r for r in results if "cdc.gov" in r.domain]
        for r in cdc_results:
            assert r.contains_aggregator_forecast is False

    def test_sorted_by_score_descending(self):
        results = self._run_pipeline()
        scores = [r.search_stage_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_total_cap_enforced(self):
        results = self._run_pipeline()
        assert len(results) <= 60

    def test_scoring_formula(self):
        """Verify the search_stage_score formula for a known result."""
        results = self._run_pipeline()
        for r in results:
            expected = 0.5 * r.domain_score + 0.3 * r.freshness_score + 0.2 * (1.0 / max(r.rank, 1))
            expected = max(0.0, min(1.0, expected))
            assert abs(r.search_stage_score - expected) < 1e-9, (
                f"Score mismatch for {r.url}: {r.search_stage_score} != {expected}"
            )

    def test_dashboard_results_included(self):
        """H5N1 question should inject dashboard lookup results."""
        results = self._run_pipeline()
        dashboard_results = [r for r in results if r.retrieval_reason and "dashboard_lookup" in r.retrieval_reason]
        assert len(dashboard_results) > 0
        for r in dashboard_results:
            assert r.rank == 0
            assert r.engine == "dashboard"
