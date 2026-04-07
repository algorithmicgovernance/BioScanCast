from datetime import datetime, timezone

from bioscancast.filtering.models import ForecastQuestion
from bioscancast.stages.search_stage.dashboard_lookup import lookup_dashboards


def _make_question(**overrides):
    defaults = {
        "id": "Q001",
        "text": "Will H5N1 cause more than 100 human cases?",
        "created_at": datetime(2025, 1, 15, tzinfo=timezone.utc),
    }
    defaults.update(overrides)
    return ForecastQuestion(**defaults)


class TestDashboardLookup:
    def test_known_pathogen_returns_results(self):
        q = _make_question(pathogen="h5n1")
        results = lookup_dashboards(q)
        assert len(results) > 0
        for r in results:
            assert r.retrieval_reason == "dashboard_lookup"
            assert r.rank == 0
            assert r.engine == "dashboard"
            assert r.question_id == "Q001"
            assert r.query_id == "dashboard_Q001"

    def test_mpox_returns_results(self):
        q = _make_question(pathogen="mpox")
        results = lookup_dashboards(q)
        assert len(results) > 0

    def test_unknown_pathogen_returns_empty(self):
        q = _make_question(pathogen="unknownvirus123")
        results = lookup_dashboards(q)
        assert results == []

    def test_no_pathogen_returns_empty(self):
        q = _make_question(pathogen=None)
        results = lookup_dashboards(q)
        assert results == []

    def test_case_insensitive(self):
        q = _make_question(pathogen="H5N1")
        results = lookup_dashboards(q)
        assert len(results) > 0

    def test_results_have_required_fields(self):
        q = _make_question(pathogen="ebola")
        results = lookup_dashboards(q)
        assert len(results) > 0
        for r in results:
            assert r.url is not None
            assert r.canonical_url is not None
            assert r.domain != ""
            assert r.source_tier in {"official", "academic", "trusted_media", "ngo", "unknown"}
            assert 0.0 <= r.domain_score <= 1.0
            assert r.freshness_score == 1.0
            assert r.search_stage_score == 0.0  # computed later by pipeline
