"""Integration test: feed search stage output into filtering pipeline.

Verifies contract compatibility between SearchStagePipeline output and
FilteringPipeline input — no manually constructed SearchResult objects.
"""

from collections import Counter
from typing import List

from bioscancast.filtering.config import FILTER_CONFIG
from bioscancast.filtering.models import FilteredDocument
from bioscancast.filtering.pipeline import FilteringPipeline
from bioscancast.stages.search_stage.backends.base import RawSearchResult
from bioscancast.stages.search_stage.pipeline import SearchStagePipeline

from bioscancast.tests.test_search_pipeline import FakeLLMClient


class RealisticFakeSearchBackend:
    """Returns results with titles/snippets that overlap with the H5N1 question,
    simulating what a real search engine would return."""

    def search(self, query: str, max_results: int = 10) -> List[RawSearchResult]:
        return [
            RawSearchResult(
                url="https://www.cdc.gov/bird-flu/situation-summary/",
                title="H5N1 Bird Flu: Current Human Cases in the United States",
                snippet="CDC reports confirmed human H5N1 avian influenza cases in the US.",
                rank=1,
                published_date="2026-03-20",
            ),
            RawSearchResult(
                url="https://www.who.int/news/h5n1-update",
                title="WHO Update on Human H5N1 Avian Influenza Cases",
                snippet="Global situation report on H5N1 cases and public health response.",
                rank=2,
                published_date="2026-03-15",
            ),
            RawSearchResult(
                url="https://www.nature.com/articles/h5n1-study",
                title="H5N1 Transmission Study in Human Populations",
                snippet="Research on avian influenza H5N1 transmission risk to humans.",
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
                title="H5N1 Bird Flu: Current Human Cases in the United States",
                snippet="CDC reports confirmed human H5N1 avian influenza cases in the US.",
                rank=6,
            ),
        ]


def _make_question():
    """Question with no known pathogen so dashboard injection doesn't overwrite
    realistic search results with generic dashboard titles."""
    from datetime import datetime, timezone

    from bioscancast.filtering.models import ForecastQuestion

    return ForecastQuestion(
        id="Q001",
        text="Will H5N1 cause more than 100 human cases in the US by December 2026?",
        created_at=datetime(2025, 1, 15, tzinfo=timezone.utc),
        pathogen="h5n1-x",  # unknown pathogen — no dashboard injection
        region="United States",
    )


def _run_end_to_end():
    """Run search stage then feed results into filtering pipeline."""
    question = _make_question()

    search_pipeline = SearchStagePipeline(
        search_backend=RealisticFakeSearchBackend(),
        llm_client=FakeLLMClient(),
        backend_name="fake",
    )
    search_results = search_pipeline.run(question)

    filtering_pipeline = FilteringPipeline(llm_client=None)
    docs = filtering_pipeline.run(question, search_results)

    return search_results, docs


class TestSearchToFilteringContract:
    def test_search_output_accepted_by_filtering_pipeline(self):
        """Core contract: search output feeds into filtering without errors."""
        search_results, docs = _run_end_to_end()

        assert isinstance(docs, list)
        assert all(isinstance(d, FilteredDocument) for d in docs)
        assert len(docs) >= 1, "At least one high-quality result should survive filtering"

    def test_official_domains_survive_filtering(self):
        """CDC and/or WHO official sources should pass heuristic threshold."""
        _, docs = _run_end_to_end()

        domains = {d.domain for d in docs}
        official_present = domains & {"cdc.gov", "who.int"}
        assert len(official_present) >= 1, (
            f"Expected at least one official domain in output, got domains: {domains}"
        )

    def test_filtered_document_fields_populated(self):
        """Every FilteredDocument has valid, non-empty required fields."""
        _, docs = _run_end_to_end()

        valid_tiers = {"official", "academic", "trusted_media", "ngo", "unknown"}
        valid_modes = {"html", "pdf", "unknown"}
        valid_values = {"high", "medium", "low"}

        for doc in docs:
            assert doc.result_id, "result_id must be non-empty"
            assert doc.question_id, "question_id must be non-empty"
            assert doc.url, "url must be non-empty"
            assert doc.domain, "domain must be non-empty"
            assert doc.title, "title must be non-empty"

            assert 0.0 <= doc.relevance_score <= 1.0
            assert 0.0 <= doc.credibility_score <= 1.0
            assert 0.0 <= doc.final_score <= 1.0

            assert doc.source_tier in valid_tiers, f"Invalid source_tier: {doc.source_tier}"
            assert doc.extraction_priority > 0, "extraction_priority must be assigned"
            assert doc.extraction_mode in valid_modes, f"Invalid extraction_mode: {doc.extraction_mode}"
            assert doc.expected_value in valid_values, f"Invalid expected_value: {doc.expected_value}"

    def test_result_ids_traceable(self):
        """Every filtered doc ID traces back to a search stage result."""
        search_results, docs = _run_end_to_end()

        search_ids = {r.id for r in search_results}
        for doc in docs:
            assert doc.result_id in search_ids, (
                f"FilteredDocument.result_id {doc.result_id} not found in search results"
            )

    def test_domain_cap_respected(self):
        """No domain appears more than max_docs_per_domain times."""
        _, docs = _run_end_to_end()

        cap = FILTER_CONFIG["max_docs_per_domain"]
        domain_counts = Counter(d.domain for d in docs)
        for domain, count in domain_counts.items():
            assert count <= cap, f"Domain {domain} has {count} docs, exceeds cap of {cap}"

    def test_aggregator_results_handled(self):
        """Aggregator results (e.g. Metaculus) don't crash the pipeline."""
        search_results, docs = _run_end_to_end()

        # Verify aggregator results existed in search output
        aggregator_in_search = [r for r in search_results if r.contains_aggregator_forecast]
        assert len(aggregator_in_search) > 0, "Test precondition: aggregator results should exist in search output"

        # Pipeline completed without error — aggregators were handled gracefully
        assert isinstance(docs, list)
