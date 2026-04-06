"""SearchStagePipeline — orchestrator for Stage 1.

Given a ForecastQuestion, produces a deduplicated, scored list of
SearchResult objects ready for the filtering stage.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from bioscancast.filtering.config import FILTER_CONFIG
from bioscancast.filtering.models import ForecastQuestion, SearchResult
from bioscancast.llm.client import LLMClient
from bioscancast.stages.search_stage.backends.base import RawSearchResult, SearchBackend
from bioscancast.stages.search_stage.cache import SearchCache
from bioscancast.stages.search_stage.dashboard_lookup import lookup_dashboards
from bioscancast.stages.search_stage.query_decomposition import SubQuery, decompose_question
from bioscancast.stages.search_stage.tier_resolution import (
    is_aggregator_domain,
    is_official_domain,
    resolve_tier,
)
from bioscancast.stages.search_stage.url_normalization import extract_domain, normalize_url

logger = logging.getLogger(__name__)

# File extensions that indicate non-content resources
_NON_CONTENT_EXTENSIONS: set[str] = {".zip", ".exe", ".msi", ".dmg", ".tar", ".gz", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".mp4", ".mp3"}


def _compute_freshness(published_date: Optional[datetime]) -> float:
    """Compute freshness score from published_date.

    Returns 0.5 (neutral) when no date is available, per spec.
    """
    if published_date is None:
        return 0.5
    days_old = (datetime.now(timezone.utc) - published_date).days
    if days_old < 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - (days_old / 365.0)))


def _compute_search_stage_score(domain_score: float, freshness_score: float, rank: int) -> float:
    """search_stage_score = 0.5 * domain_score + 0.3 * freshness_score + 0.2 * (1/rank)"""
    rank_score = 1.0 / max(rank, 1)
    raw = 0.5 * domain_score + 0.3 * freshness_score + 0.2 * rank_score
    return max(0.0, min(1.0, raw))


def _parse_published_date(date_str: Optional[str]) -> Optional[datetime]:
    """Best-effort parse of backend-provided published_date strings."""
    if not date_str:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(date_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def _is_non_content_url(url: str) -> bool:
    """Check if URL points to a non-content resource (zip, exe, image, etc.)."""
    path = url.split("?")[0].split("#")[0].lower()
    return any(path.endswith(ext) for ext in _NON_CONTENT_EXTENSIONS)


class SearchStagePipeline:
    """Orchestrates the full search stage: decompose → search → score → deduplicate."""

    def __init__(
        self,
        search_backend: SearchBackend,
        llm_client: LLMClient,
        cache: Optional[SearchCache] = None,
        results_per_query: int = 10,
        total_cap: int = 60,
        backend_name: str = "tavily",
    ) -> None:
        self._backend = search_backend
        self._llm = llm_client
        self._cache = cache
        self._results_per_query = results_per_query
        self._total_cap = total_cap
        self._backend_name = backend_name

    def run(self, question: ForecastQuestion) -> List[SearchResult]:
        """Execute the full search stage pipeline."""
        # 1. Decompose question into sub-queries
        sub_queries = decompose_question(question, self._llm)
        logger.info("Decomposed into %d sub-queries", len(sub_queries))

        # 2. Inject dashboard lookups
        dashboard_results = lookup_dashboards(question)
        logger.info("Dashboard lookup produced %d results", len(dashboard_results))

        # 3. Execute searches per sub-query
        all_results: list[SearchResult] = list(dashboard_results)
        for sq in sub_queries:
            raw_results = self._execute_search(sq.text)
            for rank_offset, raw in enumerate(raw_results):
                result = self._convert(raw, sq, question.id, rank_offset + 1)
                all_results.append(result)

            if len(all_results) >= self._total_cap:
                logger.info("Hit total cap of %d results before all sub-queries", self._total_cap)
                break

        # 4. Deduplicate
        deduped = self._deduplicate(all_results)

        # 5. Hard exclusions
        filtered = self._apply_exclusions(deduped)

        # 6. Compute search_stage_score
        for r in filtered:
            r.search_stage_score = _compute_search_stage_score(
                r.domain_score, r.freshness_score, r.rank
            )

        # 7. Sort and cap
        filtered.sort(key=lambda r: r.search_stage_score, reverse=True)
        result = filtered[: self._total_cap]
        logger.info("Search stage returning %d results", len(result))
        return result

    def _execute_search(self, query: str) -> List[RawSearchResult]:
        # TODO: multilingual support
        if self._cache:
            cached = self._cache.get(self._backend_name, query)
            if cached is not None:
                logger.debug("Cache hit for query: %s", query)
                return cached

        results = self._backend.search(query, max_results=self._results_per_query)

        if self._cache:
            self._cache.put(self._backend_name, query, results)

        return results

    def _convert(
        self, raw: RawSearchResult, sub_query: SubQuery, question_id: str, rank: int
    ) -> SearchResult:
        domain = extract_domain(raw.url)
        canonical = normalize_url(raw.url)
        tier_num, domain_score, source_tier = resolve_tier(domain)
        published = _parse_published_date(raw.published_date)
        freshness = _compute_freshness(published)

        return SearchResult(
            id=uuid.uuid4().hex,
            question_id=question_id,
            query_id=sub_query.id,
            engine=self._backend_name,
            url=raw.url,
            canonical_url=canonical,
            domain=domain,
            title=raw.title,
            snippet=raw.snippet,
            rank=rank,
            retrieved_at=datetime.now(timezone.utc),
            published_date=published,
            is_official_domain=is_official_domain(domain),
            source_tier=source_tier,
            domain_score=domain_score,
            freshness_score=freshness,
            retrieval_reason=sub_query.axis,
            # contains_aggregator_forecast is flagged for benchmarking —
            # kept in results so downstream analysis can measure contamination effects.
            contains_aggregator_forecast=is_aggregator_domain(domain),
            search_stage_score=0.0,  # computed after dedup
        )

    def _deduplicate(self, results: List[SearchResult]) -> List[SearchResult]:
        """Keep highest-ranked per canonical_url, merging retrieval_reason."""
        seen: dict[str, SearchResult] = {}
        for r in results:
            key = r.canonical_url or r.url
            if key not in seen:
                seen[key] = r
            else:
                existing = seen[key]
                # Keep the one with better rank (lower = better)
                if r.rank < existing.rank:
                    # Merge retrieval reasons
                    merged_reason = existing.retrieval_reason or ""
                    if r.retrieval_reason and r.retrieval_reason not in merged_reason:
                        r.retrieval_reason = f"{merged_reason},{r.retrieval_reason}" if merged_reason else r.retrieval_reason
                    seen[key] = r
                else:
                    # Merge reason into existing
                    if r.retrieval_reason and r.retrieval_reason not in (existing.retrieval_reason or ""):
                        existing.retrieval_reason = (
                            f"{existing.retrieval_reason},{r.retrieval_reason}"
                            if existing.retrieval_reason
                            else r.retrieval_reason
                        )
        return list(seen.values())

    def _apply_exclusions(self, results: List[SearchResult]) -> List[SearchResult]:
        """Drop blocked domains and non-content URLs."""
        blocked = FILTER_CONFIG["blocked_domains"]
        kept: list[SearchResult] = []
        for r in results:
            if r.domain in blocked:
                logger.debug("Excluded blocked domain: %s", r.domain)
                continue
            if _is_non_content_url(r.url):
                logger.debug("Excluded non-content URL: %s", r.url)
                continue
            kept.append(r)
        return kept


def run_search_stage(
    question: ForecastQuestion,
    search_backend: SearchBackend,
    llm_client: LLMClient,
    cache: Optional[SearchCache] = None,
    backend_name: str = "tavily",
) -> List[SearchResult]:
    """Convenience function to run the search stage pipeline."""
    pipeline = SearchStagePipeline(
        search_backend=search_backend,
        llm_client=llm_client,
        cache=cache,
        backend_name=backend_name,
    )
    return pipeline.run(question)
