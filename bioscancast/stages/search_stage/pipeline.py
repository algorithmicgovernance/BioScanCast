"""SearchStagePipeline — orchestrator for Stage 1.

Given a ForecastQuestion, produces a deduplicated, scored list of
SearchResult objects ready for the filtering stage.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import List, Optional

from bioscancast.filtering.config import FILTER_CONFIG
from bioscancast.filtering.models import ForecastQuestion, SearchResult
from bioscancast.llm.client import LLMClient
from bioscancast.stages.search_stage.backends.base import RawSearchResult, SearchBackend
from bioscancast.stages.search_stage.cache import SearchCache
from bioscancast.stages.search_stage.dashboard_lookup import lookup_dashboards
from bioscancast.stages.search_stage.date_recovery import recover_published_date
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


def _compute_freshness(
    published_date: Optional[datetime],
    *,
    reference_date: Optional[datetime] = None,
) -> float:
    """Compute freshness score from published_date.

    Returns 0.5 (neutral) when no date is available, per spec. ``reference_date``
    is the "now" against which age is measured; in historical-replay mode the
    pipeline passes ``question.as_of_date`` so freshness is judged from the
    human forecaster's vantage point. Defaults to wall-clock ``now`` for
    live mode.
    """
    if published_date is None:
        return 0.5
    ref = reference_date or datetime.now(timezone.utc)
    days_old = (ref - published_date).days
    if days_old < 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - (days_old / 365.0)))


def _compute_search_stage_score(domain_score: float, freshness_score: float, rank: int) -> float:
    """search_stage_score = 0.5 * domain_score + 0.3 * freshness_score + 0.2 * (1/rank)"""
    rank_score = 1.0 / max(rank, 1)
    raw = 0.5 * domain_score + 0.3 * freshness_score + 0.2 * rank_score
    return max(0.0, min(1.0, raw))


def _parse_published_date(date_str: Optional[str]) -> Optional[datetime]:
    """Best-effort parse of backend-provided published_date strings.

    Tavily inconsistently returns either ISO-8601 (``2025-02-17`` or
    ``2025-02-17T13:00:00+00:00``) or RFC 2822 (``Tue, 19 May 2026 13:00:00
    GMT``) depending on the search topic, so we try both. Returning None
    here is expensive in historical mode (it triggers the date-recovery
    chain), so it matters that we cover the formats Tavily actually emits.
    """
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
    # RFC 2822 fallback — what Tavily's news topic actually returns.
    try:
        dt = parsedate_to_datetime(date_str)
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (TypeError, ValueError):
        return None


def _is_non_content_url(url: str) -> bool:
    """Check if URL points to a non-content resource (zip, exe, image, etc.)."""
    path = url.split("?")[0].split("#")[0].lower()
    return any(path.endswith(ext) for ext in _NON_CONTENT_EXTENSIONS)


class SearchStagePipeline:
    """Orchestrates the full search stage: decompose → search → score → deduplicate.

    Historical-replay mode is activated implicitly by ``question.as_of_date``.
    When that field is non-None:

    * the Tavily/CSE backend receives ``end_date=as_of_date`` and is asked to
      restrict results to pages dated on or before the cutoff,
    * the cache key incorporates the cutoff so replay runs don't see each
      other's results,
    * freshness scoring uses the cutoff as "now" rather than wall-clock time,
    * a post-retrieval filter drops anything dated after the cutoff and any
      undated result whose date can't be recovered from a cheap fallback,
    * the dashboard injection rewrites URLs to closest Wayback snapshots,
      suppressing dashboards with no pre-cutoff snapshot entirely,
    * (opt-in) the LLM decomposition prompt is asked to roleplay the cutoff.

    The ``historical_roleplay`` constructor flag controls only the last item;
    everything else is implicit on ``as_of_date``.
    """

    def __init__(
        self,
        search_backend: SearchBackend,
        llm_client: LLMClient,
        cache: Optional[SearchCache] = None,
        results_per_query: int = 10,
        total_cap: int = 60,
        backend_name: str = "tavily",
        historical_roleplay: bool = False,
        min_post_filter_results: int = 10,
        top_up_results_per_query: int = 50,
        max_top_up_rounds: int = 1,
    ) -> None:
        self._backend = search_backend
        self._llm = llm_client
        self._cache = cache
        self._results_per_query = results_per_query
        self._total_cap = total_cap
        self._backend_name = backend_name
        self._historical_roleplay = historical_roleplay
        # Top-up parameters apply only in historical-replay mode. In live
        # mode the initial pass is always considered sufficient.
        self._min_post_filter_results = min_post_filter_results
        self._top_up_results_per_query = top_up_results_per_query
        self._max_top_up_rounds = max_top_up_rounds

    def run(self, question: ForecastQuestion) -> List[SearchResult]:
        """Execute the full search stage pipeline."""
        as_of = question.as_of_date

        # 1. Decompose question into sub-queries
        sub_queries = decompose_question(
            question,
            self._llm,
            historical_roleplay=self._historical_roleplay,
        )
        logger.info("Decomposed into %d sub-queries", len(sub_queries))

        # 2. Inject dashboard lookups
        dashboard_results = lookup_dashboards(question)
        logger.info("Dashboard lookup produced %d results", len(dashboard_results))

        # 3. Execute initial search round
        all_results: list[SearchResult] = list(dashboard_results)
        seen_canonical: set[str] = set()
        for r in dashboard_results:
            if r.canonical_url:
                seen_canonical.add(r.canonical_url)

        all_results, seen_canonical = self._search_round(
            sub_queries,
            question,
            as_of,
            max_results=self._results_per_query,
            collected=all_results,
            seen_canonical=seen_canonical,
            stop_cap=self._total_cap,
        )

        # 4-6. Dedup → exclusions → cutoff filter
        filtered = self._dedup_exclude_cutoff(all_results, as_of)

        # 6b. Top-up: in historical mode only, if we're below the survivor
        # threshold, run additional rounds with a larger results_per_query
        # to fish for older content. Tavily's end_date doesn't actually
        # filter (see tavily_backend.py), so the candidate pool is heavily
        # biased toward post-cutoff content for recent topics — top-up is
        # what makes historical mode actually return usable results.
        if as_of is not None:
            rounds_done = 0
            while (
                rounds_done < self._max_top_up_rounds
                and len(filtered) < self._min_post_filter_results
            ):
                rounds_done += 1
                logger.info(
                    "Historical top-up round %d: have %d survivors, want >= %d",
                    rounds_done, len(filtered), self._min_post_filter_results,
                )
                all_results, seen_canonical = self._search_round(
                    sub_queries,
                    question,
                    as_of,
                    max_results=self._top_up_results_per_query,
                    collected=all_results,
                    seen_canonical=seen_canonical,
                    # Allow many more candidates than the final cap because
                    # most will be dropped by the cutoff filter.
                    stop_cap=self._total_cap * 10,
                )
                filtered = self._dedup_exclude_cutoff(all_results, as_of)

            if len(filtered) < self._min_post_filter_results:
                logger.warning(
                    "Historical top-up exhausted: %d survivors after %d round(s) "
                    "(target was %d). Returning what we have.",
                    len(filtered), rounds_done, self._min_post_filter_results,
                )

        # 7. Compute search_stage_score (freshness measured from cutoff in
        # historical mode, wall-clock in live mode)
        for r in filtered:
            r.freshness_score = _compute_freshness(
                r.published_date, reference_date=as_of
            )
            r.search_stage_score = _compute_search_stage_score(
                r.domain_score, r.freshness_score, r.rank
            )

        # 8. Sort and cap
        filtered.sort(key=lambda r: r.search_stage_score, reverse=True)
        result = filtered[: self._total_cap]
        logger.info("Search stage returning %d results", len(result))
        return result

    def _search_round(
        self,
        sub_queries: List[SubQuery],
        question: ForecastQuestion,
        as_of: Optional[datetime],
        *,
        max_results: int,
        collected: list[SearchResult],
        seen_canonical: set[str],
        stop_cap: int,
    ) -> tuple[list[SearchResult], set[str]]:
        """Issue each sub-query and append converted SearchResults to
        ``collected``, skipping any URL already in ``seen_canonical``.
        Returns the updated list and seen-set. Stops early when the
        collected list reaches ``stop_cap``.
        """
        for sq in sub_queries:
            query_text = self._apply_year_hint(sq.text, as_of)
            raw_results = self._execute_search(
                query_text, as_of_date=as_of, max_results=max_results
            )
            for rank_offset, raw in enumerate(raw_results):
                canonical = normalize_url(raw.url)
                if canonical and canonical in seen_canonical:
                    continue
                result = self._convert(raw, sq, question.id, rank_offset + 1, as_of)
                collected.append(result)
                if canonical:
                    seen_canonical.add(canonical)
            if len(collected) >= stop_cap:
                logger.info(
                    "Stopping search round at %d collected results (cap=%d)",
                    len(collected), stop_cap,
                )
                break
        return collected, seen_canonical

    def _dedup_exclude_cutoff(
        self, results: list[SearchResult], as_of: Optional[datetime]
    ) -> list[SearchResult]:
        """Run dedup → hard exclusions → cutoff filter (historical mode only)."""
        deduped = self._deduplicate(results)
        filtered = self._apply_exclusions(deduped)
        if as_of is not None:
            filtered = self._apply_cutoff_filter(filtered, as_of)
        return filtered

    @staticmethod
    def _apply_year_hint(query: str, as_of: Optional[datetime]) -> str:
        """In historical mode, append the cutoff year to the query so the
        search backend's lexical match biases toward dated content. Empirically
        Tavily ignores the ``end_date`` parameter, so query-text steering is
        what actually drags results toward the right time period. No-op in
        live mode."""
        if as_of is None:
            return query
        year = as_of.year
        # Avoid double-hinting if the LLM already put the year in.
        if str(year) in query:
            return query
        return f"{query} {year}"

    def _execute_search(
        self,
        query: str,
        as_of_date: Optional[datetime] = None,
        max_results: Optional[int] = None,
    ) -> List[RawSearchResult]:
        # TODO: multilingual support
        # end_date is passed for Protocol compliance but TavilyBackend
        # explicitly does not forward it (see backends/tavily_backend.py).
        end_date_str = as_of_date.strftime("%Y-%m-%d") if as_of_date else None
        effective_max = max_results if max_results is not None else self._results_per_query
        if self._cache:
            cached = self._cache.get(self._backend_name, query, as_of_date=as_of_date)
            if cached is not None:
                logger.debug("Cache hit for query: %s", query)
                return cached

        results = self._backend.search(
            query, max_results=effective_max, end_date=end_date_str
        )

        if self._cache:
            self._cache.put(self._backend_name, query, results, as_of_date=as_of_date)

        return results

    def _convert(
        self,
        raw: RawSearchResult,
        sub_query: SubQuery,
        question_id: str,
        rank: int,
        as_of_date: Optional[datetime] = None,
    ) -> SearchResult:
        domain = extract_domain(raw.url)
        canonical = normalize_url(raw.url)
        tier_num, domain_score, source_tier = resolve_tier(domain)
        published = _parse_published_date(raw.published_date)
        freshness = _compute_freshness(published, reference_date=as_of_date)
        published_date_source = "backend" if published is not None else None

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
            published_date_source=published_date_source,
            cutoff_applied=as_of_date,
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

    def _apply_cutoff_filter(
        self, results: List[SearchResult], as_of: datetime
    ) -> List[SearchResult]:
        """Historical-replay mode: keep only results that demonstrably existed
        before ``as_of``. Drop post-cutoff and undatable results.

        Wayback-snapshot dashboards already have ``published_date`` set to the
        capture timestamp by ``dashboard_lookup``; this filter is therefore
        idempotent on them.
        """
        dropped_post_cutoff = 0
        dropped_undatable = 0
        recovered = 0
        kept: list[SearchResult] = []
        for r in results:
            if r.published_date is not None:
                if r.published_date > as_of:
                    dropped_post_cutoff += 1
                    logger.debug(
                        "Cutoff filter: dropping post-cutoff %s (pub=%s, cutoff=%s)",
                        r.url, r.published_date.isoformat(), as_of.isoformat(),
                    )
                    continue
                kept.append(r)
                continue

            # Undated — try the recovery chain
            recovered_date, source = recover_published_date(r.url)
            if recovered_date is None:
                dropped_undatable += 1
                logger.debug(
                    "Cutoff filter: dropping %s (no_date_available)", r.url
                )
                continue
            if recovered_date > as_of:
                dropped_post_cutoff += 1
                logger.debug(
                    "Cutoff filter: recovered date %s > cutoff for %s",
                    recovered_date.isoformat(), r.url,
                )
                continue
            r.published_date = recovered_date
            r.published_date_source = source
            recovered += 1
            kept.append(r)

        logger.info(
            "Cutoff filter: kept=%d, recovered=%d, dropped_post_cutoff=%d, "
            "dropped_undatable=%d (cutoff=%s)",
            len(kept), recovered, dropped_post_cutoff, dropped_undatable,
            as_of.isoformat(),
        )
        return kept


def run_search_stage(
    question: ForecastQuestion,
    search_backend: SearchBackend,
    llm_client: LLMClient,
    cache: Optional[SearchCache] = None,
    backend_name: str = "tavily",
    historical_roleplay: bool = False,
) -> List[SearchResult]:
    """Convenience function to run the search stage pipeline."""
    pipeline = SearchStagePipeline(
        search_backend=search_backend,
        llm_client=llm_client,
        cache=cache,
        backend_name=backend_name,
        historical_roleplay=historical_roleplay,
    )
    return pipeline.run(question)
