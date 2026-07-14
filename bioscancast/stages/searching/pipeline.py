"""SearchStagePipeline — orchestrator for Stage 1.

Given a ForecastQuestion, produces a deduplicated, scored list of
SearchResult objects ready for the filtering stage.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import List, Optional

from bioscancast.llm.base import LLMClient

from bioscancast.stages.filtering.config import FILTER_CONFIG
from bioscancast.stages.filtering.heuristics import build_query_terms
from bioscancast.stages.filtering.models import ForecastQuestion, SearchResult
from bioscancast.stages.filtering.utils import keyword_overlap_score
from bioscancast.stages.searching.backends.base import RawSearchResult, SearchBackend
from bioscancast.stages.searching.cache import SearchCache
from bioscancast.stages.searching.source_lookup import (
    lookup_pathogen_sources,
    lookup_yaml_sources,
)
from bioscancast.stages.searching.date_recovery import recover_published_date
from bioscancast.stages.searching.query_decomposition import (
    SubQuery,
    decompose_question,
    route_sources,
)
from bioscancast.stages.searching.tier_resolution import (
    is_aggregator_domain,
    is_official_domain,
    resolve_tier,
)
from bioscancast.stages.searching.url_normalization import extract_domain, normalize_url

logger = logging.getLogger(__name__)

# File extensions that indicate non-content resources
_NON_CONTENT_EXTENSIONS: set[str] = {".zip", ".exe", ".msi", ".dmg", ".tar", ".gz", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".mp4", ".mp3"}

# Default lookback window for historical-replay mode. Tavily's news endpoint
# requires both start_date and end_date to be set together (passing end_date
# alone is silently ignored — see ``backends/tavily_backend.py``). We synthesize
# a start_date 12 months before the cutoff: empirically (2026-05-20) this gives
# 20/20 native pre-cutoff hit rate on the resolved corpus without leaking past
# the cutoff. Tune via ``historical_lookback_days`` on the pipeline.
_DEFAULT_HISTORICAL_LOOKBACK_DAYS = 365


def _should_use_wayback_for_recovery(r: SearchResult) -> bool:
    """Selective gate for the Wayback first-seen leg of the date-recovery chain.

    Wayback CDX is rate-limited (~60 req/min server-side) and even with
    proactive throttling each call costs us a few seconds. For undated
    results that would be dropped on quality grounds anyway — aggregators
    and unknown-tier domains — there is no recall benefit to paying that
    cost. The URL-slug regex and Last-Modified strategies still run; only
    the Wayback leg is gated.
    """
    domain = extract_domain(r.url)
    if is_aggregator_domain(domain):
        logger.debug("Date recovery: skipping Wayback for aggregator %s", domain)
        return False
    if (r.source_tier or "").lower() == "unknown":
        logger.debug("Date recovery: skipping Wayback for unknown-tier %s", domain)
        return False
    return True


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


# search_stage_score weights (sum to 1.0). Relevance (keyword overlap of
# title/snippet/domain against the question terms) is the dominant term:
# domain/freshness/rank alone rank off-topic high-authority content too highly,
# because freshness is ~uniform in live mode and domain score is too coarse to
# separate on-topic from off-topic within a tier. Freshness is kept low for that
# reason. See data/investigations/findings-issues-3-4-13.md (#4).
_SCORE_W_RELEVANCE = 0.45
_SCORE_W_DOMAIN = 0.30
_SCORE_W_FRESHNESS = 0.10
_SCORE_W_RANK = 0.15


def _compute_relevance(result: SearchResult, question: ForecastQuestion) -> float:
    """Keyword overlap of the result against the question terms.

    Mirrors ``bioscancast.stages.filtering.heuristics.compute_heuristic_relevance`` so
    the search stage and the filter stage use the same relevance signal.
    """
    text = f"{result.title} {result.snippet} {result.domain}"
    return keyword_overlap_score(text, build_query_terms(question))


def _compute_search_stage_score(
    relevance: float, domain_score: float, freshness_score: float, rank: int
) -> float:
    """search_stage_score = 0.45*relevance + 0.30*domain + 0.10*freshness + 0.15*(1/rank)"""
    rank_score = 1.0 / max(rank, 1)
    raw = (
        _SCORE_W_RELEVANCE * relevance
        + _SCORE_W_DOMAIN * domain_score
        + _SCORE_W_FRESHNESS * freshness_score
        + _SCORE_W_RANK * rank_score
    )
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
        historical_lookback_days: int = _DEFAULT_HISTORICAL_LOOKBACK_DAYS,
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
        # In historical-replay mode the backend receives end_date=as_of_date
        # and start_date=as_of_date-lookback. Tavily requires the pair; see
        # the module-level note on ``_DEFAULT_HISTORICAL_LOOKBACK_DAYS``.
        self._historical_lookback_days = historical_lookback_days

    def run(self, question: ForecastQuestion) -> List[SearchResult]:
        """Execute the full search stage pipeline."""
        as_of = question.as_of_date

        # 1. Decompose question into sub-queries
        sub_queries = decompose_question(
            question,
            self._llm,
            historical_roleplay=self._historical_roleplay,
        )

        # 2. Inject curated sources.
        #
        # Routing policy (issue #41): default/general sources are injected
        # ONLY when no pathogen-specific sources are available. The structured
        # ``question.pathogen`` field is the primary, deterministic gate; the
        # LLM route is a backup for pathogen-unset / unrecognised questions.
        #
        #   (a) Pathogen-first: resolve question.pathogen against every family.
        #       On a hit, inject those curated sources and skip general.
        #   (b) LLM backup: only if (a) is empty, ask route_sources() to pick a
        #       route. An unresolved pathogen-family route yields nothing (we no
        #       longer flatten a whole family), falling through to (c).
        #   (c) General baseline: the last-resort route when nothing above
        #       produced pathogen-specific sources.
        yaml_results = lookup_pathogen_sources(question)
        if yaml_results:
            logger.info(
                "Pathogen-first injection: %d curated source(s) for pathogen=%r",
                len(yaml_results),
                question.pathogen,
            )
        else:
            source_route = route_sources(question, self._llm)
            yaml_results = lookup_yaml_sources(question, source_route)
            if not yaml_results and source_route != "general_sources":
                logger.info(
                    "Route %r yielded no sources; falling back to general_sources",
                    source_route,
                )
                source_route = "general_sources"
                yaml_results = lookup_yaml_sources(question, source_route)
            logger.info(
                "No pathogen-specific sources; LLM backup route=%s produced %d result(s)",
                source_route,
                len(yaml_results),
            )

        logger.info(
            "Decomposed into %d sub-queries; injected %d curated source(s)",
            len(sub_queries),
            len(yaml_results),
        )

        all_results: list[SearchResult] = list(yaml_results)
        seen_canonical: set[str] = {
            r.canonical_url for r in all_results if r.canonical_url
        }
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
        # to fish for more in-window content. With the start_date+end_date
        # pair now forwarded to Tavily (see backends/tavily_backend.py),
        # the candidate pool is already date-filtered upstream; top-up
        # mostly compensates for results dropped by deduplication and the
        # blocked-domain filter.
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
                _compute_relevance(r, question),
                r.domain_score,
                r.freshness_score,
                r.rank,
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
        search backend's lexical match biases toward dated content. The
        start_date+end_date pair forwarded to Tavily already filters by
        publication date, but the year hint reinforces topical relevance
        within the window (Tavily's in-window ranking can still surface
        irrelevant dated-correct results on cold or sparse queries). No-op
        in live mode."""
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
        # In historical-replay mode we pass BOTH start_date and end_date.
        # Tavily silently ignores end_date when start_date is missing
        # (verified 2026-05-20, specs/tavily-investigation-findings.md).
        end_date_str: Optional[str] = None
        start_date_str: Optional[str] = None
        if as_of_date is not None:
            end_date_str = as_of_date.strftime("%Y-%m-%d")
            start_date_str = (
                as_of_date - timedelta(days=self._historical_lookback_days)
            ).strftime("%Y-%m-%d")
        effective_max = max_results if max_results is not None else self._results_per_query
        if self._cache:
            cached = self._cache.get(self._backend_name, query, as_of_date=as_of_date)
            if cached is not None:
                logger.debug("Cache hit for query: %s", query)
                return cached

        results = self._backend.search(
            query,
            max_results=effective_max,
            end_date=end_date_str,
            start_date=start_date_str,
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
        wayback_skipped = 0
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

            # Undated — try the recovery chain. Skip the Wayback first-seen
            # leg for aggregator domains and unknown-tier sources: those
            # results would be dropped on quality grounds anyway, and the
            # CDX call (even with throttling) costs us several seconds each.
            use_wayback = _should_use_wayback_for_recovery(r)
            if not use_wayback:
                wayback_skipped += 1
            recovered_date, source = recover_published_date(
                r.url, use_wayback=use_wayback
            )
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
            "dropped_undatable=%d, wayback_skipped=%d (cutoff=%s)",
            len(kept), recovered, dropped_post_cutoff, dropped_undatable,
            wayback_skipped, as_of.isoformat(),
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
