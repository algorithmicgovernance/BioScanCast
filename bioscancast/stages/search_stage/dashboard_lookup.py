"""Dashboard lookup — inject known pathogen dashboard URLs as SearchResults.

In live mode this returns the live dashboard URL with a synthetic
``published_date=None`` and freshness=1.0 — a sensible signal that the
dashboard "is current". In historical-replay mode (``question.as_of_date``
set), live dashboards are dangerous: they return today's case counts even
for a question created in early 2025. We therefore look up the closest
Wayback snapshot at-or-before the cutoff and rewrite the URL; if no
pre-cutoff snapshot exists, we suppress the dashboard entirely rather
than fall back to live.

v1 — flagged for iteration after first benchmark run.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import List

from bioscancast.datasets.biosecurity_sources import DASHBOARD_LOOKUP
from bioscancast.filtering.models import ForecastQuestion, SearchResult
from bioscancast.stages.search_stage.tier_resolution import is_aggregator_domain, resolve_tier
from bioscancast.stages.search_stage.url_normalization import (
    extract_domain,
    normalize_url,
)
from bioscancast.stages.search_stage.wayback import closest_snapshot_before

logger = logging.getLogger(__name__)


def lookup_dashboards(question: ForecastQuestion) -> List[SearchResult]:
    """Generate synthetic SearchResult entries for known pathogen dashboards.

    Live mode: returns one SearchResult per URL with rank=0 and
    retrieval_reason="dashboard_lookup".

    Historical-replay mode (``question.as_of_date`` is not None): for each
    URL, looks up the closest Wayback snapshot at-or-before the cutoff and
    emits a SearchResult pointing at the snapshot. Dashboards with no
    pre-cutoff snapshot are suppressed entirely (NOT fallen-back to live)
    because live dashboards return today's counts and would silently
    contaminate the benchmark.
    """
    if not question.pathogen:
        return []

    pathogen_key = question.pathogen.strip().lower()
    entries = DASHBOARD_LOOKUP.get(pathogen_key, [])
    if not entries:
        return []

    as_of = question.as_of_date
    results: list[SearchResult] = []
    now = datetime.now(timezone.utc)

    for entry in entries:
        if as_of is not None:
            snapshot = closest_snapshot_before(entry.url, as_of)
            if snapshot is None:
                logger.info(
                    "Suppressing dashboard %s — no Wayback snapshot at-or-before %s",
                    entry.url, as_of.isoformat(),
                )
                continue
            snapshot_dt, snapshot_url = snapshot
            effective_url = snapshot_url
            published_date: datetime | None = snapshot_dt
            published_date_source = "wayback_snapshot"
            # Keep ``domain`` as the original publisher for tier scoring;
            # the URL itself points at archive.org for fetching.
            domain = extract_domain(entry.url)
        else:
            effective_url = entry.url
            published_date = None
            published_date_source = None
            domain = extract_domain(entry.url)

        tier_num, domain_score, source_tier = resolve_tier(domain)

        results.append(
            SearchResult(
                id=uuid.uuid4().hex,
                question_id=question.id,
                query_id=f"dashboard_{question.id}",
                engine="dashboard",
                url=effective_url,
                canonical_url=normalize_url(effective_url),
                domain=domain,
                title=entry.title,
                snippet=entry.snippet,
                rank=0,
                retrieved_at=now,
                published_date=published_date,
                is_official_domain=(tier_num == 1 and source_tier == "official"),
                source_tier=source_tier,
                domain_score=domain_score,
                freshness_score=1.0,
                retrieval_reason="dashboard_lookup",
                contains_aggregator_forecast=is_aggregator_domain(domain),
                search_stage_score=0.0,  # computed later by pipeline
                published_date_source=published_date_source,
                cutoff_applied=as_of,
            )
        )

    return results
