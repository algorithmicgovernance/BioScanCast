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


# Common name variants that should route to a canonical DASHBOARD_LOOKUP key.
# The canonical-key substring fallback in ``_resolve_pathogen_key`` already
# handles suffixes like "marburg virus disease" -> "marburg"; this map covers
# synonyms where the canonical key is NOT a substring of the alias.
_PATHOGEN_ALIASES: dict[str, str] = {
    "monkeypox": "mpox",
    "sars-cov-2": "covid-19",
    "sars-cov2": "covid-19",
    "covid": "covid-19",
    "covid19": "covid-19",
    "coronavirus": "covid-19",
    "bird flu": "h5n1",
    "avian flu": "h5n1",
}


def _resolve_pathogen_key(pathogen: str) -> str | None:
    """Map a free-text pathogen string to a DASHBOARD_LOOKUP key, tolerantly.

    Resolution order: exact key, exact alias, alias-substring, then
    canonical-key substring (longest match wins, so "ebola virus disease"
    resolves to "ebola" and "marburg virus disease" to "marburg"). Returns
    None if nothing matches.
    """
    key = pathogen.strip().lower()
    if not key:
        return None
    if key in DASHBOARD_LOOKUP:
        return key
    if key in _PATHOGEN_ALIASES and _PATHOGEN_ALIASES[key] in DASHBOARD_LOOKUP:
        return _PATHOGEN_ALIASES[key]
    for alias, canon in _PATHOGEN_ALIASES.items():
        if alias in key and canon in DASHBOARD_LOOKUP:
            return canon
    matches = [k for k in DASHBOARD_LOOKUP if k in key]
    if matches:
        return max(matches, key=len)
    return None


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

    pathogen_key = _resolve_pathogen_key(question.pathogen)
    if not pathogen_key:
        return []
    entries = DASHBOARD_LOOKUP[pathogen_key]

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
