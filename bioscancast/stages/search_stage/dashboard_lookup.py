"""Dashboard lookup — inject known pathogen dashboard URLs as SearchResults.

v1 — flagged for iteration after first benchmark run.
"""

from __future__ import annotations

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


def lookup_dashboards(question: ForecastQuestion) -> List[SearchResult]:
    """Generate synthetic SearchResult entries for known pathogen dashboards.

    If ``question.pathogen`` (lowercased) matches a key in DASHBOARD_LOOKUP,
    returns a SearchResult for each URL with rank=0 and
    retrieval_reason="dashboard_lookup".  Returns empty list if no match.
    """
    if not question.pathogen:
        return []

    pathogen_key = question.pathogen.strip().lower()
    urls = DASHBOARD_LOOKUP.get(pathogen_key, [])
    if not urls:
        return []

    results: list[SearchResult] = []
    now = datetime.now(timezone.utc)

    for url in urls:
        domain = extract_domain(url)
        tier_num, domain_score, source_tier = resolve_tier(domain)

        results.append(
            SearchResult(
                id=uuid.uuid4().hex,
                question_id=question.id,
                query_id=f"dashboard_{question.id}",
                engine="dashboard",
                url=url,
                canonical_url=normalize_url(url),
                domain=domain,
                title=f"Dashboard: {domain}",
                snippet=f"Known {pathogen_key} monitoring dashboard",
                rank=0,
                retrieved_at=now,
                is_official_domain=(tier_num == 1 and source_tier == "official"),
                source_tier=source_tier,
                domain_score=domain_score,
                freshness_score=1.0,
                retrieval_reason="dashboard_lookup",
                contains_aggregator_forecast=is_aggregator_domain(domain),
                search_stage_score=0.0,  # computed later by pipeline
            )
        )

    return results
