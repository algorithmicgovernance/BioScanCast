"""Resolve a domain to its source tier, score, and label.

Resolution order:
  1. Exact domain match against curated tier sets.
  2. Second-level domain match (e.g. wwwnc.cdc.gov → cdc.gov).
  3. TLD heuristics (.int → Tier 2, .gov → Tier 3, .edu → Tier 3).
  4. Fallback → Tier 5 (score 0.2, label "unknown").
"""

from __future__ import annotations

from bioscancast.datasets.source_tiers import (
    AGGREGATOR_DOMAINS,
    DOMAIN_TIER,
    TIER_SCORE_MAP,
    get_tier_label,
)


def _second_level_domain(domain: str) -> str | None:
    """Extract the second-level domain (e.g. 'wwwnc.cdc.gov' → 'cdc.gov')."""
    parts = domain.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return None


def resolve_tier(domain: str) -> tuple[int, float, str]:
    """Resolve a domain to (tier_number, domain_score, source_tier_label).

    Returns exact string labels compatible with
    ``FILTER_CONFIG["source_tier_scores"]``: "official", "academic",
    "trusted_media", "ngo", "unknown".
    """
    domain = domain.lower()

    # 1. Exact match
    if domain in DOMAIN_TIER:
        tier = DOMAIN_TIER[domain]
        score, _ = TIER_SCORE_MAP[tier]
        label = get_tier_label(domain, tier)
        return tier, score, label

    # 2. Second-level domain match
    sld = _second_level_domain(domain)
    if sld and sld in DOMAIN_TIER:
        tier = DOMAIN_TIER[sld]
        score, _ = TIER_SCORE_MAP[tier]
        label = get_tier_label(sld, tier)
        return tier, score, label

    # 3. TLD heuristics
    if domain.endswith(".int"):
        return 2, 0.8, "official"
    if domain.endswith(".gov") or domain.endswith(".gov.uk"):
        # .gov → Tier 3 (0.6) to avoid over-crediting unrelated agencies
        return 3, 0.6, "official"
    if domain.endswith(".edu"):
        return 3, 0.6, "academic"

    # 4. Fallback
    return 5, 0.2, "unknown"


def is_official_domain(domain: str) -> bool:
    """True for Tier 1 health-authority domains only."""
    tier, _, label = resolve_tier(domain)
    return tier == 1 and label == "official"


def is_aggregator_domain(domain: str) -> bool:
    """True if domain matches a known forecasting aggregator."""
    domain = domain.lower()
    if domain in AGGREGATOR_DOMAINS:
        return True
    sld = _second_level_domain(domain)
    return sld is not None and sld in AGGREGATOR_DOMAINS
