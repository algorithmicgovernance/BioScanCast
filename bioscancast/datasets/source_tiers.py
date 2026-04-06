"""Canonical domain-to-tier mapping for biosecurity source credibility scoring.

Tier 1 (1.0) — Major health authorities and top-tier journals
Tier 2 (0.8) — Government agencies, preprint servers, wire services
Tier 3 (0.6) — Quality newspapers, educational institutions
Tier 4 (0.4) — Regional news, trade publications, NGOs
Tier 5 (0.2) — Everything else (unknown)
"""

TIER_1_DOMAINS: set[str] = {
    # Health authorities
    "who.int",
    "cdc.gov",
    "ecdc.europa.eu",
    "promedmail.org",
    # Top-tier journals
    "nature.com",
    "thelancet.com",
    "nejm.org",
    "science.org",
    "bmj.com",
    "pnas.org",
}

TIER_2_DOMAINS: set[str] = {
    # Government / regulatory
    "nih.gov",
    "fda.gov",
    "gov.uk",
    "fhi.no",
    "health.gov.au",
    "phac-aspc.gc.ca",
    # Preprint servers
    "medrxiv.org",
    "biorxiv.org",
    # Wire services
    "reuters.com",
    "apnews.com",
    # Genomic surveillance
    "nextstrain.org",
    "gisaid.org",
    # Forecasting aggregators (Tier 2 for credibility, flagged separately)
    "metaculus.com",
    "goodjudgment.io",
    "manifold.markets",
    "kalshi.com",
    "infer-pub.com",
    "polymarket.com",
}

TIER_3_DOMAINS: set[str] = {
    "bbc.com",
    "bbc.co.uk",
    "nytimes.com",
    "washingtonpost.com",
    "theguardian.com",
    "statnews.com",
    "theconversation.com",
    "wikipedia.org",
    "sciencedirect.com",
    "pubmed.ncbi.nlm.nih.gov",
}

TIER_4_DOMAINS: set[str] = {
    "aljazeera.com",
    "france24.com",
    "dw.com",
    "cidrap.umn.edu",
    "reliefweb.int",
    "msf.org",
    "doctorswithoutborders.org",
    "redcross.org",
}

AGGREGATOR_DOMAINS: set[str] = {
    "metaculus.com",
    "goodjudgment.io",
    "manifold.markets",
    "kalshi.com",
    "infer-pub.com",
    "polymarket.com",
}

# Tier number → (score, label) mapping
# Labels must match keys in FILTER_CONFIG["source_tier_scores"]:
#   "official", "academic", "trusted_media", "ngo", "unknown"
TIER_SCORE_MAP: dict[int, tuple[float, str]] = {
    1: (1.0, "official"),
    2: (0.8, "official"),
    3: (0.6, "trusted_media"),
    4: (0.4, "ngo"),
    5: (0.2, "unknown"),
}

# Domain → tier number lookup (built at import time for fast resolution)
DOMAIN_TIER: dict[str, int] = {}
for _domain in TIER_1_DOMAINS:
    DOMAIN_TIER[_domain] = 1
for _domain in TIER_2_DOMAINS:
    DOMAIN_TIER[_domain] = 2
for _domain in TIER_3_DOMAINS:
    DOMAIN_TIER[_domain] = 3
for _domain in TIER_4_DOMAINS:
    DOMAIN_TIER[_domain] = 4


def _classify_tier_1_label(domain: str) -> str:
    """Tier 1 domains are either 'official' (health authority) or 'academic' (journal)."""
    journals = {"nature.com", "thelancet.com", "nejm.org", "science.org", "bmj.com", "pnas.org"}
    return "academic" if domain in journals else "official"


def _classify_tier_2_label(domain: str) -> str:
    """Tier 2: government → 'official', preprint → 'academic', wire → 'trusted_media'."""
    preprints = {"medrxiv.org", "biorxiv.org"}
    wires = {"reuters.com", "apnews.com"}
    genomic = {"nextstrain.org", "gisaid.org"}
    if domain in preprints:
        return "academic"
    if domain in wires:
        return "trusted_media"
    if domain in genomic:
        return "academic"
    # Aggregator domains (Metaculus, etc.) are labeled "trusted_media" because
    # the spec's label set has no "aggregator" category, and these platforms
    # do produce analysis.  They are separately flagged via
    # contains_aggregator_forecast on SearchResult for benchmarking.
    if domain in AGGREGATOR_DOMAINS:
        return "trusted_media"
    return "official"


def _classify_tier_3_label(domain: str) -> str:
    """Tier 3: academic sites → 'academic', rest → 'trusted_media'."""
    academic = {"sciencedirect.com", "pubmed.ncbi.nlm.nih.gov", "wikipedia.org"}
    return "academic" if domain in academic else "trusted_media"


def get_tier_label(domain: str, tier: int) -> str:
    """Get the source_tier string label for a domain at a given tier number."""
    if tier == 1:
        return _classify_tier_1_label(domain)
    if tier == 2:
        return _classify_tier_2_label(domain)
    if tier == 3:
        return _classify_tier_3_label(domain)
    if tier == 4:
        return "ngo"
    return "unknown"
