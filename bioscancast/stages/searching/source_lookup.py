"""YAML lookup — inject known source URLs as SearchResults.

This preserves the same SearchResult shape and historical Wayback behavior
as the current dashboard lookup, but reads entries from sources.yaml.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, List

import yaml

from bioscancast.datasets.source_tiers import TIER_SCORE_MAP, get_tier_label
from bioscancast.stages.filtering.models import ForecastQuestion, SearchResult
from bioscancast.stages.searching.tier_resolution import (
    is_aggregator_domain,
)
from bioscancast.stages.searching.url_normalization import (
    extract_domain,
    normalize_url,
)
from bioscancast.stages.searching.wayback import closest_snapshot_before

logger = logging.getLogger(__name__)

_SOURCES_YAML = Path(__file__).resolve().parents[2] / "datasets" / "sources.yaml"


# Common name variants used to resolve a free-text pathogen string inside
# the YAML structure.
_PATHOGEN_ALIASES: dict[str, str] = {
    "monkeypox": "mpox",
    "sars-cov-2": "covid-19",
    "sars-cov2": "covid-19",
    "covid": "covid-19",
    "covid19": "covid-19",
    "coronavirus": "covid-19",
    "bird flu": "h5n1",
    "avian flu": "h5n1",
    "avian influenza": "h5n1",
    "h5": "h5n1",
    "h5nx": "h5n1",
    "bundibugyo": "ebola",
    "sudan virus": "ebola",
    "filovirus": "ebola",
    "poliovirus": "polio",
    "wild poliovirus": "polio",
    "wpv1": "polio",
}

# An explicit influenza A subtype like "H5N5" / "H5N8" (but not "H5N1").
# Used to stop the broad `h5` / `h5nx` / "avian influenza" aliases from
# silently funnelling a non-H5N1 subtype into the H5N1 curated source set
# (issue #58). Bare "h5" and "h5nx" (unspecified N) are intentionally left to
# resolve to h5n1 as the best-available curated set.
_H5_SUBTYPE_RE = re.compile(r"h5n(\d+)")


@lru_cache(maxsize=1)
def _load_sources_yaml() -> dict[str, Any]:
    with open(_SOURCES_YAML, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_pathogen_key(pathogen: str, family_block: dict[str, Any]) -> str | None:
    """Map free-text pathogen text to a YAML pathogen key within one family."""
    key = pathogen.strip().lower()
    if not key:
        return None

    if key in family_block:
        return key

    if key in _PATHOGEN_ALIASES and _PATHOGEN_ALIASES[key] in family_block:
        return _PATHOGEN_ALIASES[key]

    # A text that names an explicit H5 subtype other than H5N1 must not be
    # aliased into the h5n1 source set by substring matching (issue #58).
    _m = _H5_SUBTYPE_RE.search(key)
    non_h5n1_subtype = _m is not None and _m.group(1) != "1"

    for alias, canon in _PATHOGEN_ALIASES.items():
        if alias in key and canon in family_block:
            if non_h5n1_subtype and canon == "h5n1":
                continue
            return canon

    matches = [k for k in family_block if k in key]
    if non_h5n1_subtype:
        matches = [k for k in matches if k != "h5n1"]
    if matches:
        return max(matches, key=len)

    return None

_ROUTE_ALIASES = {
    "general": "general_sources",
}


def _resolve_yaml_tier(entry: dict[str, Any], domain: str) -> tuple[int, float, str]:
    """Resolve tier for YAML-injected sources.

    Policy: default to Tier 1 unless entry explicitly sets `tier:` (1-5).
    """
    raw_tier = entry.get("tier", 1)
    try:
        tier_num = int(raw_tier)
    except (TypeError, ValueError):
        tier_num = 1

    if tier_num not in TIER_SCORE_MAP:
        tier_num = 1

    domain_score, _ = TIER_SCORE_MAP[tier_num]
    source_tier = get_tier_label(domain.lower(), tier_num)
    return tier_num, domain_score, source_tier

def _iter_family_blocks(cfg: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Yield (family_name, family_block) for every well-formed family."""
    families = cfg.get("specific_pathogen_sources", {})
    if not isinstance(families, dict):
        return []
    return [
        (name, block)
        for name, block in families.items()
        if isinstance(block, dict)
    ]


def resolve_pathogen_entries(question: ForecastQuestion) -> list[dict[str, Any]]:
    """Deterministically resolve pathogen-specific entries from ``question.pathogen``.

    Scans *every* family for the question's pathogen (rather than trusting an
    LLM to pick the family first) and returns the matching source entries.
    Returns ``[]`` when the pathogen is unset or resolves in no family — the
    caller then falls back to LLM routing / general sources.

    Entries are de-duplicated by ``id``/``url``: ``h5n1`` is intentionally
    mirrored under both ``respiratory`` and ``animal_spillover`` in the YAML,
    so a cross-family scan would otherwise return each H5N1 source twice.
    """
    if not question.pathogen:
        return []

    cfg = _load_sources_yaml()
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _family, family_block in _iter_family_blocks(cfg):
        pathogen_key = _resolve_pathogen_key(question.pathogen, family_block)
        source_list = family_block.get(pathogen_key) if pathogen_key else None
        if not isinstance(source_list, list):
            continue
        for entry in source_list:
            if not isinstance(entry, dict):
                continue
            dedup_key = str(entry.get("id") or entry.get("url") or "").strip().lower()
            if dedup_key and dedup_key in seen:
                continue
            if dedup_key:
                seen.add(dedup_key)
            entries.append(entry)
    return entries


def _resolve_entries(question: ForecastQuestion, source_route: str) -> list[dict[str, Any]]:
    """Return the YAML entries for the requested route."""
    cfg = _load_sources_yaml()

    source_route = _ROUTE_ALIASES.get(source_route, source_route)

    if source_route == "general_sources":
        entries = cfg.get("general_sources", [])
        return entries if isinstance(entries, list) else []

    families = cfg.get("specific_pathogen_sources", {})

    if not isinstance(families, dict):
        return []

    family_block = families.get(source_route, {})
    if not isinstance(family_block, dict):
        return []

    if not question.pathogen:
        return []

    pathogen_key = _resolve_pathogen_key(question.pathogen, family_block)
    if pathogen_key and isinstance(family_block.get(pathogen_key), list):
        return family_block[pathogen_key]

    # Unresolved pathogen under this family: return nothing rather than
    # flattening the whole family. Flattening would inject wrong-pathogen
    # dashboards (e.g. a Lassa question routed to `hemorrhagic` picking up
    # ebola + marburg case counts the insight stage could misread). The
    # caller falls back to general_sources instead. (issue #41, finding #1)
    return []


def lookup_pathogen_sources(question: ForecastQuestion) -> List[SearchResult]:
    """Deterministic pathogen-first injection.

    Resolves ``question.pathogen`` against every family (no LLM routing) and
    returns the curated sources as SearchResults. Returns ``[]`` when the
    pathogen is unset or unrecognised, signalling the caller to fall back to
    LLM routing / general sources. Shares the historical-replay / Wayback
    behaviour of :func:`lookup_yaml_sources`.
    """
    return _entries_to_results(question, resolve_pathogen_entries(question))


def lookup_yaml_sources(question: ForecastQuestion, source_route: str) -> List[SearchResult]:
    """Generate synthetic SearchResult entries from sources.yaml.

    Live mode: returns one SearchResult per URL with rank=0 and
    retrieval_reason="dashboard_lookup".

    Historical-replay mode: looks up the closest Wayback snapshot
    at-or-before the cutoff and emits a SearchResult pointing at the
    snapshot. Entries with no pre-cutoff snapshot are suppressed.
    """
    return _entries_to_results(question, _resolve_entries(question, source_route))


def _entries_to_results(
    question: ForecastQuestion, entries: list[dict[str, Any]]
) -> List[SearchResult]:
    """Convert resolved YAML entries into SearchResults.

    In historical-replay mode (``question.as_of_date`` set) each URL is
    rewritten to its closest Wayback snapshot at-or-before the cutoff, and
    entries with no pre-cutoff snapshot are suppressed.
    """
    if not entries:
        return []

    as_of = question.as_of_date
    results: list[SearchResult] = []
    now = datetime.now(timezone.utc)

    for entry in entries:
        url = str(entry.get("url", "")).strip()
        if not url:
            continue

        title = str(entry.get("name", url)).strip()
        snippet = str(entry.get("snippet") or entry.get("geography") or title).strip()

        if as_of is not None:
            snapshot = closest_snapshot_before(url, as_of)
            if snapshot is None:
                logger.info(
                    "Suppressing source %s — no Wayback snapshot at-or-before %s",
                    url,
                    as_of.isoformat(),
                )
                continue
            snapshot_dt, snapshot_url = snapshot
            effective_url = snapshot_url
            published_date: datetime | None = snapshot_dt
            published_date_source = "wayback_snapshot"
            domain = extract_domain(url)
        else:
            effective_url = url
            published_date = None
            published_date_source = None
            domain = extract_domain(url)

        tier_num, domain_score, source_tier = _resolve_yaml_tier(entry, domain)

        source_id = str(entry.get("id", "")).strip() or None

        results.append(
            SearchResult(
                id=uuid.uuid4().hex,
                source_id=source_id,
                question_id=question.id,
                query_id=f"dashboard_{question.id}",
                engine="dashboard",
                url=effective_url,
                canonical_url=normalize_url(effective_url),
                domain=domain,
                title=title,
                snippet=snippet,
                rank=0,
                retrieved_at=now,
                published_date=published_date,
                is_official_domain=(tier_num == 1 and source_tier == "official"),
                source_tier=source_tier,
                domain_score=domain_score,
                freshness_score=1.0,
                retrieval_reason="dashboard_lookup",
                contains_aggregator_forecast=is_aggregator_domain(domain),
                search_stage_score=0.0,
                published_date_source=published_date_source,
                cutoff_applied=as_of,
            )
        )

    return results