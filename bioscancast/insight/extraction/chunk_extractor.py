"""Extract structured facts from a single document chunk via LLM.

Per-chunk extraction is deliberately simple: one chunk -> zero or more
facts.  It's tempting to give the LLM multiple chunks at once for
"context" but this trades fewer API calls for much harder hallucination
control.  Stick with one chunk per call.
"""

from __future__ import annotations

import logging
import re
import unicodedata
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

import pycountry

from bioscancast.schemas import DocumentChunk, Document, ChunkReference, InsightRecord
from bioscancast.filtering.models import ForecastQuestion
from .prompts import build_extraction_prompt

if TYPE_CHECKING:
    from bioscancast.llm.base import LLMClient, LLMResponse

logger = logging.getLogger(__name__)

# Punctuation the hallucination guard is willing to ignore at the very end
# of a model-supplied quote. Live tests on real WHO/CDC/ECDC documents show
# the model habitually closes paraphrased quotes with '.' even when the
# source has a comma, semicolon, or no terminator at that position.
_TERMINAL_PUNCT = ".;,:!?"

# Typography → ASCII folding applied at layer 1 before substring matching.
# NFKC alone does NOT fold smart quotes (U+2018/9, U+201C/D) or em/en
# dashes — those are independent Unicode codepoints, not compatibility
# forms. But real biosecurity sources mix them freely with their ASCII
# equivalents (WHO and ECDC PDFs in particular use curly quotes and
# em-dashes), and the model normalises them inconsistently in its
# output. Folding here keeps the guard robust to those variants.
_TYPOGRAPHY_FOLD: dict[str, str] = {
    "‘": "'",  # LEFT SINGLE QUOTATION MARK
    "’": "'",  # RIGHT SINGLE QUOTATION MARK
    "‚": "'",  # SINGLE LOW-9 QUOTATION MARK
    "‛": "'",  # SINGLE HIGH-REVERSED-9
    "“": '"',  # LEFT DOUBLE QUOTATION MARK
    "”": '"',  # RIGHT DOUBLE QUOTATION MARK
    "„": '"',  # DOUBLE LOW-9 QUOTATION MARK
    "–": "-",  # EN DASH
    "—": "-",  # EM DASH
    "−": "-",  # MINUS SIGN
    "…": "...",  # HORIZONTAL ELLIPSIS
}

_TYPOGRAPHY_FOLD_RE = re.compile(
    "|".join(re.escape(k) for k in _TYPOGRAPHY_FOLD)
)

# Wrapping punctuation the guard will strip from both sides at layer 3.
# These are characters whose presence-vs-absence around inline elements
# (acronyms like "(NMDOH)", figures like "[12]", quoted speech) flips
# between model output and source text without changing meaning. We do
# NOT strip hyphens or other connecting punctuation because those carry
# semantic load (e.g. "outbreak-related"). Note: smart quotes have
# already been folded to ASCII at layer 1, so this regex only needs to
# list the ASCII variants.
_WRAPPING_PUNCT_RE = re.compile(r"[\(\)\[\]\{\}\"\']")


# pycountry covers all 249 ISO 3166-1 entries by common, official, and
# canonical names; the alias dicts below add only the forms pycountry
# doesn't resolve on its own.

# Country aliases for forms pycountry won't match directly. Keys are
# lowercased and stripped of internal periods, so write them that way
# ("uk", "us", "drc" — NOT "u.k.", "u.s."). The lookup helper does the
# same normalisation on the incoming string.
_COUNTRY_ALIASES: dict[str, str] = {
    "uk": "GB",
    "us": "US",
    "usa": "US",
    "drc": "CD",
    "dr congo": "CD",
    "democratic republic of the congo": "CD",
    "republic of the congo": "CG",
    "uae": "AE",
    "russia": "RU",
    "burma": "MM",
    "ivory coast": "CI",
    "north korea": "KP",
    "south korea": "KR",
    "england": "GB",
    "scotland": "GB",
    "wales": "GB",
    "northern ireland": "GB",
    "great britain": "GB",
}

# US states and territories — biosecurity reporting frequently uses
# state-level location text. Anything in this set resolves to "US".
# Two-letter postal codes are also included (case-insensitive matching).
_US_SUBNATIONAL: set[str] = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york",
    "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
    "pennsylvania", "rhode island", "south carolina", "south dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington",
    "west virginia", "wisconsin", "wyoming",
    "district of columbia", "d.c.", "dc",
    "puerto rico", "guam", "american samoa",
    "u.s. virgin islands", "northern mariana islands",
}

# Multi-country region labels that explicitly resolve to None. The model
# emits these often when reading WHO regional roll-ups (e.g.
# "European Region", "African Region") that aren't single countries.
_NOT_A_COUNTRY: set[str] = {
    "africa", "asia", "europe", "north america", "south america",
    "americas", "oceania",
    "european region", "african region",
    "region of the americas", "western pacific region",
    "south-east asia region", "eastern mediterranean region",
    "european union", "eu", "eu/eea", "eea",
    "world", "global", "globally", "multi-country", "multi country",
    "unknown", "various", "international",
}


def _normalize_whitespace(text: str) -> str:
    """Collapse all whitespace to single spaces for substring matching.

    Retained as a thin wrapper for callers (and tests) that pre-date the
    NFKC-aware match logic. New code should use ``_normalize_for_match``.
    """
    return re.sub(r"\s+", " ", text).strip()


def _normalize_for_match(text: str) -> str:
    """NFKC + typography-to-ASCII fold + whitespace collapse.

    Used by the hallucination guard to compare quotes against chunk text
    on a stable footing. NFKC handles compatibility chars (non-breaking
    spaces, full-width ASCII); the explicit typography fold handles
    smart quotes and em/en dashes (which are NOT compatibility chars in
    Unicode). Without these, the guard rejects real quotes whose only
    difference from the source is a typographic variant.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _TYPOGRAPHY_FOLD_RE.sub(lambda m: _TYPOGRAPHY_FOLD[m.group(0)], text)
    return re.sub(r"\s+", " ", text).strip()


def _quote_matches(quote: str, chunk_text: str) -> Optional[str]:
    """Hallucination guard: return the canonical chunk substring the quote
    matches, or ``None`` if no match.

    Layers applied in order:

    1. **NFKC + whitespace collapse → exact substring.** Catches curly vs
       straight apostrophes, non-breaking spaces, em-dashes, full-width
       ASCII, and the model's whitespace habits.
    2. **Strip terminal punctuation** (``.;,:!?``) from the normalised
       quote, then substring check again. Catches the model's strong
       tendency to close paraphrased quotes with ``.`` even when the
       source has a comma or no punctuation at that position (e.g.
       source: ``"...reported by Italy (63), Spain..."``; model quote:
       ``"...reported by Italy (63)."``).
    3. **Strip wrapping punctuation** (``()[]{}""``) from both quote and
       chunk and retry (also dropping terminal punctuation from the
       quote). Catches the model's habit of dropping the parens around
       acronyms (source: ``"f Health (NMDOH) eventually reported..."``;
       model quote: ``"NMDOH eventually reported..."``).

    The function returns the *canonical* form of the matched substring
    (with the same transformations applied that made the match succeed)
    rather than the model's original output, so the stored
    ``ChunkReference.quote`` always corresponds to actual chunk content
    after the same normalisation. Returns ``None`` when no layer
    matches — caller drops the fact.

    Note: this loosening was driven by live tests showing the strict
    substring-only guard rejected ~85% of real factual quotes due to
    minor punctuation/unicode drift on real WHO/CDC/ECDC documents,
    while the looser three-layer guard still rejects substantive
    paraphrases (e.g. the model bolting a prefix from one sentence onto
    a fragment of another) and content-insertion hallucinations (extra
    words in a list).
    """
    if not quote:
        return None
    norm_quote = _normalize_for_match(quote)
    if not norm_quote:
        return None
    norm_chunk = _normalize_for_match(chunk_text)

    # Layer 1: exact substring after NFKC + whitespace
    if norm_quote in norm_chunk:
        return norm_quote

    # Layer 2: strip terminal punctuation from the quote and retry
    stripped = norm_quote.rstrip(_TERMINAL_PUNCT).strip()
    if stripped and stripped != norm_quote and stripped in norm_chunk:
        return stripped

    # Layer 3: strip wrapping punctuation everywhere on both sides, then
    # strip terminal punctuation from the quote, and retry.
    unwrap_quote = _WRAPPING_PUNCT_RE.sub("", stripped or norm_quote)
    unwrap_quote = re.sub(r"\s+", " ", unwrap_quote).strip()
    unwrap_quote = unwrap_quote.rstrip(_TERMINAL_PUNCT).strip()
    if not unwrap_quote:
        return None
    unwrap_chunk = _WRAPPING_PUNCT_RE.sub("", norm_chunk)
    unwrap_chunk = re.sub(r"\s+", " ", unwrap_chunk).strip()
    if unwrap_quote in unwrap_chunk:
        return unwrap_quote

    # Layer 4: case-insensitive substring. Catches the model lowercasing
    # the leading letter of a sentence it quotes from mid-paragraph -
    # otherwise verbatim drift that's very common (q12 live runs:
    # "there are now 750 suspected cases..." vs the source's "There are
    # now 750..."). Returns the chunk's own casing so the stored quote
    # reflects the source. Crucially this does NOT recover content-
    # insertion hallucinations: a fabricated continuation still fails the
    # substring test regardless of case (verified against the q12
    # "...have been reported in Ituri, North Kivu" fabrication, whose real
    # source text continues "...and 906 suspected cases").
    ci_chunk = norm_chunk.lower()
    for candidate in (norm_quote, stripped):
        if not candidate:
            continue
        idx = ci_chunk.find(candidate.lower())
        if idx >= 0:
            return norm_chunk[idx: idx + len(candidate)]

    return None


def _lookup_one(token: str) -> Optional[str]:
    """Resolve a single location token to an ISO alpha-2 code.

    Order: explicit not-a-country set, alias dict, US subnational, then
    pycountry's built-in lookup (which handles all 249 ISO 3166-1
    entries by common/canonical/official name and alpha-2/alpha-3 codes).
    Returns ``None`` when nothing matches. ``search_fuzzy`` is
    deliberately not used because it produces surprising false positives.

    The token is normalised against ``_NOT_A_COUNTRY``, ``_COUNTRY_ALIASES``,
    and ``_US_SUBNATIONAL`` with internal punctuation removed (so "U.K."
    matches the "uk" alias). pycountry is then called on the typography-
    folded version (so "Côte d'Ivoire" with curly apostrophe also
    resolves) — its own internal matching is case-insensitive but does
    not handle smart quotes.
    """
    if not token:
        return None
    # Typography-fold for pycountry's benefit (smart quotes → straight)
    folded = _TYPOGRAPHY_FOLD_RE.sub(
        lambda m: _TYPOGRAPHY_FOLD[m.group(0)], token.strip()
    )
    # Alias/region lookup key — lowercased, no internal periods, single spaces
    key = folded.lower().replace(".", "")
    key = re.sub(r"\s+", " ", key).strip()
    if not key:
        return None
    if key in _NOT_A_COUNTRY:
        return None
    if key in _COUNTRY_ALIASES:
        return _COUNTRY_ALIASES[key]
    if key in _US_SUBNATIONAL:
        return "US"
    try:
        return pycountry.countries.lookup(folded).alpha_2
    except LookupError:
        return None


def _resolve_country_code(location: Optional[str]) -> Optional[str]:
    """Resolve a free-text location to an ISO 3166-1 alpha-2 country code.

    Handles common biosecurity reporting patterns:

    * Bare country names ("Uganda", "Côte d'Ivoire") — via pycountry.
    * Common abbreviations ("DRC", "UK", "USA") — via the alias dict.
    * US states ("Texas", "New Mexico") → "US".
    * Multi-country regional roll-ups ("European Region", "EU/EEA",
      "Africa") → ``None`` (these aren't single countries).
    * Compound locations like ``"Mubende district, Uganda"`` — tries
      the whole string first, then falls back to each comma-separated
      segment from right to left.
    """
    if not location:
        return None
    cleaned = location.strip()
    if not cleaned:
        return None

    # Try the whole string first
    result = _lookup_one(cleaned)
    if result is not None:
        return result
    # Defer to the not-a-country whitelist on the full string before
    # falling through to per-segment matching. Normalise the same way
    # as _lookup_one so "European Region" / "europe" etc. all match.
    if re.sub(r"\s+", " ", cleaned.lower().replace(".", "")).strip() in _NOT_A_COUNTRY:
        return None

    # Try comma-separated segments right-to-left ("Mubende district, Uganda")
    for part in reversed(cleaned.split(",")):
        part = part.strip()
        if not part:
            continue
        result = _lookup_one(part)
        if result is not None:
            return result
    return None


def _parse_event_date(
    date_str: Optional[str],
) -> tuple[Optional[datetime], Optional[str]]:
    """Parse a date string from the LLM output into a (datetime, precision)
    pair.

    Accepts (in order of preference):

    * ``YYYY-MM-DD`` and similar full ISO forms → precision="day"
    * ``YYYY-MM`` → precision="month"; datetime is the start of that month
    * ``YYYY`` → precision="year"; datetime is January 1 of that year
    * Free-form ``"15 January 2026"``, ``"January 15, 2026"`` → "day"

    Returns ``(None, None)`` when no format matches. Storing precision
    alongside the canonicalised datetime lets the dedup logic merge a
    month-precision record like 2026-01 with a day-precision record
    inside that month without throwing away the underlying granularity.
    """
    if not date_str:
        return None, None

    cleaned = date_str.strip()

    # Day-precision attempts (in order)
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%d %B %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(cleaned, fmt), "day"
        except ValueError:
            continue

    # Month-precision attempt
    try:
        return datetime.strptime(cleaned, "%Y-%m"), "month"
    except ValueError:
        pass

    # Year-precision attempt
    try:
        return datetime.strptime(cleaned, "%Y"), "year"
    except ValueError:
        pass

    return None, None


def extract_facts_from_chunk(
    chunk: DocumentChunk,
    document: Document,
    question: ForecastQuestion,
    llm_client: LLMClient,
    *,
    model: str,
    max_tokens: int = 4096,
) -> tuple[list[InsightRecord], LLMResponse]:
    """Extract structured facts from a single chunk via LLM.

    Args:
        chunk: The chunk to extract from.
        document: Parent document.
        question: Forecast question for context.
        llm_client: LLM client (fake or real).
        model: Model identifier for extraction.
        max_tokens: Per-call output-token cap. Raise this if the model is
            truncating dense pages mid-JSON.

    Returns:
        Tuple of (list of InsightRecords, LLMResponse for budget tracking).
        The response is always returned even if zero facts are extracted.
    """
    system, user, schema = build_extraction_prompt(chunk, document, question)

    response = llm_client.generate_json(
        system=system,
        user=user,
        schema=schema,
        model=model,
        max_tokens=max_tokens,
    )

    facts_raw = response.content.get("facts", [])
    records: list[InsightRecord] = []

    for fact in facts_raw:
        raw_quote = fact.get("quote", "")

        # --- Hallucination guard ---
        # The quote must appear as a substring in the chunk text under
        # NFKC + whitespace normalisation, optionally with terminal
        # punctuation stripped. The guard rejects substantive paraphrases
        # and content-insertion hallucinations. See ``_quote_matches`` for
        # the rationale and the layers.
        canonical_quote = _quote_matches(raw_quote, chunk.text)
        if canonical_quote is None:
            logger.warning(
                "Hallucination guard: dropping fact with non-matching quote. "
                "chunk_id=%s, quote=%r",
                chunk.chunk_id,
                raw_quote[:100],
            )
            continue

        location = fact.get("location")
        iso_code = _resolve_country_code(location)
        event_date, event_date_precision = _parse_event_date(
            fact.get("event_date")
        )

        record = InsightRecord(
            id=f"ins-{uuid.uuid4().hex[:12]}",
            question_id=question.id,
            event_type=fact.get("event_type", "other"),
            confidence=float(fact.get("confidence", 0.5)),
            location=location,
            iso_country_code=iso_code,
            pathogen=fact.get("pathogen"),
            metric_name=fact.get("metric_name"),
            metric_value=(
                float(fact["metric_value"])
                if fact.get("metric_value") is not None
                else None
            ),
            metric_unit=fact.get("metric_unit"),
            event_date=event_date,
            event_date_precision=event_date_precision,
            summary=fact.get("summary"),
            model=model,
            extracted_at=datetime.now(timezone.utc),
            sources=[
                ChunkReference(
                    document_id=document.id,
                    chunk_id=chunk.chunk_id,
                    source_url=document.source_url,
                    quote=canonical_quote[:200],
                ),
            ],
        )
        records.append(record)

    return records, response
