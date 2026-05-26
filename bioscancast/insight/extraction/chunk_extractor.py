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


# Hardcoded country name -> ISO 3166-1 alpha-2 map for the ~30 most
# likely countries in biosecurity reporting.  Don't pull in pycountry.
COUNTRY_TO_ISO: dict[str, str] = {
    "united states": "US",
    "usa": "US",
    "us": "US",
    "united kingdom": "GB",
    "uk": "GB",
    "china": "CN",
    "india": "IN",
    "brazil": "BR",
    "uganda": "UG",
    "democratic republic of the congo": "CD",
    "drc": "CD",
    "congo": "CG",
    "nigeria": "NG",
    "south africa": "ZA",
    "kenya": "KE",
    "ethiopia": "ET",
    "tanzania": "TZ",
    "egypt": "EG",
    "australia": "AU",
    "canada": "CA",
    "mexico": "MX",
    "germany": "DE",
    "france": "FR",
    "italy": "IT",
    "spain": "ES",
    "japan": "JP",
    "south korea": "KR",
    "indonesia": "ID",
    "thailand": "TH",
    "vietnam": "VN",
    "pakistan": "PK",
    "bangladesh": "BD",
    "saudi arabia": "SA",
    "iran": "IR",
    "turkey": "TR",
    "russia": "RU",
    "texas": "US",
    "california": "US",
    "iowa": "US",
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

    return None


def _resolve_country_code(location: Optional[str]) -> Optional[str]:
    """Try to resolve a location string to an ISO country code."""
    if not location:
        return None
    key = location.lower().strip()
    if key in COUNTRY_TO_ISO:
        return COUNTRY_TO_ISO[key]
    # Try matching the last part (e.g., "Mubende district, Uganda" -> "uganda")
    parts = key.split(",")
    for part in reversed(parts):
        part = part.strip()
        if part in COUNTRY_TO_ISO:
            return COUNTRY_TO_ISO[part]
    return None


def _parse_event_date(date_str: Optional[str]) -> Optional[datetime]:
    """Try to parse a date string from the LLM output."""
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%d %B %Y", "%B %d, %Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


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
        event_date = _parse_event_date(fact.get("event_date"))

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
