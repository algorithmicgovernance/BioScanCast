"""Extract structured facts from a single document chunk via LLM.

Per-chunk extraction is deliberately simple: one chunk -> zero or more
facts.  It's tempting to give the LLM multiple chunks at once for
"context" but this trades fewer API calls for much harder hallucination
control.  Stick with one chunk per call.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from bioscancast.schemas import DocumentChunk, Document, ChunkReference, InsightRecord
from bioscancast.filtering.models import ForecastQuestion
from .prompts import build_extraction_prompt

if TYPE_CHECKING:
    from bioscancast.llm.base import LLMClient, LLMResponse

logger = logging.getLogger(__name__)


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
    """Collapse all whitespace to single spaces for substring matching."""
    return re.sub(r"\s+", " ", text).strip()


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
) -> tuple[list[InsightRecord], LLMResponse]:
    """Extract structured facts from a single chunk via LLM.

    Args:
        chunk: The chunk to extract from.
        document: Parent document.
        question: Forecast question for context.
        llm_client: LLM client (fake or real).
        model: Model identifier for extraction.

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
    )

    facts_raw = response.content.get("facts", [])
    records: list[InsightRecord] = []
    chunk_text_normalized = _normalize_whitespace(chunk.text)

    for fact in facts_raw:
        quote = fact.get("quote", "")
        quote_normalized = _normalize_whitespace(quote)

        # --- Hallucination guard ---
        # The quote must appear as a substring in the chunk text.
        # Exact substring check (whitespace-normalized) is the point —
        # don't soften to fuzzy match without careful consideration.
        if not quote_normalized or quote_normalized not in chunk_text_normalized:
            logger.warning(
                "Hallucination guard: dropping fact with non-matching quote. "
                "chunk_id=%s, quote=%r",
                chunk.chunk_id,
                quote[:100],
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
                    quote=quote[:200],
                ),
            ],
        )
        records.append(record)

    return records, response
