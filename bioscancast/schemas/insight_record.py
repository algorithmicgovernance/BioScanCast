from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class ChunkReference:
    """A provenance link from an InsightRecord back to a specific document chunk.

    Every insight must cite at least one chunk so that forecasts are
    auditable back to a source URL and passage.
    """

    document_id: str
    """References Document.id."""

    chunk_id: str
    """References DocumentChunk.chunk_id within that document."""

    source_url: str
    """Denormalised source URL for convenient display without a join."""

    quote: str
    """Short verbatim excerpt from the chunk (max ~200 characters)."""


@dataclass
class InsightRecord:
    """An atomic factual claim extracted from one or more document chunks.

    The insight stage produces a list of InsightRecords per forecast
    question.  Each record captures a single biosecurity-relevant fact
    in a structured form suitable for the forecasting stage to reason
    over.  Structured fields are Optional so that partial extractions
    (e.g. an event with no numeric metric) are valid.
    """

    # ---- identity ----
    id: str
    """Unique insight identifier."""

    question_id: str
    """Foreign key to ForecastQuestion.id."""

    event_type: str
    """Category of the fact: 'case_count', 'death_count', 'outbreak_declared',
    'intervention', 'policy_change', or 'other'."""

    confidence: float
    """Model confidence in this extraction, in [0, 1].  Not a forecast probability."""

    # ---- structured fact (all optional for partial extractions) ----
    location: Optional[str] = None
    """Free-text geographic location (e.g. 'Mubende district, Uganda')."""

    iso_country_code: Optional[str] = None
    """ISO 3166-1 alpha-2 country code (e.g. 'UG')."""

    pathogen: Optional[str] = None
    """Pathogen or disease name (e.g. 'Sudan virus')."""

    metric_name: Optional[str] = None
    """What is being counted (e.g. 'confirmed_cases', 'affected_herds')."""

    metric_value: Optional[float] = None
    """Numeric value of the metric."""

    metric_unit: Optional[str] = None
    """Unit of the metric (e.g. 'cases', 'herds', 'deaths')."""

    event_date: Optional[datetime] = None
    """Date the fact pertains to (not the date it was reported)."""

    # ---- free-text fallback ----
    summary: Optional[str] = None
    """Free-text description for facts that don't fit the structured fields."""

    # ---- extraction metadata ----
    model: Optional[str] = None
    """Identifier of the LLM or extraction model that produced this record."""

    extracted_at: Optional[datetime] = None
    """UTC timestamp of when the insight was extracted."""

    notes: Optional[str] = None
    """Free-text notes from the extraction process."""

    # ---- provenance ----
    sources: List[ChunkReference] = field(default_factory=list)
    """Chunk references supporting this insight.  At least one is expected."""
