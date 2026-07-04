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

    count_basis: Optional[str] = None
    """How the metric is counted: ``'cumulative'`` (running total to date),
    ``'incident'`` (new in a stated period), ``'active'`` (currently active),
    ``'prevalence'`` (point-prevalent), or ``'unknown'``. Distinguishes
    epidemiologically different numbers that would otherwise flatten to the
    same ``metric_name``/``metric_value`` (e.g. "282 cumulative" vs "282 new
    this week" vs "282 active"). Extracted, never inferred beyond explicit
    textual cues. Orthogonal to the (separate, planned) ``value_basis`` axis
    that distinguishes observed from projected/modeled numbers."""

    time_window: Optional[str] = None
    """Reporting period for an *incident* count: ``'day'``, ``'week'``,
    ``'month'``, ``'year'``, or ``'unknown'``. ``'unknown'`` for
    cumulative/active/prevalence counts, which have no window."""

    surveillance_method: Optional[str] = None
    """Surveillance/ascertainment method, captured only when explicitly
    stated in the source (e.g. 'laboratory surveillance', 'syndromic
    surveillance', 'enhanced surveillance', 'passive reporting'). ``None``
    when not stated. Never inferred."""

    data_quality: Optional[str] = None
    """Explicit data-quality caveat stated in the source about how the
    reported numbers relate to reality: under-reporting, limited
    testing/ascertainment, reporting lag, suspected-vs-confirmed definition
    issues, or a surveillance/case-definition change (e.g. 'testing capacity
    limited; many mild cases not captured'). ``None`` when the source states
    no such caveat. Never inferred from a number alone — the forecasting
    stage decides what to do with it. Surfaced in the evidence digest even
    on metric-bearing records (whose ``summary`` the digest otherwise omits)."""

    event_date: Optional[datetime] = None
    """Date the fact pertains to (not the date it was reported).

    Canonicalised to the start of the period when only a partial date is
    known (e.g. ``"2026-01"`` → ``datetime(2026, 1, 1)``). Read together
    with ``event_date_precision`` to recover the original granularity.
    """

    event_date_precision: Optional[str] = None
    """Granularity of ``event_date``: ``"year"`` | ``"month"`` | ``"day"``,
    or ``None`` when no date was extracted. The dedup logic in the insight
    pipeline merges two records whose date buckets overlap at the coarser
    precision (e.g. a record with month precision 2026-01 merges with a
    day-precision record dated 2026-01-25)."""

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
