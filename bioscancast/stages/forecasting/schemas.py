"""Result schemas for the forecasting stage.

The forecasting stage's output contract is owned by the eval stage: a
forecast is a probability distribution over a discrete option set. We
reuse the eval stage's ``ForecastRecord`` (the flat row the scorer and
CSV consume) and ``ForecastDistribution`` (option -> probability) rather
than reinventing them, so conformance is guaranteed by construction.

``SampleForecast`` and ``ForecastResult`` are forecasting-stage-internal:
they carry the per-sample reasoning and run metadata needed for audit and
postmortem, which the eval contract doesn't model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# Reused verbatim from the eval stage so the output matches what the
# scorer consumes. These are lightweight dataclasses with no heavy deps.
from bioscancast.stages.evaluation.schemas import (
    ForecastDistribution,
    ForecastRecord,
)

__all__ = [
    "ForecastDistribution",
    "ForecastRecord",
    "SampleForecast",
    "ForecastResult",
]


@dataclass
class SampleForecast:
    """One ensemble member: its probability vector plus the reasoning that
    produced it.

    Stored for auditability (the repo treats provenance as first-class)
    and to enable postmortems on systematic bias. ``ok`` is False when the
    model's output couldn't be parsed into a usable distribution; such
    samples are excluded from aggregation but kept for the record.
    """

    probabilities: List[float] = field(default_factory=list)
    """Normalized probability vector aligned by index to the option list."""

    ok: bool = True
    """Whether this sample parsed into a usable distribution."""

    reference_class: Optional[str] = None
    base_rate: Optional[float] = None
    drivers_up: List[str] = field(default_factory=list)
    drivers_down: List[str] = field(default_factory=list)
    why_might_be_wrong: Optional[str] = None
    rationale: Optional[str] = None
    model: Optional[str] = None
    seed: Optional[int] = None


@dataclass
class ForecastResult:
    """Result of a full forecasting pipeline run for one question."""

    question_id: str
    options: List[str]

    distributions: List[ForecastDistribution] = field(default_factory=list)
    """One per forecast_source — the evidence-based forecast and, when
    enabled, the retrieval-free baseline."""

    records: List[ForecastRecord] = field(default_factory=list)
    """Flat (question_id, forecast_source, option, probability) rows the
    eval stage / CSV consume. Derived from ``distributions``."""

    samples: List[SampleForecast] = field(default_factory=list)
    """The ensemble members behind the evidence-based forecast."""

    baseline_rationale: Optional[str] = None
    """The retrieval-free baseline's free-text rationale, captured so the
    no-evidence forecast can be compared against the evidence-based one.
    None when the baseline is disabled or returned no rationale."""

    evidence_record_ids: List[str] = field(default_factory=list)
    """IDs of the InsightRecords that fed the evidence digest."""

    budget_summary: dict = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
