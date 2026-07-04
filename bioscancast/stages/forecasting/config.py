"""Configuration for the forecasting stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ForecastingConfig:
    """Typed configuration for the forecasting pipeline.

    Defaults are production-ready and serialized into the run manifest, so
    a run is reproducible from its manifest alone.
    """

    model: str = "gpt-4o"
    """Strong model for the superforecaster reasoning call. Calibration
    tracks model capability more than prompt cleverness, so this is the
    one knob worth spending on."""

    ensemble_samples: int = 3
    """Number of independent reasoning samples to draw and aggregate.
    Research puts the calibration gain of a small ensemble at ~5-8% Brier
    over a single sample, with diminishing returns past a handful."""

    temperature: float = 0.7
    """Sampling temperature for the ensemble. Higher than the client
    default (0.2) on purpose: the ensemble only helps if its members
    actually differ. Each sample also gets a distinct seed."""

    seed: int = 42
    """Base seed. Sample ``i`` is drawn with ``seed + i`` so the ensemble
    is diverse but the run as a whole stays reproducible."""

    aggregation: str = "geometric_mean_of_odds"
    """How to pool the ensemble members. One of ``geometric_mean_of_odds``
    (log opinion pool; reduces to geometric mean of odds for binary),
    ``median``, or ``mean``."""

    extremize: float = 1.0
    """Extremizing strength applied via a ``p ** (1 + strength)``
    renormalization. The ensemble is systematically under-confident on the
    historical-replay benchmark (mean ~0.61 on the truth bucket where
    already correct at n=11), so sharpening helps — but only behind a gate
    (see ``extremize_gate``), since blanket sharpening also amplifies
    confident-but-wrong forecasts. Turned on (1.0) after the n=11 post-fix
    benchmark: gated extremizing cut mean Brier 0.489 → 0.455, robust across
    strength 0.5-2.0. Set to 0.0 to disable."""

    extremize_gate: Optional[float] = 0.5
    """Peak-probability gate for ``extremize``: sharpening is applied only
    when the pooled distribution's top option is already at or above this
    probability — concentrating it on decisive forecasts and leaving
    diffuse (more often wrong) ones alone. ``None`` extremizes
    unconditionally (worse on the benchmark). 0.5 was the robust winner in
    the offline analysis (``scripts/eval_forecast_calibration.py``); a
    calendar-based time-ramp was the worst schedule."""

    reasoning_max_tokens: int = 4096
    """Per-call output-token cap for the reasoning model. The structured
    rationale (reference class, drivers, etc.) plus the probability vector
    needs headroom; 4096 mirrors the insight stage's extraction cap."""

    max_evidence_records: int = 40
    """Cap on InsightRecords folded into the evidence digest, taken by
    recency then confidence. Keeps the prompt focused and the input-token
    cost bounded on questions with many extracted facts."""

    max_input_tokens_per_run: int = 1_000_000
    """Budget ceiling on cumulative input tokens for one forecast. The
    pipeline checks this between ensemble samples and stops early."""

    forecast_source: str = "bioscancast"
    """``forecast_source`` label on the evidence-based forecast rows. The
    eval stage groups and compares forecasts by this label."""

    emit_baseline: bool = True
    """When True, also emit a retrieval-free baseline forecast (model
    priors only) as a second source. The eval stage compares it against
    the evidence-based forecast to surface training-data leakage."""

    baseline_source: str = "bioscancast_baseline"
    """``forecast_source`` label on the retrieval-free baseline rows."""

    baseline_model: str = "gpt-4o-mini"
    """Model for the baseline call. The baseline is a small no-evidence
    JSON request, so a cheap model is plenty (matches contamination.py)."""

    @classmethod
    def from_dict(cls, d: dict) -> "ForecastingConfig":
        """Create a ForecastingConfig from a dict, ignoring unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})
