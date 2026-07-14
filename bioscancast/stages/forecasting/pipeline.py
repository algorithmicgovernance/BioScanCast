"""Forecasting pipeline orchestrator.

Turns a forecast question plus the insight stage's structured facts into a
probability distribution over the question's option set — the shape the
eval stage scores.

Flow:
1. Build a compact evidence digest from the InsightRecords (pure Python).
2. Draw ``ensemble_samples`` independent superforecaster reasoning samples
   (one structured LLM call each, varied seed/temperature for diversity).
3. Aggregate the usable samples into one distribution (geometric mean of
   odds by default; optional extremizing).
4. Optionally emit a retrieval-free baseline forecast as a second source
   so the eval stage can quantify training-data leakage.
5. Return a ForecastResult (distributions + flat records + per-sample
   reasoning + budget).

Defensive throughout: a malformed or failed sample is dropped, not fatal;
if every sample fails the forecast degrades to a uniform distribution with
a note, mirroring the baseline path in contamination.py.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Sequence

from bioscancast.stages.filtering.models import ForecastQuestion
from bioscancast.stages.insight.budget import BudgetTracker
from bioscancast.llm.base import LLMClient, LLMResponse
from bioscancast.schemas import InsightRecord
from bioscancast.stages.evaluation.contamination import (
    retrieval_free_baseline_forecast,
)

from . import aggregation
from .config import ForecastingConfig
from .evidence import build_evidence_digest
from .prompts import build_forecast_prompt
from .schemas import (
    ForecastDistribution,
    ForecastRecord,
    ForecastResult,
    SampleForecast,
)

logger = logging.getLogger(__name__)


class _BudgetRecordingClient:
    """Wraps an LLMClient so every call is recorded in a BudgetTracker.

    Used for the baseline call, which goes through
    ``retrieval_free_baseline_forecast`` and so doesn't hand the response
    back for manual budget accounting.
    """

    def __init__(self, inner: LLMClient, budget: BudgetTracker) -> None:
        self._inner = inner
        self._budget = budget

    def generate_json(self, **kwargs) -> LLMResponse:
        response = self._inner.generate_json(**kwargs)
        self._budget.record(response)
        return response

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        return self._inner.embed(texts, model=model)


def _coerce_probabilities(
    raw: object, n_options: int
) -> Optional[List[float]]:
    """Validate and normalize a model-supplied probability list.

    Returns ``None`` (caller drops the sample) when the list is the wrong
    length, non-numeric, negative, non-finite, or sums to <= 0.
    """
    if not isinstance(raw, list) or len(raw) != n_options:
        return None
    try:
        probs = [float(p) for p in raw]
    except (TypeError, ValueError):
        return None
    if any(p < 0 or math.isnan(p) or math.isinf(p) for p in probs):
        return None
    total = sum(probs)
    if total <= 0:
        return None
    return [p / total for p in probs]


def _make_records(
    question_id: str,
    source: str,
    options: Sequence[str],
    probabilities: Sequence[float],
) -> List[ForecastRecord]:
    return [
        ForecastRecord(
            question_id=question_id,
            forecast_source=source,
            option=opt,
            probability=prob,
        )
        for opt, prob in zip(options, probabilities)
    ]


def _make_distribution(
    question_id: str,
    source: str,
    options: Sequence[str],
    probabilities: Sequence[float],
) -> ForecastDistribution:
    return ForecastDistribution(
        question_id=question_id,
        forecast_source=source,
        probabilities=dict(zip(options, probabilities)),
    )


class ForecastingPipeline:
    """Orchestrates ensemble reasoning and aggregation for one question."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        config: Optional[ForecastingConfig] = None,
    ) -> None:
        self._llm = llm_client
        self._config = config or ForecastingConfig()

    def run(
        self,
        question: ForecastQuestion,
        records: List[InsightRecord],
        options: List[str],
        historical_context: str | None = None,
        historical_insight_context: str | None = None,
    ) -> ForecastResult:
        """Produce a forecast distribution over ``options`` for ``question``.

        Args:
            question: The forecast question.
            records: InsightRecords from the insight stage.
            options: The mutually exclusive, exhaustive answer options
                (e.g. ``["YES", "NO"]`` or range bins).

        Returns:
            ForecastResult with one distribution per forecast_source.
        """
        config = self._config
        if not options:
            raise ValueError("Forecasting requires a non-empty option list.")

        result = ForecastResult(question_id=question.id, options=list(options))
        budget = BudgetTracker()

        evidence_digest, evidence_ids = build_evidence_digest(
            records, max_records=config.max_evidence_records
        )
        result.evidence_record_ids = evidence_ids
        if not evidence_digest:
            result.notes.append(
                "No usable evidence records; forecast relies on model priors."
            )

        # --- Ensemble of superforecaster reasoning samples ---
        system, user, schema = build_forecast_prompt(
            question,
            options,
            evidence_digest,
            historical_context,
            historical_insight_context,
        )
        usable_vectors: List[List[float]] = []
        for i in range(config.ensemble_samples):
            if budget.would_exceed(config.max_input_tokens_per_run):
                result.notes.append(
                    f"Budget exceeded ({budget.total_input_tokens} input "
                    f"tokens); stopped after {i} of "
                    f"{config.ensemble_samples} samples."
                )
                break

            sample_seed = config.seed + i
            try:
                response = self._llm.generate_json(
                    system=system,
                    user=user,
                    schema=schema,
                    model=config.model,
                    max_tokens=config.reasoning_max_tokens,
                    temperature=config.temperature,
                    seed=sample_seed,
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Forecast sample %d failed for question %s", i, question.id
                )
                result.samples.append(
                    SampleForecast(ok=False, model=config.model, seed=sample_seed)
                )
                continue

            budget.record(response)
            content = response.content or {}
            probs = _coerce_probabilities(content.get("probabilities"), len(options))
            sample = SampleForecast(
                ok=probs is not None,
                probabilities=probs or [],
                reference_class=content.get("reference_class"),
                base_rate=content.get("base_rate"),
                drivers_up=content.get("drivers_up") or [],
                drivers_down=content.get("drivers_down") or [],
                why_might_be_wrong=content.get("why_might_be_wrong"),
                rationale=content.get("rationale"),
                model=response.model,
                seed=sample_seed,
            )
            result.samples.append(sample)
            if probs is not None:
                usable_vectors.append(probs)
            else:
                logger.warning(
                    "Forecast sample %d for question %s returned malformed "
                    "probabilities: %r",
                    i, question.id, content.get("probabilities"),
                )

        # --- Aggregate ---
        if usable_vectors:
            primary = aggregation.aggregate(
                usable_vectors,
                config.aggregation,
                extremize_strength=config.extremize,
                extremize_gate=config.extremize_gate,
            )
        else:
            primary = [1.0 / len(options)] * len(options)
            result.notes.append(
                "All ensemble samples failed; evidence-based forecast "
                "fell back to a uniform distribution."
            )

        result.distributions.append(
            _make_distribution(question.id, config.forecast_source, options, primary)
        )
        result.records.extend(
            _make_records(question.id, config.forecast_source, options, primary)
        )

        # --- Retrieval-free baseline (training-data-leakage diagnostic) ---
        if config.emit_baseline:
            baseline = retrieval_free_baseline_forecast(
                question,
                options,
                _BudgetRecordingClient(self._llm, budget),
                model=config.baseline_model,
            )
            result.baseline_rationale = baseline.rationale
            result.distributions.append(
                _make_distribution(
                    question.id,
                    config.baseline_source,
                    baseline.options,
                    baseline.probabilities,
                )
            )
            result.records.extend(
                _make_records(
                    question.id,
                    config.baseline_source,
                    baseline.options,
                    baseline.probabilities,
                )
            )

        result.budget_summary = budget.summary()
        return result
