"""Contamination diagnostics for the human-comparison benchmark.

These metrics are *not* the same as proving fairness. They are reporting
aids: they let a reviewer see how much of the model's evidence base
demonstrably violated the cutoff, and how much of the model's edge over a
human came from training-data leakage rather than retrieval.

Two metrics live here:

* ``filter_caught_contamination_rate`` — a LOWER BOUND on contamination.
  Counts SearchResults whose ``published_date`` is later than the cutoff
  *and that nonetheless reached the final result list*. After the search
  stage's cutoff filter runs this should be ~0; in live-mode benchmark
  runs it can be substantial. Undated post-cutoff content is invisible
  to this metric — that is the largest source of contamination it cannot
  see, and reports MUST say so.

* ``retrieval_free_baseline_forecast`` — asks the LLM to forecast with no
  retrieved evidence, then scores it like any other forecast. A small gap
  between this baseline and the full pipeline is itself evidence of
  training-data leakage in the model's weights, distinct from leakage in
  the retrieval pipeline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional

from bioscancast.stages.filtering.models import ForecastQuestion, SearchResult
from bioscancast.llm.base import LLMClient

logger = logging.getLogger(__name__)


# The baseline call is a small JSON request — gpt-4o-mini is plenty. Both
# overridable per-call so eval harnesses can swap in a different model.
DEFAULT_BASELINE_MODEL = "gpt-4o-mini"
DEFAULT_BASELINE_MAX_TOKENS = 512

_BASELINE_SYSTEM_PROMPT = (
    "You forecast biosecurity questions using ONLY your prior knowledge. "
    "Do not assume any additional research has been done. Return JSON "
    "matching the schema."
)

_BASELINE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "probabilities": {
            "type": "array",
            "items": {"type": "number"},
        },
        "rationale": {"type": "string"},
    },
    "required": ["probabilities", "rationale"],
    "additionalProperties": False,
}


# Phrased verbosely on purpose: a non-coder reading the eval report should
# not confuse "filter-caught" with "absolute".
FILTER_CAUGHT_CONTAMINATION_LOWER_BOUND_LABEL = (
    "filter_caught_contamination_rate "
    "(LOWER BOUND — undated post-cutoff content is not counted)"
)


@dataclass
class ContaminationCounts:
    total: int
    post_cutoff_in_final: int
    undated_in_final: int
    pre_cutoff_in_final: int

    @property
    def filter_caught_rate(self) -> float:
        """Share of the final results whose dated publication is post-cutoff.

        IMPORTANT: this is a lower bound. It does not count undated results,
        and it does not count pages whose content was edited after the
        cutoff but whose first-publication date is pre-cutoff. Reports
        derived from this metric must surface that caveat.
        """
        if self.total == 0:
            return 0.0
        return self.post_cutoff_in_final / self.total


def filter_caught_contamination_rate(
    final_results: Iterable[SearchResult],
    as_of: datetime,
) -> ContaminationCounts:
    """Count post-cutoff results that nevertheless reached the final list.

    Pass the SearchResult list that the search stage *returned* (i.e. after
    its own cutoff filter ran). In a well-behaved historical-replay run the
    post_cutoff_in_final count should be 0; in a live-mode run it usually
    won't be.
    """
    post_cutoff = 0
    undated = 0
    pre_cutoff = 0
    total = 0
    for r in final_results:
        total += 1
        if r.published_date is None:
            undated += 1
        elif r.published_date > as_of:
            post_cutoff += 1
        else:
            pre_cutoff += 1
    return ContaminationCounts(
        total=total,
        post_cutoff_in_final=post_cutoff,
        undated_in_final=undated,
        pre_cutoff_in_final=pre_cutoff,
    )


@dataclass
class BaselineForecast:
    question_id: str
    options: List[str]
    probabilities: List[float]
    rationale: Optional[str] = None


def retrieval_free_baseline_forecast(
    question: ForecastQuestion,
    options: List[str],
    llm_client: LLMClient,
    *,
    model: str = DEFAULT_BASELINE_MODEL,
    max_tokens: int = DEFAULT_BASELINE_MAX_TOKENS,
) -> BaselineForecast:
    """Ask the LLM to forecast the question with NO retrieved evidence.

    The gap between this baseline and the full-pipeline forecast quantifies
    how much of the model's signal comes from retrieval vs. training-data
    knowledge. A small gap on a 2024 question to a 2026-trained LLM is
    strong evidence that the LLM already "knew the answer" — which is a
    separate fairness problem from retrieval leakage that no amount of
    pipeline filtering can fix. Report alongside Brier/log scores, never
    in place of them.
    """
    user_payload = json.dumps(
        {
            "task": (
                "Forecast the probability of each option for this biosecurity "
                "question. Probabilities must sum to 1."
            ),
            "question": question.text,
            "pathogen": question.pathogen,
            "region": question.region,
            "target_date": (
                question.target_date.isoformat() if question.target_date else None
            ),
            "as_of_date": (
                question.as_of_date.date().isoformat() if question.as_of_date else None
            ),
            "options": options,
        }
    )
    try:
        response = llm_client.generate_json(
            system=_BASELINE_SYSTEM_PROMPT,
            user=user_payload,
            schema=_BASELINE_SCHEMA,
            model=model,
            max_tokens=max_tokens,
        )
        result = response.content
    except Exception:
        logger.exception("Retrieval-free baseline LLM call failed for %s", question.id)
        uniform = [1.0 / len(options)] * len(options)
        return BaselineForecast(
            question_id=question.id,
            options=options,
            probabilities=uniform,
            rationale="LLM call failed; uniform fallback",
        )

    raw_probs = result.get("probabilities") or []
    if not isinstance(raw_probs, list) or len(raw_probs) != len(options):
        logger.warning(
            "Baseline LLM returned malformed probabilities for %s: %r",
            question.id, raw_probs,
        )
        uniform = [1.0 / len(options)] * len(options)
        return BaselineForecast(
            question_id=question.id,
            options=options,
            probabilities=uniform,
            rationale="Malformed LLM output; uniform fallback",
        )
    try:
        probs = [float(p) for p in raw_probs]
    except (TypeError, ValueError):
        uniform = [1.0 / len(options)] * len(options)
        return BaselineForecast(
            question_id=question.id,
            options=options,
            probabilities=uniform,
            rationale="Non-numeric LLM output; uniform fallback",
        )

    total = sum(probs)
    if total <= 0:
        uniform = [1.0 / len(options)] * len(options)
        return BaselineForecast(
            question_id=question.id,
            options=options,
            probabilities=uniform,
            rationale="Zero-sum LLM output; uniform fallback",
        )
    probs = [p / total for p in probs]
    return BaselineForecast(
        question_id=question.id,
        options=options,
        probabilities=probs,
        rationale=result.get("rationale"),
    )
