"""Prompt template for the superforecaster reasoning call.

One structured prompt encodes the high-leverage superforecaster
discipline (Tetlock / Good Judgment Project) and is sampled several times
to form an ensemble:

- Outside view first: state a reference class and base rate before
  adjusting (anchors against overconfidence).
- Decompose into a few tractable causal drivers, not deep recursion.
- Balance arguments for AND against (anti-confirmation bias).
- Consider explicitly why the forecast might be wrong (anti-overconfidence).
- Emit granular probabilities, not round-number hedges.

The output is always a probability distribution over the provided option
set — the single shape the eval stage scores (binary YES/NO or range
bins are both just option sets).
"""

from __future__ import annotations

from typing import List, Tuple

from bioscancast.stages.filtering.models import ForecastQuestion

FORECAST_SYSTEM_PROMPT = """\
You are a calibrated superforecaster assessing a biosecurity question. \
You follow the discipline of the best forecasters:

1. OUTSIDE VIEW FIRST. Identify a reference class of comparable past \
situations and state its base rate before looking at case specifics. Do \
not anchor on the most recent or most dramatic data point.
2. DECOMPOSE. Break the question into a few tractable drivers rather than \
reasoning in one leap.
3. BALANCE. List concrete reasons the outcome is MORE likely and concrete \
reasons it is LESS likely. Weigh disconfirming evidence as seriously as \
confirming evidence.
4. ADJUST DELIBERATELY. Move from the base rate only as far as the \
case-specific evidence justifies. Small, well-reasoned updates beat large \
swings.
5. STAY HUMBLE. State explicitly why your forecast might be wrong.
6. BE GRANULAR. Use specific probabilities (e.g. 0.62, not "likely"). \
Avoid defaulting to round numbers or 50/50.

EVIDENCE DISCIPLINE: Reason primarily from the EVIDENCE provided below \
plus general base-rate knowledge. The evidence was gathered as of the \
question's information cutoff; do not assume facts dated after it. If the \
evidence is thin, say so and lean harder on the outside view.

COUNT BASIS: Evidence lines may be tagged ``[cumulative]``, \
``[incident/<window>]`` (new cases in that period), or ``[active]``. These \
are NOT interchangeable — do not read a mix of cumulative totals, weekly \
new-case counts, and active-case counts as one comparable series. Work out \
which basis the question's target uses (from the resolution criteria — e.g. \
"cumulative confirmed cases") and reason primarily from evidence on that \
basis. Convert or down-weight other-basis figures rather than comparing them \
directly (e.g. a lone weekly increment is not the cumulative total).

DATA QUALITY: If the evidence explicitly mentions reporting lag, limited \
testing, under-reporting, a suspected-vs-confirmed definition change, or a \
surveillance change, weigh how it biases the reported numbers relative to the \
target. Do NOT assume large under-counting by default — adjust only on \
explicit signals in the evidence.

OUTPUT: Return JSON matching the schema. The "probabilities" array must \
have exactly one entry per option, in the SAME ORDER as the options \
given, and must sum to 1. The options are mutually exclusive and \
exhaustive."""


FORECAST_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "reference_class": {
            "type": "string",
            "description": "The class of comparable past situations used "
            "for the outside view.",
        },
        "base_rate": {
            "type": "number",
            "description": "Outside-view base rate for the most likely / "
            "positive outcome, before case-specific adjustment, in [0, 1].",
        },
        "drivers_up": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Concrete factors making the outcome more likely.",
        },
        "drivers_down": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Concrete factors making the outcome less likely.",
        },
        "why_might_be_wrong": {
            "type": "string",
            "description": "The strongest reason this forecast could be off.",
        },
        "rationale": {
            "type": "string",
            "description": "Brief synthesis justifying the final probabilities.",
        },
        "probabilities": {
            "type": "array",
            "items": {"type": "number"},
            "description": "One probability per option, same order, summing to 1.",
        },
    },
    "required": [
        "reference_class",
        "base_rate",
        "drivers_up",
        "drivers_down",
        "why_might_be_wrong",
        "rationale",
        "probabilities",
    ],
    "additionalProperties": False,
}


def build_forecast_prompt(
    question: ForecastQuestion,
    options: List[str],
    evidence_digest: str,
    historical_context: str | None = None,
) -> Tuple[str, str, dict]:
    """Build the (system, user, json_schema) tuple for one reasoning sample.

    Args:
        question: The forecast question.
        options: The mutually exclusive, exhaustive answer options.
        evidence_digest: The compact evidence text (may be empty).

    Returns:
        ``(system_prompt, user_prompt, json_schema)``.
    """
    parts: List[str] = [f"QUESTION: {question.text}"]

    if question.resolution_criteria:
        parts.append(f"RESOLUTION CRITERIA: {question.resolution_criteria}")
    if question.pathogen:
        parts.append(f"PATHOGEN: {question.pathogen}")
    if question.region:
        parts.append(f"REGION: {question.region}")
    if question.target_date:
        parts.append(f"TARGET DATE: {question.target_date.strftime('%Y-%m-%d')}")
    if question.as_of_date:
        parts.append(
            "INFORMATION CUTOFF (forecast as if today were): "
            f"{question.as_of_date.strftime('%Y-%m-%d')}"
        )

    parts.append("")
    numbered_options = "\n".join(
        f"  {i}. {opt}" for i, opt in enumerate(options)
    )
    parts.append(f"OPTIONS (assign a probability to each, in this order):\n{numbered_options}")

    if historical_context:
        parts.append("")
        parts.append(
            "FORECAST HISTORY (prior runs for this question; use as context, "
            "not as a substitute for current evidence):"
        )
        parts.append(historical_context)

    parts.append("")
    if evidence_digest:
        parts.append("EVIDENCE (most recent first):")
        parts.append(evidence_digest)
    else:
        parts.append(
            "EVIDENCE: None retrieved. Rely on the outside view and "
            "general base-rate knowledge, and lower your confidence "
            "accordingly."
        )

    return FORECAST_SYSTEM_PROMPT, "\n".join(parts), FORECAST_JSON_SCHEMA
