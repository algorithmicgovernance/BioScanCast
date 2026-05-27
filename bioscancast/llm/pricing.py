"""USD price table for OpenAI models and a per-call cost estimator.

The orchestrator uses this to surface estimated cost per pipeline run.
Prices are a point-in-time snapshot — refresh when OpenAI changes rates or
when new model identifiers enter the stage configs.

Snapshot taken 2026-05-27 from OpenAI's public API pricing pages
(https://devtk.ai/en/models/gpt-4o-mini/ and
https://www.cloudzero.com/blog/openai-pricing/). All numbers are USD per
1,000,000 tokens. The cached-input rate is OpenAI's standard 50% discount
on the cached prefix of an input; embedding models have no separate cached
rate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    """USD per 1,000,000 tokens for one model."""

    input: float
    cached_input: float
    output: float


MODEL_PRICES: dict[str, ModelPrice] = {
    # Cheap chat workhorse — used by search (query decomposition + filter
    # rescue) and insight (chunk extraction).
    "gpt-4o-mini": ModelPrice(input=0.15, cached_input=0.075, output=0.60),
    # Strong model — scaffolded for issue #26 refinement but not in
    # production use as of 2026-05-27.
    "gpt-4o": ModelPrice(input=2.50, cached_input=1.25, output=10.00),
    "gpt-4o-2024-08-06": ModelPrice(input=2.50, cached_input=1.25, output=10.00),
    # Embeddings (insight retrieval).
    "text-embedding-3-small": ModelPrice(input=0.02, cached_input=0.02, output=0.0),
    "text-embedding-3-large": ModelPrice(input=0.13, cached_input=0.13, output=0.0),
}


class UnknownModelError(KeyError):
    """Raised when a model name is not in MODEL_PRICES."""


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float:
    """Estimate USD cost of an LLM call.

    Args:
        model: Identifier matching a key in MODEL_PRICES.
        input_tokens: Total input tokens (including any cached portion).
        output_tokens: Output tokens generated.
        cached_input_tokens: Subset of input_tokens that hit the prompt
            cache; must be <= input_tokens. The non-cached remainder is
            billed at the full input rate.

    Raises:
        UnknownModelError: If ``model`` is not in MODEL_PRICES — refresh
            this module when adding a new model to a stage config.
        ValueError: If cached_input_tokens > input_tokens or any token
            count is negative.
    """
    if input_tokens < 0 or output_tokens < 0 or cached_input_tokens < 0:
        raise ValueError("Token counts must be non-negative")
    if cached_input_tokens > input_tokens:
        raise ValueError(
            f"cached_input_tokens ({cached_input_tokens}) exceeds "
            f"input_tokens ({input_tokens})"
        )
    try:
        price = MODEL_PRICES[model]
    except KeyError as exc:
        raise UnknownModelError(
            f"Model {model!r} is not in MODEL_PRICES; refresh "
            f"bioscancast/llm/pricing.py with current rates."
        ) from exc

    fresh_input = input_tokens - cached_input_tokens
    return (
        fresh_input * price.input
        + cached_input_tokens * price.cached_input
        + output_tokens * price.output
    ) / 1_000_000.0


def estimate_cost_from_summary(summary: dict) -> float:
    """Estimate USD cost from an InsightRunResult.budget_summary-style dict.

    Expects a dict with a ``per_model`` key whose value is
    ``{model: {input_tokens, output_tokens, [cached_input_tokens]}}`` —
    the shape the insight pipeline already produces. Unknown models are
    skipped with a noisy KeyError so callers can decide whether to
    suppress or surface them.
    """
    per_model = summary.get("per_model") or {}
    total = 0.0
    for model, counts in per_model.items():
        total += estimate_cost(
            model,
            input_tokens=int(counts.get("input_tokens", 0)),
            output_tokens=int(counts.get("output_tokens", 0)),
            cached_input_tokens=int(counts.get("cached_input_tokens", 0)),
        )
    return total
