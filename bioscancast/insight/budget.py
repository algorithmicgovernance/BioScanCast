"""Token budget tracking for the insight pipeline.

The single most important thing the insight stage does is keep token
costs honest.  The grant proposal budgets ~$100 of LLM credits.
The pipeline must never make an LLM call without going through
BudgetTracker.
"""

from __future__ import annotations

from collections import defaultdict

from bioscancast.llm.base import LLMResponse


class BudgetTracker:
    """Tracks cumulative token usage across a pipeline run.

    Every LLM call in the insight pipeline must be recorded here.
    The pipeline checks ``would_exceed()`` between documents and
    stops gracefully if the budget is hit.
    """

    def __init__(self) -> None:
        self._input_tokens: dict[str, int] = defaultdict(int)
        self._output_tokens: dict[str, int] = defaultdict(int)
        self._call_count: dict[str, int] = defaultdict(int)

    @property
    def total_input_tokens(self) -> int:
        return sum(self._input_tokens.values())

    @property
    def total_output_tokens(self) -> int:
        return sum(self._output_tokens.values())

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def record(self, response: LLMResponse) -> None:
        """Record token usage from an LLM response."""
        self._input_tokens[response.model] += response.input_tokens
        self._output_tokens[response.model] += response.output_tokens
        self._call_count[response.model] += 1

    def would_exceed(self, limit: int) -> bool:
        """Check whether total input tokens have exceeded the limit.

        The limit is on *input* tokens specifically because that's
        the dominant cost driver (retrieval chunks sent to the model).
        """
        return self.total_input_tokens >= limit

    def summary(self) -> dict:
        """Return a summary dict of token usage by model."""
        models = set(self._input_tokens) | set(self._output_tokens)
        per_model = {}
        for model in sorted(models):
            per_model[model] = {
                "input_tokens": self._input_tokens[model],
                "output_tokens": self._output_tokens[model],
                "calls": self._call_count[model],
            }
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "per_model": per_model,
        }
