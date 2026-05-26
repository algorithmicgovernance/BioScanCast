"""Legacy LLM client kept for the search stage's transitional period.

This module pre-dates ``bioscancast/llm/base.py`` and uses a simpler
``generate_json(prompt: str) -> dict`` signature. New code should use
``bioscancast.llm.base.LLMClient`` (with structured ``system/user/schema/
model/max_tokens`` arguments returning an ``LLMResponse``) and the
production client at ``bioscancast/llm/openai_client.py``.

Filtering and the insight stage already use the modern protocol. The
search stage (``bioscancast/stages/search_stage/``) is the only
remaining consumer of this module and will be migrated in a follow-up.
Do not add new callers here.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional, Protocol


class LLMClient(Protocol):
    def generate_json(self, prompt: str) -> dict: ...


class OpenAIClient:
    """Concrete LLM client using OpenAI's chat completions API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        seed: int = 42,
    ) -> None:
        import openai

        self._client = openai.OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self._model = model
        self._temperature = temperature
        self._seed = seed

    def generate_json(self, prompt: str) -> dict:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=self._temperature,
            seed=self._seed,
        )
        text = response.choices[0].message.content or "{}"
        return json.loads(text)
