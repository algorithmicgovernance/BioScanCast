"""Real OpenAI-backed LLM client for production use.

This module is never imported in tests.  It requires the ``openai``
package and an ``OPENAI_API_KEY`` environment variable (or explicit key).
"""

from __future__ import annotations

import json
import os
from typing import Optional

from .base import LLMResponse


class OpenAILLMClient:
    """Production LLM client using the OpenAI API.

    Supports both chat completions (with JSON schema output) and
    embeddings.  The ``openai`` package is imported lazily so that
    the rest of the codebase can be loaded without it installed.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        seed: int = 42,
    ) -> None:
        import openai

        self._client = openai.OpenAI(
            api_key=api_key or os.environ["OPENAI_API_KEY"]
        )
        self._temperature = temperature
        self._seed = seed

    def generate_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        model: str,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        response = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction",
                    "strict": True,
                    "schema": schema,
                },
            },
            max_tokens=max_tokens,
            temperature=self._temperature,
            seed=self._seed,
        )
        raw_text = response.choices[0].message.content or "{}"
        content = json.loads(raw_text)
        usage = response.usage
        return LLMResponse(
            content=content,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=response.model,
            raw_text=raw_text,
        )

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        response = self._client.embeddings.create(
            model=model,
            input=texts,
        )
        return [item.embedding for item in response.data]
