"""Fake LLM client for deterministic testing without network calls.

The FakeLLMClient replaces the real OpenAI client in all tests.
It returns scripted responses and deterministic embeddings so that
retrieval and extraction tests are fully reproducible.
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections import deque
from typing import Sequence

from .base import LLMResponse


class FakeLLMClient:
    """Scripted LLM client for testing.

    Responses are consumed in FIFO order.  If the queue is exhausted,
    a RuntimeError is raised — failing loudly beats returning empty dicts.

    ``generate_json`` is thread-safe under multiple concurrent callers
    (e.g. when the insight pipeline's per-chunk ThreadPoolExecutor calls
    it from several workers): a single ``threading.Lock`` protects the
    response deque and the call-count / token totals. Without this, the
    deque popleft + counter increment race and tests run under concurrency
    can drop responses or under-count tokens.
    """

    def __init__(
        self,
        responses: Sequence[LLMResponse] | None = None,
        *,
        embedding_dim: int = 64,
    ) -> None:
        self._responses: deque[LLMResponse] = deque(responses or [])
        self._embedding_dim = embedding_dim
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._lock = threading.Lock()

    def enqueue(self, *responses: LLMResponse) -> None:
        """Add responses to the end of the queue."""
        with self._lock:
            self._responses.extend(responses)

    def generate_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        model: str,
        max_tokens: int = 1024,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> LLMResponse:
        # temperature/seed are accepted for protocol compatibility but
        # ignored: scripted responses are deterministic by construction.
        with self._lock:
            if not self._responses:
                raise RuntimeError(
                    f"FakeLLMClient: no scripted responses left "
                    f"(call #{self.call_count + 1}). "
                    f"Enqueue more responses before running the test."
                )
            response = self._responses.popleft()
            self.call_count += 1
            self.total_input_tokens += response.input_tokens
            self.total_output_tokens += response.output_tokens
        return response

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        """Return deterministic pseudo-embeddings based on text hashes.

        Each text is hashed to produce a reproducible embedding vector.
        Vectors are normalized to unit length so cosine similarity works
        correctly in retrieval tests.
        """
        embeddings = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            raw = [
                float(b) / 255.0
                for b in digest[: self._embedding_dim]
            ]
            # Pad if digest is shorter than embedding_dim
            while len(raw) < self._embedding_dim:
                extra = hashlib.sha256(
                    text.encode("utf-8") + len(raw).to_bytes(4, "big")
                ).digest()
                raw.extend(float(b) / 255.0 for b in extra)
            raw = raw[: self._embedding_dim]
            # Normalize to unit vector
            norm = math.sqrt(sum(x * x for x in raw))
            if norm > 0:
                raw = [x / norm for x in raw]
            embeddings.append(raw)
        return embeddings
