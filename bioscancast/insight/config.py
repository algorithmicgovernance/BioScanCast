"""Configuration for the insight stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


INSIGHT_CONFIG = {
    "retrieval_top_k": 12,
    "bm25_weight": 0.5,
    "embedding_weight": 0.5,
    "cheap_model": "gpt-4o-mini",
    "embedding_model": "text-embedding-3-small",
    "max_input_tokens_per_run": 500_000,
    "max_chunks_per_document": 12,
    "extraction_max_output_tokens": 4096,
    "chunk_workers": 6,
}


@dataclass
class InsightConfig:
    """Typed configuration for the insight pipeline."""

    retrieval_top_k: int = 12
    bm25_weight: float = 0.5
    embedding_weight: float = 0.5
    cheap_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    max_input_tokens_per_run: int = 500_000
    max_chunks_per_document: int = 12
    extraction_max_output_tokens: int = 4096
    """Per-call cap on LLM output tokens for chunk extraction. The default
    1024 ceiling in LLMClient.generate_json truncates dense pages (e.g. the
    ECDC CDTR) mid-JSON; 4096 leaves comfortable headroom."""

    chunk_workers: int = 6
    """Max parallel per-chunk extraction calls per document. Each call is
    a single OpenAI chat-completions request; the SDK is thread-safe so a
    ThreadPoolExecutor is used. Six matches typical retrieval_top_k while
    staying well below OpenAI's per-minute rate limits for gpt-4o-mini.
    Set to 1 for sequential execution (useful for debugging or rate-
    limit-sensitive setups)."""

    @classmethod
    def from_dict(cls, d: dict) -> InsightConfig:
        """Create an InsightConfig from a dict, ignoring unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})
