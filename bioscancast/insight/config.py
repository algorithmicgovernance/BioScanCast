"""Configuration for the insight stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


INSIGHT_CONFIG = {
    "retrieval_top_k": 12,
    "bm25_weight": 0.5,
    "embedding_weight": 0.5,
    "cheap_model": "gpt-4o-mini",
    "strong_model": "gpt-4o",
    "use_strong_model_refinement": False,
    "embedding_model": "text-embedding-3-small",
    "max_input_tokens_per_run": 500_000,
    "max_chunks_per_document": 12,
}


@dataclass
class InsightConfig:
    """Typed configuration for the insight pipeline."""

    retrieval_top_k: int = 12
    bm25_weight: float = 0.5
    embedding_weight: float = 0.5
    cheap_model: str = "gpt-4o-mini"
    strong_model: str = "gpt-4o"
    use_strong_model_refinement: bool = False
    embedding_model: str = "text-embedding-3-small"
    max_input_tokens_per_run: int = 500_000
    max_chunks_per_document: int = 12

    @classmethod
    def from_dict(cls, d: dict) -> InsightConfig:
        """Create an InsightConfig from a dict, ignoring unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})
