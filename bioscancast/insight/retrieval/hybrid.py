"""Hybrid retrieval combining BM25 and embedding scores.

Runs both retrieval methods independently, normalizes scores to [0, 1],
and combines via weighted sum.  Rule filters are applied as soft
re-rank boosts, not hard cuts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from bioscancast.schemas import Document, DocumentChunk
from bioscancast.filtering.models import ForecastQuestion

from . import bm25 as bm25_mod
from . import embeddings as emb_mod
from .bm25 import ScoredChunk
from .rule_filters import filter_by_keyword

if TYPE_CHECKING:
    from bioscancast.llm.base import LLMClient


def _min_max_normalize(scores: list[float]) -> list[float]:
    """Normalize scores to [0, 1] using min-max scaling."""
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi == lo:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def hybrid_retrieve(
    question: ForecastQuestion,
    document: Document,
    llm_client: LLMClient,
    *,
    top_k: int = 12,
    bm25_weight: float = 0.5,
    embedding_weight: float = 0.5,
    embedding_model: str = "text-embedding-3-small",
    embedding_cache: Optional[dict[str, list[float]]] = None,
) -> list[ScoredChunk]:
    """Retrieve the most relevant chunks for a question from a document.

    Combines BM25 and embedding retrieval with optional keyword-based
    soft re-ranking.  Returns up to ``top_k`` chunks.

    Args:
        question: The forecast question driving retrieval.
        document: Source document containing chunks.
        llm_client: Client for embeddings.
        top_k: Maximum chunks to return.
        bm25_weight: Weight for BM25 scores in the hybrid combination.
        embedding_weight: Weight for embedding scores.
        embedding_model: Model to use for embeddings.
        embedding_cache: Optional shared cache for embedding vectors.

    Returns:
        List of ScoredChunks sorted by hybrid score, up to top_k.
    """
    chunks = document.chunks
    if not chunks:
        return []

    query = question.text

    # --- BM25 retrieval ---
    bm25_results = bm25_mod.top_k(
        query, chunks, k=len(chunks), document_id=document.id
    )

    # --- Embedding retrieval ---
    if embedding_cache is None:
        embedding_cache = {}
    chunk_embeddings = emb_mod.embed_chunks(
        chunks, llm_client, model=embedding_model, cache=embedding_cache
    )
    emb_results = emb_mod.top_k(
        query, chunks, chunk_embeddings, llm_client,
        model=embedding_model, k=len(chunks), document_id=document.id,
    )

    # --- Normalize scores to [0, 1] ---
    bm25_scores_raw = [sc.score for sc in bm25_results]
    emb_scores_raw = [sc.score for sc in emb_results]

    bm25_normed = _min_max_normalize(bm25_scores_raw)
    emb_normed = _min_max_normalize(emb_scores_raw)

    # Build lookup: chunk_id -> normalized score
    bm25_by_id = {
        bm25_results[i].chunk.chunk_id: bm25_normed[i]
        for i in range(len(bm25_results))
    }
    emb_by_id = {
        emb_results[i].chunk.chunk_id: emb_normed[i]
        for i in range(len(emb_results))
    }

    # --- Soft keyword re-rank boost ---
    keywords = []
    if question.pathogen:
        keywords.append(question.pathogen)
    if question.region:
        keywords.append(question.region)

    keyword_scores = {}
    if keywords:
        kw_results = filter_by_keyword(chunks, keywords)
        keyword_scores = {
            chunk.chunk_id: score for chunk, score in kw_results
        }

    # --- Combine into hybrid scores ---
    chunk_map = {c.chunk_id: c for c in chunks}
    hybrid_scored: dict[str, ScoredChunk] = {}

    for chunk_id, chunk in chunk_map.items():
        b_score = bm25_by_id.get(chunk_id, 0.0)
        e_score = emb_by_id.get(chunk_id, 0.0)
        combined = bm25_weight * b_score + embedding_weight * e_score

        # Apply keyword boost (up to +0.1)
        kw_score = keyword_scores.get(chunk_id, 0.5)
        combined += 0.1 * (kw_score - 0.5)

        if chunk_id in hybrid_scored:
            if combined > hybrid_scored[chunk_id].score:
                hybrid_scored[chunk_id] = ScoredChunk(
                    chunk=chunk,
                    document_id=document.id,
                    score=combined,
                    score_source="hybrid",
                )
        else:
            hybrid_scored[chunk_id] = ScoredChunk(
                chunk=chunk,
                document_id=document.id,
                score=combined,
                score_source="hybrid",
            )

    results = sorted(
        hybrid_scored.values(), key=lambda s: s.score, reverse=True
    )
    return results[:top_k]
