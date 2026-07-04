"""BM25 retrieval over document chunks.

BM25 is critical for finding numeric facts (case counts, dates, etc.)
that pure embedding similarity often misses.  The grant proposal
explicitly notes this — don't skip it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from rank_bm25 import BM25Okapi

from bioscancast.schemas import DocumentChunk


@dataclass
class ScoredChunk:
    """A chunk with a retrieval score and its origin metadata."""

    chunk: DocumentChunk
    document_id: str
    score: float
    score_source: str  # "bm25", "embedding", or "hybrid"


def _tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokenization for BM25."""
    return text.lower().split()


def top_k(
    query: str,
    chunks: list[DocumentChunk],
    k: int,
    *,
    document_id: str = "",
) -> list[ScoredChunk]:
    """Retrieve the top-k chunks by BM25 relevance to the query.

    Args:
        query: The search query (e.g., a forecast question).
        chunks: Candidate chunks to score.
        k: Number of top results to return.
        document_id: ID of the parent document (attached to ScoredChunk).

    Returns:
        Up to k ScoredChunks sorted by descending BM25 score.
    """
    if not chunks:
        return []

    # Build corpus from chunk text + heading for richer matching
    corpus = []
    for c in chunks:
        text = c.text
        if c.heading:
            text = c.heading + " " + text
        corpus.append(_tokenize(text))

    bm25 = BM25Okapi(corpus)
    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)

    scored = [
        ScoredChunk(
            chunk=chunks[i],
            document_id=document_id,
            score=float(scores[i]),
            score_source="bm25",
        )
        for i in range(len(chunks))
    ]

    scored.sort(key=lambda s: s.score, reverse=True)
    return scored[:k]
