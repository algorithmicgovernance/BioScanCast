"""Embedding-based retrieval over document chunks.

Deliberately uses an in-memory numpy array rather than a vector
database.  At the scale we operate at (dozens to low-hundreds of
chunks per question), a vector DB adds operational complexity
with no meaningful performance benefit.  If scale increases
significantly, revisit this decision.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bioscancast.schemas import DocumentChunk
from .bm25 import ScoredChunk

if TYPE_CHECKING:
    from bioscancast.llm.base import LLMClient


def embed_chunks(
    chunks: list[DocumentChunk],
    llm_client: LLMClient,
    *,
    model: str,
    cache: dict[str, list[float]] | None = None,
) -> np.ndarray:
    """Embed a list of chunks, using a cache to avoid re-embedding.

    Args:
        chunks: Chunks to embed.
        llm_client: Client with an embed() method.
        model: Embedding model identifier.
        cache: Optional dict mapping chunk_id -> embedding vector.
            Populated in-place for reuse within a single pipeline run.

    Returns:
        numpy array of shape (len(chunks), embedding_dim).
    """
    if cache is None:
        cache = {}

    texts_to_embed: list[str] = []
    indices_to_embed: list[int] = []

    for i, chunk in enumerate(chunks):
        if chunk.chunk_id not in cache:
            text = chunk.text
            if chunk.heading:
                text = chunk.heading + " " + text
            texts_to_embed.append(text)
            indices_to_embed.append(i)

    if texts_to_embed:
        new_embeddings = llm_client.embed(texts_to_embed, model=model)
        for idx, emb in zip(indices_to_embed, new_embeddings):
            cache[chunks[idx].chunk_id] = emb

    result = np.array([cache[c.chunk_id] for c in chunks])
    return result


def top_k(
    query: str,
    chunks: list[DocumentChunk],
    embeddings: np.ndarray,
    llm_client: LLMClient,
    *,
    model: str,
    k: int,
    document_id: str = "",
) -> list[ScoredChunk]:
    """Retrieve the top-k chunks by cosine similarity to the query embedding.

    Args:
        query: The search query text.
        chunks: Chunks corresponding to rows of ``embeddings``.
        embeddings: Pre-computed chunk embeddings (n_chunks, dim).
        llm_client: Client to embed the query.
        model: Embedding model identifier.
        k: Number of top results to return.
        document_id: Parent document ID.

    Returns:
        Up to k ScoredChunks sorted by descending cosine similarity.
    """
    if not chunks:
        return []

    query_emb = np.array(llm_client.embed([query], model=model)[0])

    # Cosine similarity: dot product of unit vectors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = embeddings / norms

    query_norm = np.linalg.norm(query_emb)
    if query_norm > 0:
        query_emb = query_emb / query_norm

    similarities = normed @ query_emb

    scored = [
        ScoredChunk(
            chunk=chunks[i],
            document_id=document_id,
            score=float(similarities[i]),
            score_source="embedding",
        )
        for i in range(len(chunks))
    ]

    scored.sort(key=lambda s: s.score, reverse=True)
    return scored[:k]
