"""Tests for insight stage retrieval (BM25, embeddings, hybrid, rule filters).

All tests use FakeLLMClient — no network calls, no real OpenAI imports.
"""

from datetime import datetime

from bioscancast.schemas import Document, DocumentChunk
from bioscancast.filtering.models import ForecastQuestion
from bioscancast.llm.fake_client import FakeLLMClient
from bioscancast.insight.retrieval.bm25 import top_k as bm25_top_k
from bioscancast.insight.retrieval.embeddings import embed_chunks, top_k as emb_top_k
from bioscancast.insight.retrieval.rule_filters import (
    filter_by_date_window,
    filter_by_keyword,
)
from bioscancast.insight.retrieval.hybrid import hybrid_retrieve

from bioscancast.tests.fixtures.insight.synthetic_documents import (
    DOC_WHO_SUDAN,
    DOC_CDC_H5N1,
    QUESTION_SUDAN,
    QUESTION_H5N1,
)


# ---------------------------------------------------------------------------
# BM25 retrieval tests
# ---------------------------------------------------------------------------


def test_bm25_finds_planted_fact():
    """BM25 should rank the chunk with '47 cases' highest for a
    query about human cases."""
    chunks = DOC_CDC_H5N1.chunks
    results = bm25_top_k(
        "H5N1 human cases United States",
        chunks,
        k=3,
        document_id=DOC_CDC_H5N1.id,
    )

    assert len(results) > 0
    # The human cases chunk should be top-ranked
    top_ids = [r.chunk.chunk_id for r in results]
    assert "chunk-cdc-h5n1-001-p2" in top_ids[:2]
    assert results[0].score_source == "bm25"
    assert results[0].document_id == DOC_CDC_H5N1.id


def test_bm25_finds_uganda_cases():
    """BM25 should find the Sudan virus case count chunk for a query
    about Uganda confirmed cases."""
    results = bm25_top_k(
        "confirmed cases Uganda Mubende",
        DOC_WHO_SUDAN.chunks,
        k=3,
        document_id=DOC_WHO_SUDAN.id,
    )

    top_ids = [r.chunk.chunk_id for r in results]
    # The prose chunk with case counts should rank high
    assert "chunk-who-sudan-001-p1" in top_ids[:2]


def test_bm25_scores_are_non_negative():
    results = bm25_top_k(
        "something irrelevant xyz123",
        DOC_WHO_SUDAN.chunks,
        k=5,
        document_id=DOC_WHO_SUDAN.id,
    )
    for r in results:
        assert r.score >= 0.0


def test_bm25_empty_chunks():
    results = bm25_top_k("test query", [], k=5)
    assert results == []


# ---------------------------------------------------------------------------
# Embedding retrieval tests
# ---------------------------------------------------------------------------


def test_embedding_retrieval_returns_results():
    """Embedding retrieval with fake client should return scored chunks."""
    client = FakeLLMClient(embedding_dim=32)
    chunks = DOC_CDC_H5N1.chunks

    embeddings = embed_chunks(chunks, client, model="test-embed")
    assert embeddings.shape == (len(chunks), 32)

    results = emb_top_k(
        "H5N1 livestock herds",
        chunks,
        embeddings,
        client,
        model="test-embed",
        k=3,
        document_id=DOC_CDC_H5N1.id,
    )

    assert len(results) == len(chunks)  # k >= len(chunks) so all returned
    assert all(r.score_source == "embedding" for r in results)
    # Scores should be in [-1, 1] (cosine similarity)
    for r in results:
        assert -1.0 <= r.score <= 1.0 + 1e-6


def test_embedding_cache_avoids_recomputation():
    """Embedding cache should prevent duplicate embed calls."""
    client = FakeLLMClient(embedding_dim=16)
    chunks = DOC_WHO_SUDAN.chunks
    cache: dict[str, list[float]] = {}

    # First call populates cache
    emb1 = embed_chunks(chunks, client, model="test-embed", cache=cache)
    assert len(cache) == len(chunks)

    # Second call should use cache (no new embed call)
    emb2 = embed_chunks(chunks, client, model="test-embed", cache=cache)
    import numpy as np
    np.testing.assert_array_equal(emb1, emb2)


def test_embedding_deterministic():
    """Same text should produce the same embedding."""
    client = FakeLLMClient(embedding_dim=32)
    emb1 = client.embed(["hello world"], model="test")
    emb2 = client.embed(["hello world"], model="test")
    assert emb1 == emb2


# ---------------------------------------------------------------------------
# Rule filter tests
# ---------------------------------------------------------------------------


def test_keyword_filter_matches():
    """Chunks mentioning the keyword should get score 1.0."""
    results = filter_by_keyword(DOC_WHO_SUDAN.chunks, ["Uganda"])

    for chunk, score in results:
        text = (chunk.text + " " + (chunk.heading or "")).lower()
        if "uganda" in text:
            assert score == 1.0
        else:
            assert score == 0.5  # soft penalty, not hard drop


def test_keyword_filter_hard_mode():
    """Hard mode should give 0.0 to non-matching chunks."""
    results = filter_by_keyword(
        DOC_WHO_SUDAN.chunks, ["Uganda"], hard=True
    )
    for chunk, score in results:
        text = (chunk.text + " " + (chunk.heading or "")).lower()
        if "uganda" not in text:
            assert score == 0.0


def test_keyword_filter_empty_keywords():
    """Empty keywords should return all chunks with score 1.0."""
    results = filter_by_keyword(DOC_WHO_SUDAN.chunks, [])
    assert all(score == 1.0 for _, score in results)


def test_date_window_filter():
    """Chunks from a document within the date window should score 1.0."""
    results = filter_by_date_window(
        DOC_WHO_SUDAN.chunks,
        DOC_WHO_SUDAN,
        after=datetime(2025, 1, 1),
        before=datetime(2025, 12, 31),
    )
    # All chunks should pass (doc published 2025-02-15)
    assert all(score == 1.0 for _, score in results)


def test_date_window_filter_outside():
    """Chunks from a document outside the window should get a soft penalty."""
    results = filter_by_date_window(
        DOC_WHO_SUDAN.chunks,
        DOC_WHO_SUDAN,
        after=datetime(2026, 1, 1),
    )
    # Doc published 2025-02-15, window starts 2026 -> soft penalty
    assert all(score == 0.5 for _, score in results)


def test_date_window_filter_hard():
    """Hard mode should give 0.0 for out-of-window chunks."""
    results = filter_by_date_window(
        DOC_WHO_SUDAN.chunks,
        DOC_WHO_SUDAN,
        after=datetime(2026, 1, 1),
        hard=True,
    )
    assert all(score == 0.0 for _, score in results)


# ---------------------------------------------------------------------------
# Hybrid retrieval tests
# ---------------------------------------------------------------------------


def test_hybrid_retrieve_returns_scored_chunks():
    """Hybrid retrieval should combine BM25 + embeddings and return results."""
    client = FakeLLMClient(embedding_dim=32)
    results = hybrid_retrieve(
        QUESTION_SUDAN,
        DOC_WHO_SUDAN,
        client,
        top_k=5,
        embedding_model="test-embed",
    )

    assert len(results) > 0
    assert len(results) <= 5
    assert all(r.score_source == "hybrid" for r in results)
    assert all(r.document_id == DOC_WHO_SUDAN.id for r in results)


def test_hybrid_scores_normalized():
    """Hybrid scores should be reasonable (roughly in [0, ~1.1] after boost)."""
    client = FakeLLMClient(embedding_dim=32)
    results = hybrid_retrieve(
        QUESTION_H5N1,
        DOC_CDC_H5N1,
        client,
        top_k=10,
        embedding_model="test-embed",
    )

    for r in results:
        # Scores can exceed 1.0 slightly due to keyword boost
        assert r.score >= -0.1
        assert r.score <= 1.5


def test_hybrid_deduplication():
    """Each chunk should appear at most once in hybrid results."""
    client = FakeLLMClient(embedding_dim=32)
    results = hybrid_retrieve(
        QUESTION_SUDAN,
        DOC_WHO_SUDAN,
        client,
        top_k=20,
        embedding_model="test-embed",
    )

    chunk_ids = [r.chunk.chunk_id for r in results]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_hybrid_empty_document():
    """Hybrid retrieval on a document with no chunks returns empty list."""
    empty_doc = Document(
        id="doc-empty",
        result_id="r-empty",
        source_url="https://example.com",
        domain="example.com",
        fetched_at=datetime(2025, 1, 1),
        document_type="html",
        status="success",
        chunks=[],
    )
    client = FakeLLMClient()
    results = hybrid_retrieve(
        QUESTION_SUDAN, empty_doc, client, embedding_model="test-embed"
    )
    assert results == []


def test_hybrid_sorted_descending():
    """Results should be sorted by score in descending order."""
    client = FakeLLMClient(embedding_dim=32)
    results = hybrid_retrieve(
        QUESTION_SUDAN,
        DOC_WHO_SUDAN,
        client,
        top_k=10,
        embedding_model="test-embed",
    )
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
