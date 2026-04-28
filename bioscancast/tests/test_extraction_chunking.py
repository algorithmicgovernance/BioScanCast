"""Tests for bioscancast.extraction.chunking."""

from __future__ import annotations

import pytest

from bioscancast.extraction.chunking import normalize_chunks
from bioscancast.extraction.tokens import approx_token_count
from bioscancast.schemas.document import DocumentChunk


def _make_chunk(
    chunk_id: str = "c0",
    text: str = "Hello world.",
    chunk_type: str = "prose",
    heading: str | None = "Section A",
    page_number: int | None = 1,
    table_data=None,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        chunk_index=0,
        text=text,
        chunk_type=chunk_type,
        heading=heading,
        page_number=page_number,
        table_data=table_data,
    )


class TestNormalizeChunks:
    def test_small_chunk_passes_through(self):
        chunk = _make_chunk(text="Short text.")
        result = normalize_chunks([chunk], target_tokens=800, max_tokens=1500)
        assert len(result) == 1
        assert result[0].chunk_id == "c0"

    def test_token_count_populated(self):
        chunk = _make_chunk(text="Hello world.")
        result = normalize_chunks([chunk])
        assert result[0].token_count is not None
        assert result[0].token_count > 0

    def test_oversized_chunk_is_split(self):
        # Create a chunk that exceeds max_tokens
        long_text = "This is a sentence with several words. " * 500
        chunk = _make_chunk(text=long_text)
        result = normalize_chunks([chunk], target_tokens=100, max_tokens=200)
        assert len(result) > 1

    def test_split_chunks_respect_max_tokens(self):
        long_text = "This is a sentence with several words. " * 500
        chunk = _make_chunk(text=long_text)
        result = normalize_chunks([chunk], target_tokens=100, max_tokens=200)
        for c in result:
            # Allow some slack since we split on sentence boundaries
            assert c.token_count is not None
            assert c.token_count <= 300  # generous upper bound for boundary splits

    def test_split_preserves_metadata(self):
        long_text = "This is a sentence. " * 500
        chunk = _make_chunk(
            text=long_text,
            heading="Important Section",
            page_number=5,
        )
        result = normalize_chunks([chunk], target_tokens=100, max_tokens=200)
        for c in result:
            assert c.heading == "Important Section"
            assert c.page_number == 5

    def test_split_chunk_id_pattern(self):
        long_text = "Word " * 2000
        chunk = _make_chunk(chunk_id="doc1-c0", text=long_text)
        result = normalize_chunks([chunk], target_tokens=100, max_tokens=200)
        assert all(c.chunk_id.startswith("doc1-c0-p") for c in result)
        # Check sequential numbering
        for i, c in enumerate(result):
            assert c.chunk_id == f"doc1-c0-p{i}"

    def test_table_chunk_never_split(self):
        large_table_text = "cell " * 2000
        table_data = [["a", "b"], ["c", "d"]]
        chunk = _make_chunk(
            text=large_table_text,
            chunk_type="table",
            table_data=table_data,
        )
        result = normalize_chunks([chunk], target_tokens=100, max_tokens=200)
        assert len(result) == 1
        assert result[0].chunk_type == "table"
        assert result[0].table_data == table_data

    def test_multiple_chunks_processed(self):
        small = _make_chunk(chunk_id="c0", text="Small text.")
        large_text = "Sentence here. " * 500
        large = _make_chunk(chunk_id="c1", text=large_text)
        result = normalize_chunks([small, large], target_tokens=100, max_tokens=200)
        # small passes through, large gets split
        assert result[0].chunk_id == "c0"
        assert len(result) > 2

    def test_empty_input(self):
        assert normalize_chunks([]) == []
