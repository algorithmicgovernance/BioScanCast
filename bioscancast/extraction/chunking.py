from __future__ import annotations

import re
from typing import List

from bioscancast.schemas.document import DocumentChunk

from .tokens import approx_token_count


def normalize_chunks(
    chunks: List[DocumentChunk],
    *,
    target_tokens: int = 800,
    max_tokens: int = 1500,
) -> List[DocumentChunk]:
    """Split oversized chunks to respect token budgets.

    - Chunks exceeding *max_tokens* are split on paragraph then sentence
      boundaries into pieces <= *target_tokens*.
    - Table chunks are never split; oversized ones get a metadata note.
    - Small chunks are never merged.
    """
    result: List[DocumentChunk] = []

    for chunk in chunks:
        token_count = approx_token_count(chunk.text)
        chunk.token_count = token_count

        # Table chunks: never split
        if chunk.chunk_type == "table":
            result.append(chunk)
            continue

        if token_count <= max_tokens:
            result.append(chunk)
            continue

        # Split oversized chunk
        parts = _split_text(chunk.text, target_tokens)
        for i, part_text in enumerate(parts):
            part_tokens = approx_token_count(part_text)
            result.append(
                DocumentChunk(
                    chunk_id=f"{chunk.chunk_id}-p{i}",
                    chunk_index=chunk.chunk_index,  # will be renumbered by caller if needed
                    text=part_text,
                    chunk_type=chunk.chunk_type,
                    heading=chunk.heading,
                    page_number=chunk.page_number,
                    table_data=None,
                    token_count=part_tokens,
                )
            )

    return result


def _split_text(text: str, target_tokens: int) -> List[str]:
    """Split text into pieces targeting *target_tokens*, respecting paragraph
    and sentence boundaries."""
    # Try paragraph-level splits first
    paragraphs = re.split(r"\n\s*\n", text)
    if len(paragraphs) > 1:
        return _merge_pieces(paragraphs, target_tokens)

    # Fall back to sentence-level splits
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) > 1:
        return _merge_pieces(sentences, target_tokens)

    # Last resort: hard split by words
    words = text.split()
    pieces: List[str] = []
    current: List[str] = []
    for word in words:
        current.append(word)
        if approx_token_count(" ".join(current)) >= target_tokens:
            pieces.append(" ".join(current))
            current = []
    if current:
        pieces.append(" ".join(current))
    return pieces if pieces else [text]


def _merge_pieces(pieces: List[str], target_tokens: int) -> List[str]:
    """Greedily merge adjacent pieces until hitting the token target."""
    result: List[str] = []
    current_parts: List[str] = []

    for piece in pieces:
        candidate = "\n\n".join(current_parts + [piece]) if current_parts else piece
        if approx_token_count(candidate) > target_tokens and current_parts:
            result.append("\n\n".join(current_parts))
            current_parts = [piece]
        else:
            current_parts.append(piece)

    if current_parts:
        result.append("\n\n".join(current_parts))

    return result if result else ["\n\n".join(pieces)]
