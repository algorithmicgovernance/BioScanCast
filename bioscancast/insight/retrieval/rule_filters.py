"""Pre-retrieval rule-based filters for shrinking the candidate pool.

These are cheap and run before BM25/embeddings.  Critically, they
are **soft by default** -- they produce bias scores that the hybrid
retriever uses as re-rank boosts, not hard cuts.  Hard-dropping
chunks risks silently losing important context (e.g. a chunk that
mentions "the outbreak" without re-naming the pathogen).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from bioscancast.schemas import Document, DocumentChunk


def filter_by_date_window(
    chunks: list[DocumentChunk],
    document: Document,
    *,
    after: Optional[datetime] = None,
    before: Optional[datetime] = None,
    hard: bool = False,
) -> list[tuple[DocumentChunk, float]]:
    """Score chunks by date relevance within a time window.

    Uses the document's published_date as a proxy for all chunks
    (chunk-level dates are not currently tracked).

    Args:
        chunks: Candidate chunks.
        document: Parent document (for published_date).
        after: Only consider content after this date.
        before: Only consider content before this date.
        hard: If True, return score 0.0 for out-of-window chunks
            (effectively dropping them). Default False returns a
            reduced score instead.

    Returns:
        List of (chunk, score) pairs where score is in [0, 1].
        1.0 = within window, 0.5 = soft penalty, 0.0 = hard drop.
    """
    results = []
    pub_date = document.published_date

    for chunk in chunks:
        if pub_date is None:
            # No date info -- assume relevant (neutral score)
            results.append((chunk, 0.75))
            continue

        in_window = True
        if after and pub_date < after:
            in_window = False
        if before and pub_date > before:
            in_window = False

        if in_window:
            results.append((chunk, 1.0))
        elif hard:
            results.append((chunk, 0.0))
        else:
            results.append((chunk, 0.5))

    return results


def filter_by_keyword(
    chunks: list[DocumentChunk],
    keywords: list[str],
    *,
    hard: bool = False,
) -> list[tuple[DocumentChunk, float]]:
    """Score chunks by presence of any keyword (case-insensitive).

    Used for country/pathogen pre-screening.

    Args:
        chunks: Candidate chunks.
        keywords: Terms to search for (any-match).
        hard: If True, score 0.0 for non-matching chunks.

    Returns:
        List of (chunk, score) pairs. 1.0 = match found,
        0.5 = soft penalty, 0.0 = hard drop.
    """
    if not keywords:
        return [(c, 1.0) for c in chunks]

    lower_keywords = [kw.lower() for kw in keywords]
    results = []

    for chunk in chunks:
        text_lower = chunk.text.lower()
        heading_lower = (chunk.heading or "").lower()
        combined = text_lower + " " + heading_lower

        if any(kw in combined for kw in lower_keywords):
            results.append((chunk, 1.0))
        elif hard:
            results.append((chunk, 0.0))
        else:
            results.append((chunk, 0.5))

    return results
