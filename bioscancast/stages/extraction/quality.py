"""Post-extraction quality guard — flag silent-truncation extractions.

The extraction pipeline marks any fetch that returns bytes and produces at
least one chunk ``status="success"`` (see ``pipeline.py``), regardless of
whether the *content that mattered* was actually captured. In practice this
means a healthy-looking "N/N success" stage line can hide a total retrieval
miss: a WHO emergencies *index* page (263 chars of nav links), a paywall
snippet, or a JS-rendered data page whose figures never made it into the
static scrape (#34, mpox).

This module adds a measurement-only guard: it flags ``success`` documents
whose extracted text is suspiciously thin so the orchestrator can surface
them. It never drops or mutates a document — the forecast pipeline sees the
same inputs whether or not flagging is on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from bioscancast.schemas.document import Document


@dataclass(frozen=True)
class ThinExtractionWarning:
    """A ``success`` document whose char count fell below the floor."""

    doc_id: str
    result_id: str
    domain: str
    source_url: str
    document_type: str
    char_count: int
    min_chars: int

    def describe(self) -> str:
        return (
            f"{self.domain} ({self.char_count} chars < {self.min_chars}) "
            f"{self.source_url}"
        )

    def as_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "result_id": self.result_id,
            "domain": self.domain,
            "source_url": self.source_url,
            "document_type": self.document_type,
            "char_count": self.char_count,
            "min_chars": self.min_chars,
            "reason": "thin_extraction",
        }


def flag_thin_extractions(
    documents: List[Document], *, min_chars: int
) -> List[ThinExtractionWarning]:
    """Return a warning for each ``success`` document thinner than ``min_chars``.

    ``min_chars <= 0`` disables flagging (returns an empty list). Failed and
    partial documents are skipped — their ``error_message`` already records
    why they're short.
    """
    if min_chars <= 0:
        return []

    warnings: List[ThinExtractionWarning] = []
    for doc in documents:
        if doc.status != "success":
            continue
        char_count = doc.char_count or 0
        if char_count < min_chars:
            warnings.append(
                ThinExtractionWarning(
                    doc_id=doc.id,
                    result_id=doc.result_id,
                    domain=doc.domain,
                    source_url=doc.source_url,
                    document_type=doc.document_type,
                    char_count=char_count,
                    min_chars=min_chars,
                )
            )
    return warnings
