"""Tests for the thin-extraction guard (bioscancast.stages.extraction.quality)."""

from __future__ import annotations

from datetime import datetime, timezone

from bioscancast.stages.extraction.quality import (
    ThinExtractionWarning,
    flag_thin_extractions,
)
from bioscancast.schemas.document import Document


def _make_doc(
    *,
    doc_id: str = "doc-r1",
    result_id: str = "r1",
    domain: str = "who.int",
    char_count: int = 263,
    status: str = "success",
) -> Document:
    return Document(
        id=doc_id,
        result_id=result_id,
        source_url=f"https://{domain}/page",
        domain=domain,
        fetched_at=datetime.now(timezone.utc),
        document_type="html",
        status=status,
        char_count=char_count,
    )


def test_flags_thin_success_document():
    # The #34 culprit: WHO emergencies index, 263 chars, reported success.
    docs = [_make_doc(char_count=263)]
    warnings = flag_thin_extractions(docs, min_chars=500)
    assert len(warnings) == 1
    w = warnings[0]
    assert isinstance(w, ThinExtractionWarning)
    assert w.domain == "who.int"
    assert w.char_count == 263
    assert w.as_dict()["reason"] == "thin_extraction"


def test_does_not_flag_substantial_document():
    docs = [_make_doc(char_count=3576)]
    assert flag_thin_extractions(docs, min_chars=500) == []


def test_boundary_is_exclusive():
    # Exactly at the floor is not flagged; one below is.
    assert flag_thin_extractions([_make_doc(char_count=500)], min_chars=500) == []
    assert len(flag_thin_extractions([_make_doc(char_count=499)], min_chars=500)) == 1


def test_skips_non_success_documents():
    # Failed/partial docs already carry an error_message; don't double-flag.
    docs = [
        _make_doc(char_count=10, status="failed"),
        _make_doc(char_count=10, status="partial"),
    ]
    assert flag_thin_extractions(docs, min_chars=500) == []


def test_handles_none_char_count():
    docs = [_make_doc(char_count=None)]  # type: ignore[arg-type]
    warnings = flag_thin_extractions(docs, min_chars=500)
    assert len(warnings) == 1
    assert warnings[0].char_count == 0


def test_disabled_when_floor_non_positive():
    docs = [_make_doc(char_count=1)]
    assert flag_thin_extractions(docs, min_chars=0) == []
    assert flag_thin_extractions(docs, min_chars=-1) == []
