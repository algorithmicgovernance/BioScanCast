from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Protocol


@dataclass
class SectionContent:
    """A single section/chunk extracted by a parser."""

    section_path: Optional[str]
    """Hierarchical heading path, e.g. 'Summary > Country reports > Uganda'."""

    page_number: Optional[int]
    """1-based page number (for PDFs)."""

    text: str
    """Plain-text content."""

    chunk_type: str
    """'prose', 'table', 'heading', 'list', 'caption', 'other'."""

    table_rows: Optional[List[List[str]]] = None
    """Row-major table data when chunk_type is 'table'."""


@dataclass
class ParsedContent:
    """Intermediate result from a parser, before conversion to Document."""

    raw_text: str
    """Full extracted text (for char-count and fallback)."""

    sections: List[SectionContent] = field(default_factory=list)
    """Ordered list of content sections."""

    title: Optional[str] = None
    language: Optional[str] = None
    page_count: Optional[int] = None
    published_date: Optional[datetime] = None
    is_partial: bool = False
    """True when the parser truncated content (e.g. PDF page cap)."""
    partial_reason: Optional[str] = None


class Parser(Protocol):
    def can_parse(self, content_type: str, content: bytes) -> bool: ...
    def parse(self, content: bytes, *, source_url: str) -> ParsedContent: ...
