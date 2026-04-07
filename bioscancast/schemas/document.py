from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class DocumentChunk:
    """A single chunk of content extracted from a source document.

    The extraction stage splits each fetched document into an ordered
    sequence of chunks so that downstream stages can reference specific
    passages rather than entire documents.
    """

    chunk_id: str
    """Stable identifier, unique within the parent document."""

    chunk_index: int
    """Zero-based position that preserves document reading order."""

    text: str
    """Plain-text content of this chunk."""

    chunk_type: str
    """Content category: 'prose', 'table', 'caption', 'list', 'heading', or 'other'."""

    heading: Optional[str] = None
    """Section heading or hierarchical path, e.g. 'Epidemiological summary > Country reports > Uganda'."""

    page_number: Optional[int] = None
    """Source page number (1-based), primarily useful for PDFs."""

    table_data: Optional[List[List[str]]] = None
    """Row-major table cells when chunk_type is 'table'.  Each inner list is one row."""

    token_count: Optional[int] = None
    """Approximate token count (tokeniser-dependent)."""


@dataclass
class Document:
    """A fetched and chunked document produced by the extraction stage.

    The extraction stage creates one Document per FilteredDocument it
    successfully (or unsuccessfully) processes.  A successful Document
    contains one or more DocumentChunk objects in reading order; a failed
    Document has status='failed', an error_message, and an empty chunks list.
    """

    # ---- identity / provenance ----
    id: str
    """Unique document identifier (UUID or deterministic hash)."""

    result_id: str
    """Foreign key to FilteredDocument.result_id from the filtering stage."""

    source_url: str
    """The URL that was actually fetched."""

    domain: str
    """Domain of the source (e.g. 'who.int')."""

    fetched_at: datetime
    """UTC timestamp of when the fetch was performed."""

    document_type: str
    """Format of the source material: 'html', 'pdf', or 'api_json'."""

    status: str
    """Fetch outcome: 'success' or 'failed'."""

    # ---- document-level metadata (optional) ----
    canonical_url: Optional[str] = None
    """Canonical URL if it differs from source_url."""

    title: Optional[str] = None
    """Document title extracted from the source."""

    published_date: Optional[datetime] = None
    """Publication or last-updated date, if known."""

    language: Optional[str] = None
    """ISO 639-1 language code (e.g. 'en')."""

    page_count: Optional[int] = None
    """Total number of pages (PDFs)."""

    char_count: Optional[int] = None
    """Total character count across all chunks."""

    token_count: Optional[int] = None
    """Total approximate token count across all chunks."""

    # ---- fetch outcome details ----
    error_message: Optional[str] = None
    """Human-readable error description when status is 'failed'."""

    http_status: Optional[int] = None
    """HTTP response status code observed at fetch time."""

    content_type: Optional[str] = None
    """HTTP Content-Type header value as actually observed (not guessed from URL)."""

    # ---- content ----
    chunks: List[DocumentChunk] = field(default_factory=list)
    """Ordered list of content chunks extracted from the document."""

    extracted_tables: List[List[List[str]]] = field(default_factory=list)
    """Document-level tables for table-first sources (each table is row-major)."""

    extracted_dates: List[str] = field(default_factory=list)
    """Date strings found anywhere in the document, preserved as-is."""
