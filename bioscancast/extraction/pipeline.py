from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import List, Optional

from bioscancast.filtering.models import FilteredDocument
from bioscancast.schemas.document import Document, DocumentChunk

from .chunking import normalize_chunks
from .config import ExtractionConfig
from .fetcher import FetchResult, fetch
from .parsers import get_parsers
from .parsers.base import ParsedContent
from .tokens import approx_token_count

logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """Orchestrates document fetching, parsing, and chunk normalization."""

    def __init__(self, *, config: ExtractionConfig | None = None) -> None:
        self._config = config or ExtractionConfig()
        self._parsers = get_parsers(pdf_max_pages=self._config.pdf_max_pages)

    def run(self, filtered_docs: List[FilteredDocument]) -> List[Document]:
        """Process documents in order of extraction_priority.

        A failure on one document never affects others.
        """
        sorted_docs = sorted(filtered_docs, key=lambda d: d.extraction_priority)
        results: List[Document] = []

        for fdoc in sorted_docs:
            try:
                doc = self.extract_one(fdoc)
            except Exception as exc:
                logger.error(
                    "Unexpected error extracting %s: %s", fdoc.url, exc, exc_info=True
                )
                doc = self._make_failed_document(
                    fdoc, error=f"unexpected_error: {exc}"
                )
            results.append(doc)

        return results

    def extract_one(self, filtered_doc: FilteredDocument) -> Document:
        """Fetch, parse, chunk, and return a Document for a single FilteredDocument."""
        doc_id = f"doc-{filtered_doc.result_id}"

        # Step 1: Fetch
        fetch_result = fetch(filtered_doc.url, config=self._config)

        if fetch_result.error or fetch_result.content_bytes is None:
            return self._make_failed_document(
                filtered_doc,
                error=fetch_result.error or "empty_response",
                fetch_result=fetch_result,
            )

        # Step 2: Select parser
        content_type = fetch_result.content_type or ""
        parser = None
        for p in self._parsers:
            if p.can_parse(content_type, fetch_result.content_bytes):
                parser = p
                break

        if parser is None:
            return self._make_failed_document(
                filtered_doc,
                error="no_parser_available",
                fetch_result=fetch_result,
            )

        # Step 3: Parse
        try:
            parsed = parser.parse(
                fetch_result.content_bytes, source_url=filtered_doc.url
            )
        except Exception as exc:
            logger.warning("Parser failed for %s: %s", filtered_doc.url, exc)
            return self._make_failed_document(
                filtered_doc,
                error=f"parse_error: {exc}",
                fetch_result=fetch_result,
            )

        # Check for OCR-required PDFs
        if parsed.is_partial and parsed.partial_reason == "requires_ocr":
            return self._make_failed_document(
                filtered_doc,
                error="requires_ocr",
                fetch_result=fetch_result,
            )

        # Step 4: Convert ParsedContent → Document with chunks
        document_type = self._detect_document_type(content_type)
        chunks = self._build_chunks(parsed, doc_id)

        # Step 5: Normalize chunks
        chunks = normalize_chunks(
            chunks,
            target_tokens=self._config.chunk_target_tokens,
            max_tokens=self._config.chunk_max_tokens,
        )

        # Renumber chunk indices after normalization
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i

        # Step 6: Compute document-level stats
        char_count = sum(len(c.text) for c in chunks)
        token_count = sum(c.token_count or 0 for c in chunks)

        # Collect all tables for document-level extracted_tables
        extracted_tables = [
            c.table_data for c in chunks if c.chunk_type == "table" and c.table_data
        ]

        # Extract dates from text
        extracted_dates = self._extract_dates(parsed.raw_text)

        status = "partial" if parsed.is_partial else "success"

        return Document(
            id=doc_id,
            result_id=filtered_doc.result_id,
            source_url=filtered_doc.url,
            domain=filtered_doc.domain,
            fetched_at=fetch_result.fetched_at,
            document_type=document_type,
            status=status,
            canonical_url=filtered_doc.canonical_url,
            title=parsed.title or filtered_doc.title,
            published_date=parsed.published_date or filtered_doc.published_date,
            language=parsed.language,
            page_count=parsed.page_count,
            char_count=char_count,
            token_count=token_count,
            error_message=parsed.partial_reason,
            http_status=fetch_result.status_code,
            content_type=fetch_result.content_type,
            chunks=chunks,
            extracted_tables=extracted_tables,
            extracted_dates=extracted_dates,
        )

    def _make_failed_document(
        self,
        fdoc: FilteredDocument,
        *,
        error: str,
        fetch_result: FetchResult | None = None,
    ) -> Document:
        return Document(
            id=f"doc-{fdoc.result_id}",
            result_id=fdoc.result_id,
            source_url=fdoc.url,
            domain=fdoc.domain,
            fetched_at=(
                fetch_result.fetched_at
                if fetch_result
                else datetime.now(timezone.utc)
            ),
            document_type=self._detect_document_type(
                fetch_result.content_type if fetch_result else None
            ),
            status="failed",
            error_message=error,
            http_status=fetch_result.status_code if fetch_result else None,
            content_type=fetch_result.content_type if fetch_result else None,
        )

    def _build_chunks(
        self, parsed: ParsedContent, doc_id: str
    ) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        for i, section in enumerate(parsed.sections):
            chunk_id = f"{doc_id}-c{i}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    chunk_index=i,
                    text=section.text,
                    chunk_type=section.chunk_type,
                    heading=section.section_path,
                    page_number=section.page_number,
                    table_data=section.table_rows,
                    token_count=approx_token_count(section.text),
                )
            )
        return chunks

    def _detect_document_type(self, content_type: Optional[str]) -> str:
        if not content_type:
            return "html"
        if "pdf" in content_type:
            return "pdf"
        if "html" in content_type:
            return "html"
        if "json" in content_type:
            return "api_json"
        return "html"

    def _extract_dates(self, text: str) -> List[str]:
        """Best-effort date extraction from raw text."""
        if not text:
            return []
        # Match common date formats
        patterns = [
            r"\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}",
            r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}",
            r"\d{4}-\d{2}-\d{2}",
            r"\d{1,2}/\d{1,2}/\d{4}",
        ]
        dates: List[str] = []
        for pattern in patterns:
            dates.extend(re.findall(pattern, text, re.IGNORECASE))
        # Deduplicate preserving order
        seen: set = set()
        unique: List[str] = []
        for d in dates:
            if d not in seen:
                seen.add(d)
                unique.append(d)
        return unique
