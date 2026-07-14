from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import List, Optional

from bioscancast.stages.filtering.models import FilteredDocument
from bioscancast.schemas.document import Document, DocumentChunk

from .chunking import normalize_chunks
from .config import ExtractionConfig
from .fetcher import FetchResult, fetch
from .parsers import get_parsers
from .parsers.base import ParsedContent
from .tokens import approx_token_count

logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """Orchestrates document fetching, parsing, and chunk normalization.

    ``as_of_date`` opts the fetcher into Wayback-rewrite mode. See
    ``bioscancast.stages.extraction.fetcher.fetch`` for the strategy semantics
    (live / wayback / wayback_fallback_to_live). The resulting strategy
    and snapshot timestamp are copied onto each Document for audit.
    """

    def __init__(
        self,
        *,
        config: ExtractionConfig | None = None,
        as_of_date: Optional[datetime] = None,
    ) -> None:
        self._config = config or ExtractionConfig()
        self._as_of_date = as_of_date
        self._parsers = get_parsers(pdf_max_pages=self._config.pdf_max_pages)
        # Lazily constructed on first PDF that reaches the refiner step.
        self._docling_refiner = None

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

    @property
    def docling_telemetry(self) -> list:
        """Per-PDF Docling refiner telemetry accumulated during ``run``.

        Empty when the refiner never ran (no PDFs, or feature flag off).
        See ``docling_refiner.RefinerTelemetry`` (issue #17).
        """
        refiner = self._docling_refiner
        if refiner is None:
            return []
        return list(getattr(refiner, "telemetry", []))

    def extract_one(self, filtered_doc: FilteredDocument) -> Document:
        """Fetch, parse, chunk, and return a Document for a single FilteredDocument."""
        doc_id = f"doc-{filtered_doc.result_id}"

        # Step 1: Fetch
        fetch_result = fetch(
            filtered_doc.url,
            config=self._config,
            as_of_date=self._as_of_date,
            source_id=filtered_doc.source_id,
            region=filtered_doc.region,
            question_text=filtered_doc.question_text,
        )

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

        # Step 3b: Docling table refiner (PDFs only, feature-flagged)
        document_type = self._detect_document_type(content_type)
        if (
            self._config.enable_docling_refiner
            and document_type == "pdf"
        ):
            refiner = self._get_docling_refiner()
            if refiner is not None:
                try:
                    # Match the Docling allowlist against the URL actually
                    # fetched. Custom hub scrapers (who_cholera, who_h5_hai, ...)
                    # resolve a landing page to a ``cdn.who.int/.../*.pdf``; the
                    # allowlist entries target those resolved PDF paths, so
                    # matching on ``filtered_doc.url`` (the hub) would never fire
                    # the ``situation-reports`` / ``_sage-`` allowlist for exactly
                    # the PDFs it exists to catch.
                    parsed = refiner.refine(
                        parsed,
                        source_url=fetch_result.final_url or filtered_doc.url,
                        content=fetch_result.content_bytes,
                    )
                except Exception as exc:
                    logger.warning(
                        "Docling refiner failed for %s: %s",
                        filtered_doc.url,
                        exc,
                    )

        # Step 4: Convert ParsedContent → Document with chunks
        chunks = self._build_chunks(parsed, doc_id)

        # Step 5: Normalize chunks
        chunks = normalize_chunks(
            chunks,
            target_tokens=self._config.chunk_target_tokens,
            max_tokens=self._config.chunk_max_tokens,
        )

        # Step 5b: Drop or repair empty chunks before any downstream
        # consumer sees them. Empty chunks make retrieval waste budget
        # (BM25 still indexes the heading) and confuse the insight stage.
        chunks = _drop_or_repair_empty_chunks(chunks)

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
            fetch_strategy=fetch_result.fetch_strategy,
            snapshot_timestamp=fetch_result.snapshot_timestamp,
            cutoff_applied=self._as_of_date,
        )

    def _get_docling_refiner(self):
        """Lazily build (and cache) the Docling refiner.

        Returns None if the heavy Docling imports or model load fail — the
        pipeline then falls back to the in-tree parser output unchanged.
        """
        if self._docling_refiner is not None:
            return self._docling_refiner
        try:
            from .docling_refiner import DoclingTableRefiner

            self._docling_refiner = DoclingTableRefiner(self._config)
        except Exception as exc:
            logger.warning("Docling refiner unavailable, continuing without: %s", exc)
            self._docling_refiner = None
        return self._docling_refiner

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
            fetch_strategy=fetch_result.fetch_strategy if fetch_result else "live",
            snapshot_timestamp=fetch_result.snapshot_timestamp if fetch_result else None,
            cutoff_applied=self._as_of_date,
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
                    extractor=section.extractor,
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


def _render_table_data_as_text(rows: List[List[str]]) -> str:
    """Render row-major table data as a flat text block.

    Cells are joined with tabs within a row and rows with newlines.
    Empty cells are preserved (so column alignment stays implicit) but
    fully-empty rows are skipped. This is good enough for BM25 keyword
    matching when the underlying PDF parser produced a table whose cells
    are present but whose flowed text was empty.
    """
    out_rows: List[str] = []
    for row in rows:
        cells = [(cell or "").strip() for cell in row]
        if not any(cells):
            continue
        out_rows.append("\t".join(cells))
    return "\n".join(out_rows)


def _drop_or_repair_empty_chunks(
    chunks: List[DocumentChunk],
) -> List[DocumentChunk]:
    """Filter chunks whose ``text`` is blank after stripping whitespace.

    Two paths, in order of preference:

    * If the chunk is a table with non-empty ``table_data`` rows, render
      those rows into the ``text`` field so downstream retrieval and
      LLM extraction can see the cell contents. The structured
      ``table_data`` is preserved unchanged for any consumer that wants
      cell-level access.
    * Otherwise drop the chunk and log at DEBUG. An empty prose chunk
      usually indicates a half-broken section in the upstream parser
      (heading without body, footer artefact, decorative caption), not
      something the insight stage can act on.

    Running this *after* ``normalize_chunks`` means it sees the final
    post-split chunk text, not pre-split fragments that the splitter
    might have rebalanced.
    """
    out: List[DocumentChunk] = []
    for chunk in chunks:
        if chunk.text and chunk.text.strip():
            out.append(chunk)
            continue
        if chunk.chunk_type == "table" and chunk.table_data:
            rendered = _render_table_data_as_text(chunk.table_data)
            if rendered:
                chunk.text = rendered
                chunk.token_count = approx_token_count(rendered)
                logger.debug(
                    "Empty-text table chunk repaired from table_data "
                    "(chunk_id=%s, rows=%d, rendered_chars=%d)",
                    chunk.chunk_id,
                    len(chunk.table_data),
                    len(rendered),
                )
                out.append(chunk)
                continue
        logger.debug(
            "Dropping empty chunk (chunk_id=%s, type=%s, heading=%r)",
            chunk.chunk_id,
            chunk.chunk_type,
            (chunk.heading or "")[:60],
        )
    return out
