from __future__ import annotations

import io
import logging
import statistics
from datetime import datetime
from typing import List, Optional

from .base import ParsedContent, SectionContent

logger = logging.getLogger(__name__)

# TODO: Detect scanned PDFs (no extractable text on any page) and return
#       a failed ParsedContent with reason "requires_ocr".

_DEFAULT_MAX_PAGES = 100


class PdfParser:
    """Extracts structured content from PDF documents using PyMuPDF."""

    def __init__(self, *, max_pages: int = _DEFAULT_MAX_PAGES) -> None:
        self._max_pages = max_pages

    def can_parse(self, content_type: str, content: bytes) -> bool:
        if "pdf" in (content_type or ""):
            return True
        return content[:5] == b"%PDF-"

    def parse(self, content: bytes, *, source_url: str) -> ParsedContent:
        import pymupdf

        doc = pymupdf.open(stream=content, filetype="pdf")
        total_pages = len(doc)
        is_partial = total_pages > self._max_pages
        pages_to_process = min(total_pages, self._max_pages)

        # Extract metadata
        meta = doc.metadata or {}
        title = meta.get("title") or None
        pub_date = self._parse_pdf_date(meta.get("creationDate"))

        all_text_parts: List[str] = []
        sections: List[SectionContent] = []
        heading_stack: List[str] = []

        # First pass: collect font sizes across all pages for heading heuristic
        font_sizes: List[float] = []
        for page_num in range(pages_to_process):
            page = doc[page_num]
            blocks = page.get_text("dict", flags=0)["blocks"]
            for block in blocks:
                if block.get("type") != 0:  # text blocks only
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        size = span.get("size", 0)
                        if size > 0:
                            font_sizes.append(size)

        median_size = statistics.median(font_sizes) if font_sizes else 12.0

        # Second pass: extract content
        for page_num in range(pages_to_process):
            page = doc[page_num]
            page_number = page_num + 1  # 1-based

            # Extract tables with PyMuPDF
            tables_on_page = self._extract_tables_pymupdf(page)

            # If PyMuPDF found no tables, try pdfplumber as fallback
            if not tables_on_page and self._page_looks_tabular(page):
                tables_on_page = self._extract_tables_pdfplumber(
                    content, page_num
                )

            for table_rows in tables_on_page:
                sections.append(
                    SectionContent(
                        section_path=" > ".join(heading_stack) if heading_stack else None,
                        page_number=page_number,
                        text="",
                        chunk_type="table",
                        table_rows=table_rows,
                    )
                )

            # Extract text blocks
            blocks = page.get_text("dict", flags=0)["blocks"]
            current_text_parts: List[str] = []

            for block in blocks:
                if block.get("type") != 0:
                    continue

                block_text = ""
                is_heading = False

                for line in block.get("lines", []):
                    line_text_parts = []
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        line_text_parts.append(text)
                        size = span.get("size", 0)
                        flags = span.get("flags", 0)
                        is_bold = bool(flags & (1 << 4))
                        if size > median_size * 1.15 or (is_bold and size >= median_size):
                            is_heading = True

                    if line_text_parts:
                        block_text += " ".join(line_text_parts) + "\n"

                block_text = block_text.strip()
                if not block_text:
                    continue

                all_text_parts.append(block_text)

                if is_heading and len(block_text) < 200:
                    # Flush accumulated text
                    if current_text_parts:
                        combined = "\n".join(current_text_parts).strip()
                        if combined:
                            sections.append(
                                SectionContent(
                                    section_path=" > ".join(heading_stack) if heading_stack else None,
                                    page_number=page_number,
                                    text=combined,
                                    chunk_type="prose",
                                )
                            )
                        current_text_parts = []

                    heading_stack = [block_text]
                else:
                    current_text_parts.append(block_text)

            # Flush remaining text for this page
            if current_text_parts:
                combined = "\n".join(current_text_parts).strip()
                if combined:
                    sections.append(
                        SectionContent(
                            section_path=" > ".join(heading_stack) if heading_stack else None,
                            page_number=page_number,
                            text=combined,
                            chunk_type="prose",
                        )
                    )

        doc.close()

        raw_text = "\n".join(all_text_parts)

        # Detect scanned PDFs
        if not raw_text.strip() and total_pages > 0:
            return ParsedContent(
                raw_text="",
                sections=[],
                title=title,
                page_count=total_pages,
                published_date=pub_date,
                is_partial=True,
                partial_reason="requires_ocr",
            )

        return ParsedContent(
            raw_text=raw_text,
            sections=sections,
            title=title,
            page_count=total_pages,
            published_date=pub_date,
            is_partial=is_partial,
            partial_reason=f"Truncated to {self._max_pages} of {total_pages} pages" if is_partial else None,
        )

    def _extract_tables_pymupdf(self, page) -> List[List[List[str]]]:
        """Extract tables from a PyMuPDF page using find_tables()."""
        try:
            tables = page.find_tables()
            result = []
            for table in tables:
                rows = []
                for row in table.extract():
                    cells = [str(cell) if cell is not None else "" for cell in row]
                    rows.append(cells)
                if rows:
                    result.append(rows)
            return result
        except Exception:
            return []

    def _page_looks_tabular(self, page) -> bool:
        """Heuristic: lots of short lines with whitespace alignment suggests tables."""
        text = page.get_text("text")
        if not text:
            return False
        lines = text.strip().split("\n")
        if len(lines) < 3:
            return False
        short_lines = sum(1 for line in lines if len(line.strip()) < 60)
        return short_lines > len(lines) * 0.5

    def _extract_tables_pdfplumber(
        self, content: bytes, page_index: int
    ) -> List[List[List[str]]]:
        """Fallback table extraction using pdfplumber for a single page."""
        try:
            import pdfplumber

            with pdfplumber.open(io.BytesIO(content)) as pdf:
                if page_index >= len(pdf.pages):
                    return []
                page = pdf.pages[page_index]
                tables = page.extract_tables() or []
                result = []
                for table in tables:
                    rows = []
                    for row in table:
                        cells = [str(cell) if cell is not None else "" for cell in row]
                        rows.append(cells)
                    if rows:
                        result.append(rows)
                return result
        except Exception as exc:
            logger.debug("pdfplumber fallback failed on page %d: %s", page_index, exc)
            return []

    def _parse_pdf_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse PDF date strings like 'D:20240115120000+00'00''."""
        if not date_str:
            return None
        # Strip the D: prefix
        date_str = date_str.strip()
        if date_str.startswith("D:"):
            date_str = date_str[2:]
        # Take just the YYYYMMDD part for robustness
        try:
            return datetime.strptime(date_str[:8], "%Y%m%d")
        except (ValueError, IndexError):
            return None
