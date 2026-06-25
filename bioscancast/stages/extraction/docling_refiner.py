"""Docling-based table refiner.

Optional post-pass over `ParsedContent` produced by `PdfParser`. When triggered
(URL allowlist hit or a heuristic flag on a "broken" in-tree table), runs
Docling's TableFormer on the original PDF bytes and replaces the in-tree table
sections with Docling's rendering.

Docling and its transitive deps (`transformers`, `torch`, ...) are intentionally
*lazy-imported* — instantiating `DoclingTableRefiner` is the only path that
touches them. When the feature flag is off, no Docling import ever happens.
"""

from __future__ import annotations

import io
import logging
from dataclasses import replace
from typing import Any, FrozenSet, List, Optional, Sequence, Tuple

from .config import ExtractionConfig
from .parsers.base import ParsedContent, SectionContent

logger = logging.getLogger(__name__)


class DoclingTableRefiner:
    """Refines table sections in a `ParsedContent` using Docling.

    The converter is constructed once per instance (Docling models cost
    ~10-30s and ~1.5 GB RAM to load), so the pipeline should hold one
    instance per process.
    """

    def __init__(
        self,
        config: ExtractionConfig,
        *,
        converter: Optional[Any] = None,
    ) -> None:
        self._config = config
        # Allow dependency injection for tests; real construction is lazy.
        self._converter = converter

    # ---------- public API ----------

    def refine(
        self,
        parsed: ParsedContent,
        *,
        source_url: str,
        content: bytes,
    ) -> ParsedContent:
        """Return either the original `parsed` or a copy with table sections
        replaced by Docling output.

        Triggers (first match wins):
          1. `source_url` matches the configured allowlist.
          2. Any in-tree table looks "broken" by the heuristic.

        Short-circuits to a no-op for OCR-required PDFs — Docling without OCR
        cannot help there.
        """
        if parsed.is_partial and parsed.partial_reason == "requires_ocr":
            logger.debug("docling refiner skipped: requires_ocr")
            return parsed

        # Always compute broken-table indices: even URL-triggered runs need
        # them, so the merge step knows which in-tree sections to drop when
        # Docling produces a different but better table on another page.
        flagged = _broken_table_reasons(
            parsed, threshold=self._config.docling_sparse_cell_threshold
        )
        broken_indices = frozenset(i for i, _ in flagged)

        url_match = _should_refine_by_url(
            source_url, self._config.docling_source_allowlist
        )
        if url_match:
            logger.info(
                "docling refiner triggered: source-allowlist hit for %s", source_url
            )
        elif flagged:
            for _, reason in flagged:
                logger.info("docling refiner triggered: %s", reason)
        else:
            logger.debug(
                "docling refiner skipped: no trigger matched for %s", source_url
            )
            return parsed

        return self._do_refine(parsed, content, broken_indices=broken_indices)

    # ---------- internals ----------

    def _do_refine(
        self,
        parsed: ParsedContent,
        content: bytes,
        *,
        broken_indices: FrozenSet[int] = frozenset(),
    ) -> ParsedContent:
        try:
            converter = self._get_converter()
        except Exception as exc:  # pragma: no cover - construction failures
            logger.warning("docling converter unavailable: %s", exc)
            return parsed

        try:
            result = converter.convert(content)
        except Exception as exc:
            logger.warning("docling conversion failed: %s", exc)
            return parsed

        docling_doc = getattr(result, "document", None)
        if docling_doc is None:
            logger.warning("docling result has no document; leaving parsed unchanged")
            return parsed

        return _merge_docling_tables_into_parsed(
            parsed, docling_doc, broken_indices=broken_indices
        )

    def _get_converter(self) -> Any:
        if self._converter is not None:
            return self._converter
        self._converter = _build_converter()
        return self._converter


# ---------- helpers ----------


def _should_refine_by_url(source_url: str, allowlist: Sequence[str]) -> bool:
    if not source_url:
        return False
    lowered = source_url.lower()
    return any(pattern.lower() in lowered for pattern in allowlist)


def _broken_table_reasons(
    parsed: ParsedContent, *, threshold: float
) -> List[Tuple[int, str]]:
    """Inspect every table section in `parsed` and return `(section_index,
    reason)` pairs for any that look broken.

    A table is suspect when:
      - non-empty-cell ratio < `threshold` and it has at least 3 rows and 2 cols
      - more than half its rows have exactly one non-empty cell (over-segmentation)
    """
    flagged: List[Tuple[int, str]] = []
    for i, section in enumerate(parsed.sections):
        if section.chunk_type != "table" or not section.table_rows:
            continue
        rows = section.table_rows
        if len(rows) < 3:
            continue
        max_cols = max((len(r) for r in rows), default=0)
        if max_cols < 2:
            continue

        total_cells = sum(len(r) for r in rows)
        if total_cells == 0:
            continue
        non_empty = sum(
            1 for row in rows for cell in row if cell and str(cell).strip()
        )
        ratio = non_empty / total_cells

        page_label = section.page_number if section.page_number is not None else "?"

        if ratio < threshold:
            flagged.append(
                (
                    i,
                    f"suspect table on page {page_label} "
                    f"(empty-cell ratio {ratio:.2f})",
                )
            )
            continue

        single_cell_rows = sum(
            1
            for row in rows
            if sum(1 for cell in row if cell and str(cell).strip()) == 1
        )
        if single_cell_rows > len(rows) / 2:
            flagged.append(
                (
                    i,
                    f"suspect table on page {page_label} "
                    f"(over-segmented: {single_cell_rows}/{len(rows)} rows have a single cell)",
                )
            )
    return flagged


def _merge_docling_tables_into_parsed(
    parsed: ParsedContent,
    docling_doc: Any,
    *,
    broken_indices: FrozenSet[int] = frozenset(),
) -> ParsedContent:
    """Replace in-tree table sections with Docling-rendered tables.

    Strategy:
      1. Page-based matching: for each in-tree table section, find a Docling
         table on the same page (in order of appearance) and replace.
      2. Drop unmatched in-tree table sections whose index is in
         `broken_indices` (the heuristic flagged them as suspect).
      3. Insert any leftover Docling tables as new sections, at the
         document-order position corresponding to their page.

    `broken_indices` refers to indices into `parsed.sections` as it was
    when the heuristic ran (i.e. the original section list).
    """
    docling_tables_by_page: dict[int, list] = {}
    for table in getattr(docling_doc, "tables", []) or []:
        prov = getattr(table, "prov", None) or []
        if not prov:
            continue
        page_no = getattr(prov[0], "page_no", None)
        if page_no is None:
            continue
        docling_tables_by_page.setdefault(page_no, []).append(table)

    cursor: dict[int, int] = {}
    matched_docling: set[int] = set()
    new_sections: List[SectionContent] = []

    for i, section in enumerate(parsed.sections):
        if section.chunk_type != "table" or section.page_number is None:
            new_sections.append(section)
            continue

        page = section.page_number
        idx = cursor.get(page, 0)
        candidates = docling_tables_by_page.get(page, [])
        if idx < len(candidates):
            docling_table = candidates[idx]
            cursor[page] = idx + 1
            new_rows = _docling_table_to_rows(docling_table)
            if new_rows:
                matched_docling.add(id(docling_table))
                new_sections.append(
                    replace(
                        section,
                        table_rows=new_rows,
                        extractor="docling",
                    )
                )
                continue
            # Fall through: Docling table empty -> treat as no match.

        # No Docling replacement for this in-tree table.
        if i in broken_indices:
            # The heuristic confirmed this in-tree table is garbage;
            # drop it rather than leave noise in the output.
            continue
        new_sections.append(section)

    # Insert leftover Docling tables in page order.
    leftover: List[Tuple[int, Any]] = []
    for page_no, tables in docling_tables_by_page.items():
        for table in tables:
            if id(table) not in matched_docling:
                leftover.append((page_no, table))
    leftover.sort(key=lambda pair: pair[0])

    for page_no, table in leftover:
        rows = _docling_table_to_rows(table)
        if not rows:
            continue
        insert_at = 0
        for j, existing in enumerate(new_sections):
            if (
                existing.page_number is not None
                and existing.page_number <= page_no
            ):
                insert_at = j + 1
        new_sections.insert(
            insert_at,
            SectionContent(
                section_path=None,
                page_number=page_no,
                text="",
                chunk_type="table",
                table_rows=rows,
                extractor="docling",
            ),
        )

    parsed.sections = new_sections
    return parsed


def _docling_table_to_rows(table: Any) -> List[List[str]]:
    """Convert a Docling `TableItem` into row-major plain-string cells.

    Walks `table.data.table_cells` directly so we don't pull in pandas.
    Each cell carries `start_row_offset_idx`/`start_col_offset_idx`; we lay
    them out on a grid of size `num_rows x num_cols` and stringify the text.
    """
    data = getattr(table, "data", None)
    if data is None:
        return []
    cells = list(getattr(data, "table_cells", []) or [])
    if not cells:
        return []

    num_rows = int(getattr(data, "num_rows", 0) or 0)
    num_cols = int(getattr(data, "num_cols", 0) or 0)
    if num_rows <= 0 or num_cols <= 0:
        # Fall back to inferring shape from cell offsets.
        num_rows = max(int(getattr(c, "end_row_offset_idx", 0) or 0) for c in cells)
        num_cols = max(int(getattr(c, "end_col_offset_idx", 0) or 0) for c in cells)
        if num_rows <= 0 or num_cols <= 0:
            return []

    grid: List[List[str]] = [["" for _ in range(num_cols)] for _ in range(num_rows)]
    for cell in cells:
        r = int(getattr(cell, "start_row_offset_idx", 0) or 0)
        c = int(getattr(cell, "start_col_offset_idx", 0) or 0)
        text = (getattr(cell, "text", "") or "").strip()
        if 0 <= r < num_rows and 0 <= c < num_cols:
            grid[r][c] = text
    return grid


def _build_converter() -> Any:
    """Construct a thin wrapper around the real Docling `DocumentConverter`
    that takes raw PDF bytes.

    Imports are deferred to this function so that turning off the refiner
    means no Docling/torch/transformers import ever happens. The wrapper
    layer lets the refiner stay agnostic of Docling-specific stream types,
    which keeps test injection simple.
    """
    from docling.datamodel.base_models import DocumentStream, InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableFormerMode,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=True,
    )
    pipeline_options.table_structure_options.mode = TableFormerMode.FAST

    real_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        },
    )

    class _BytesConverter:
        def convert(self, content: bytes):
            stream = DocumentStream(
                name="document.pdf", stream=io.BytesIO(content)
            )
            return real_converter.convert(stream)

    return _BytesConverter()
