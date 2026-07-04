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
import time
from dataclasses import dataclass, field, replace
from typing import Any, FrozenSet, List, Optional, Sequence, Tuple

from .config import ExtractionConfig
from .parsers.base import ParsedContent, SectionContent

logger = logging.getLogger(__name__)


@dataclass
class RefinerTelemetry:
    """One record per ``refine()`` call (i.e. per PDF that reaches the refiner).

    Captures the inputs issue #17 needs to decide per-page vs full-doc Docling:
    per-doc wall-clock, page count, and the suspect-table footprint. ``fired``
    distinguishes calls where Docling actually ran from skips (no trigger /
    OCR-required). Conversion is the page-scaling cost — ``convert_s`` is the
    metric ``page_range`` would reduce; ``total_s`` adds the (cheap) merge.
    """

    source_url: str
    fired: bool
    trigger: Optional[str]  # "allowlist" | "heuristic" | None
    status: str
    page_count: Optional[int]
    n_suspect_tables: int
    suspect_pages: List[int] = field(default_factory=list)
    convert_s: Optional[float] = None
    total_s: float = 0.0

    def as_dict(self) -> dict:
        return {
            "source_url": self.source_url,
            "fired": self.fired,
            "trigger": self.trigger,
            "status": self.status,
            "page_count": self.page_count,
            "n_suspect_tables": self.n_suspect_tables,
            "suspect_pages": list(self.suspect_pages),
            "convert_s": (
                round(self.convert_s, 3) if self.convert_s is not None else None
            ),
            "total_s": round(self.total_s, 3),
        }


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
        # One RefinerTelemetry per refine() call, in call order. The
        # extraction pipeline drains this after a run so the orchestrator can
        # persist it (issue #17: 2-week refiner-time collection).
        self.telemetry: List[RefinerTelemetry] = []

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
        t0 = time.perf_counter()

        if parsed.is_partial and parsed.partial_reason == "requires_ocr":
            logger.debug("docling refiner skipped: requires_ocr")
            self._record(
                source_url, fired=False, trigger=None, status="skipped_ocr",
                parsed=parsed, flagged=[], total_s=time.perf_counter() - t0,
            )
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
            trigger = "allowlist"
            logger.info(
                "docling refiner triggered: source-allowlist hit for %s", source_url
            )
        elif flagged:
            trigger = "heuristic"
            for _, reason in flagged:
                logger.info("docling refiner triggered: %s", reason)
        else:
            logger.debug(
                "docling refiner skipped: no trigger matched for %s", source_url
            )
            self._record(
                source_url, fired=False, trigger=None, status="skipped_no_trigger",
                parsed=parsed, flagged=flagged, total_s=time.perf_counter() - t0,
            )
            return parsed

        refined, convert_s, status = self._do_refine(
            parsed, content, broken_indices=broken_indices
        )
        self._record(
            source_url, fired=(status == "refined"), trigger=trigger, status=status,
            parsed=parsed, flagged=flagged, convert_s=convert_s,
            total_s=time.perf_counter() - t0,
        )
        return refined

    def _record(
        self,
        source_url: str,
        *,
        fired: bool,
        trigger: Optional[str],
        status: str,
        parsed: ParsedContent,
        flagged: Sequence[Tuple[int, str]],
        total_s: float,
        convert_s: Optional[float] = None,
    ) -> None:
        suspect_pages = sorted(
            {
                parsed.sections[i].page_number
                for i, _ in flagged
                if 0 <= i < len(parsed.sections)
                and parsed.sections[i].page_number is not None
            }
        )
        self.telemetry.append(
            RefinerTelemetry(
                source_url=source_url,
                fired=fired,
                trigger=trigger,
                status=status,
                page_count=parsed.page_count,
                n_suspect_tables=len(flagged),
                suspect_pages=suspect_pages,
                convert_s=convert_s,
                total_s=total_s,
            )
        )

    # ---------- internals ----------

    def _do_refine(
        self,
        parsed: ParsedContent,
        content: bytes,
        *,
        broken_indices: FrozenSet[int] = frozenset(),
    ) -> Tuple[ParsedContent, Optional[float], str]:
        """Run Docling and merge its tables back in.

        Returns ``(parsed, convert_s, status)`` where ``convert_s`` is the
        wall-clock of the Docling ``convert()`` call (None if it never ran)
        and ``status`` is one of ``refined``, ``converter_unavailable``,
        ``conversion_failed``, ``no_document``.
        """
        try:
            converter = self._get_converter()
        except Exception as exc:  # pragma: no cover - construction failures
            logger.warning("docling converter unavailable: %s", exc)
            return parsed, None, "converter_unavailable"

        t0 = time.perf_counter()
        try:
            result = converter.convert(content)
        except Exception as exc:
            logger.warning("docling conversion failed: %s", exc)
            return parsed, time.perf_counter() - t0, "conversion_failed"
        convert_s = time.perf_counter() - t0

        docling_doc = getattr(result, "document", None)
        if docling_doc is None:
            logger.warning("docling result has no document; leaving parsed unchanged")
            return parsed, convert_s, "no_document"

        merged = _merge_docling_tables_into_parsed(
            parsed, docling_doc, broken_indices=broken_indices
        )
        return merged, convert_s, "refined"

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
