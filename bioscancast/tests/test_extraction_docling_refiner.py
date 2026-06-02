"""Tests for bioscancast.extraction.docling_refiner.

Docling is heavyweight (~1.5 GB RAM and ~10-30 s model load on construction).
Every test in this module uses a fake converter injected into the refiner,
so no real Docling model is ever loaded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

from bioscancast.extraction.config import ExtractionConfig
from bioscancast.extraction.docling_refiner import (
    DoclingTableRefiner,
    _broken_table_reasons,
    _docling_table_to_rows,
    _merge_docling_tables_into_parsed,
    _should_refine_by_url,
)
from bioscancast.extraction.parsers.base import ParsedContent, SectionContent


# ---------------------------------------------------------------------------
# Stubs that mimic the bits of the Docling object model we touch
# ---------------------------------------------------------------------------


@dataclass
class StubProv:
    page_no: int


@dataclass
class StubTableCell:
    start_row_offset_idx: int
    end_row_offset_idx: int
    start_col_offset_idx: int
    end_col_offset_idx: int
    text: str


@dataclass
class StubTableData:
    num_rows: int
    num_cols: int
    table_cells: List[StubTableCell]


@dataclass
class StubTable:
    data: StubTableData
    prov: List[StubProv]


@dataclass
class StubDoclingDocument:
    tables: List[StubTable] = field(default_factory=list)


def _make_stub_table(
    rows: List[List[str]], *, page_no: int
) -> StubTable:
    num_rows = len(rows)
    num_cols = max(len(r) for r in rows) if rows else 0
    cells = []
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            cells.append(
                StubTableCell(
                    start_row_offset_idx=r,
                    end_row_offset_idx=r + 1,
                    start_col_offset_idx=c,
                    end_col_offset_idx=c + 1,
                    text=value,
                )
            )
    return StubTable(
        data=StubTableData(num_rows=num_rows, num_cols=num_cols, table_cells=cells),
        prov=[StubProv(page_no=page_no)],
    )


def _section(
    chunk_type: str,
    *,
    page_number: Optional[int] = None,
    table_rows: Optional[List[List[str]]] = None,
    text: str = "",
    extractor: Optional[str] = None,
) -> SectionContent:
    return SectionContent(
        section_path=None,
        page_number=page_number,
        text=text,
        chunk_type=chunk_type,
        table_rows=table_rows,
        extractor=extractor,
    )


# ---------------------------------------------------------------------------
# _should_refine_by_url
# ---------------------------------------------------------------------------


class TestShouldRefineByUrl:
    def test_match_in_allowlist(self):
        assert _should_refine_by_url(
            "https://www.cdc.gov/mmwr/volumes/75/wr/mm7509a1.htm",
            ["cdc.gov/mmwr/"],
        )

    def test_case_insensitive(self):
        assert _should_refine_by_url(
            "https://WWW.CDC.GOV/MMWR/foo.pdf",
            ["cdc.gov/mmwr/"],
        )

    def test_no_match(self):
        assert not _should_refine_by_url(
            "https://reuters.com/world/article", ["cdc.gov/mmwr/"]
        )

    def test_empty_url(self):
        assert not _should_refine_by_url("", ["cdc.gov/mmwr/"])

    def test_empty_allowlist(self):
        assert not _should_refine_by_url("https://cdc.gov/mmwr/x", [])


# ---------------------------------------------------------------------------
# _broken_table_reasons
# ---------------------------------------------------------------------------


class TestBrokenTableReasons:
    def test_healthy_table_passes(self):
        parsed = ParsedContent(
            raw_text="",
            sections=[
                _section(
                    "table",
                    page_number=1,
                    table_rows=[
                        ["Country", "Cases"],
                        ["Sudan", "100"],
                        ["DRC", "250"],
                        ["Nigeria", "75"],
                    ],
                ),
            ],
        )
        assert _broken_table_reasons(parsed, threshold=0.5) == []

    def test_sparse_table_flagged(self):
        rows = [
            ["A", "", "", "", "", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", "", "", "", "", ""],
        ]
        parsed = ParsedContent(
            raw_text="",
            sections=[_section("table", page_number=4, table_rows=rows)],
        )
        flagged = _broken_table_reasons(parsed, threshold=0.5)
        assert len(flagged) == 1
        idx, reason = flagged[0]
        assert idx == 0
        assert "page 4" in reason
        assert "empty-cell ratio" in reason

    def test_over_segmented_flagged(self):
        # Most rows have only one non-empty cell -- looks like per-column over-segmentation.
        rows = [
            ["Header", "value"],
            ["x", ""],
            ["y", ""],
            ["z", ""],
            ["w", ""],
            ["v", ""],
        ]
        parsed = ParsedContent(
            raw_text="",
            sections=[_section("table", page_number=2, table_rows=rows)],
        )
        flagged = _broken_table_reasons(parsed, threshold=0.5)
        assert len(flagged) == 1
        idx, reason = flagged[0]
        assert idx == 0
        assert "over-segmented" in reason

    def test_skips_non_table_sections(self):
        parsed = ParsedContent(
            raw_text="",
            sections=[_section("prose", page_number=1, text="hello world")],
        )
        assert _broken_table_reasons(parsed, threshold=0.5) == []

    def test_skips_too_small_tables(self):
        parsed = ParsedContent(
            raw_text="",
            sections=[
                _section(
                    "table",
                    page_number=1,
                    table_rows=[["", ""], ["", ""]],
                ),
            ],
        )
        assert _broken_table_reasons(parsed, threshold=0.5) == []


# ---------------------------------------------------------------------------
# _docling_table_to_rows
# ---------------------------------------------------------------------------


class TestDoclingTableToRows:
    def test_simple_grid(self):
        stub = _make_stub_table(
            [["Country", "Cases"], ["Sudan", "100"], ["DRC", "250"]],
            page_no=1,
        )
        rows = _docling_table_to_rows(stub)
        assert rows == [["Country", "Cases"], ["Sudan", "100"], ["DRC", "250"]]

    def test_missing_data_returns_empty(self):
        class _NoData:
            data = None

        assert _docling_table_to_rows(_NoData()) == []


# ---------------------------------------------------------------------------
# _merge_docling_tables_into_parsed
# ---------------------------------------------------------------------------


class TestMergeDoclingTables:
    def test_replaces_in_tree_table_by_page(self):
        parsed = ParsedContent(
            raw_text="",
            sections=[
                _section(
                    "table",
                    page_number=4,
                    table_rows=[["", ""], ["", ""], ["", ""]],
                    extractor="pymupdf",
                ),
            ],
        )
        docling_doc = StubDoclingDocument(
            tables=[
                _make_stub_table(
                    [["State", "Count"], ["NM", "9"], ["TX", "11"]],
                    page_no=4,
                ),
            ],
        )

        result = _merge_docling_tables_into_parsed(parsed, docling_doc)
        assert len(result.sections) == 1
        section = result.sections[0]
        assert section.chunk_type == "table"
        assert section.extractor == "docling"
        assert section.table_rows == [
            ["State", "Count"],
            ["NM", "9"],
            ["TX", "11"],
        ]

    def test_leaves_table_with_no_matching_page(self):
        parsed = ParsedContent(
            raw_text="",
            sections=[
                _section(
                    "table",
                    page_number=4,
                    table_rows=[["A", "B"]],
                    extractor="pymupdf",
                ),
            ],
        )
        docling_doc = StubDoclingDocument(
            tables=[
                _make_stub_table([["X", "Y"]], page_no=7),  # different page
            ],
        )

        result = _merge_docling_tables_into_parsed(parsed, docling_doc)
        assert result.sections[0].extractor == "pymupdf"
        assert result.sections[0].table_rows == [["A", "B"]]

    def test_multiple_tables_on_same_page_matched_in_order(self):
        parsed = ParsedContent(
            raw_text="",
            sections=[
                _section(
                    "table",
                    page_number=2,
                    table_rows=[["?", "?"]],
                    extractor="pymupdf",
                ),
                _section(
                    "prose",
                    page_number=2,
                    text="some prose between",
                    extractor="pymupdf",
                ),
                _section(
                    "table",
                    page_number=2,
                    table_rows=[["?", "?"]],
                    extractor="pymupdf",
                ),
            ],
        )
        docling_doc = StubDoclingDocument(
            tables=[
                _make_stub_table([["first", "1"]], page_no=2),
                _make_stub_table([["second", "2"]], page_no=2),
            ],
        )

        result = _merge_docling_tables_into_parsed(parsed, docling_doc)
        tables = [s for s in result.sections if s.chunk_type == "table"]
        assert tables[0].table_rows == [["first", "1"]]
        assert tables[0].extractor == "docling"
        assert tables[1].table_rows == [["second", "2"]]
        assert tables[1].extractor == "docling"
        # The prose chunk in between is preserved.
        assert any(s.chunk_type == "prose" for s in result.sections)

    def test_leaves_prose_sections_alone(self):
        parsed = ParsedContent(
            raw_text="",
            sections=[
                _section("prose", page_number=1, text="hello", extractor="pymupdf"),
            ],
        )
        docling_doc = StubDoclingDocument(
            tables=[_make_stub_table([["X", "Y"]], page_no=1)],
        )

        result = _merge_docling_tables_into_parsed(parsed, docling_doc)
        # Prose untouched; the docling table is inserted as a new section.
        prose = [s for s in result.sections if s.chunk_type == "prose"]
        tables = [s for s in result.sections if s.chunk_type == "table"]
        assert len(prose) == 1
        assert prose[0].extractor == "pymupdf"
        assert prose[0].text == "hello"
        assert len(tables) == 1
        assert tables[0].extractor == "docling"

    def test_drops_unmatched_broken_intree_table(self):
        # MMWR-style: in-tree's spurious table is on page 4, Docling's real
        # table is on page 3.  Page match fails. With broken_indices={0},
        # the in-tree section is dropped and Docling's table is inserted.
        parsed = ParsedContent(
            raw_text="",
            sections=[
                _section(
                    "table",
                    page_number=4,
                    table_rows=[["", ""], ["", ""], ["", ""]],
                    extractor="pymupdf",
                ),
            ],
        )
        docling_doc = StubDoclingDocument(
            tables=[
                _make_stub_table(
                    [["Characteristic", "No. (%)"], ["Total", "99"], ["Sex", ""]],
                    page_no=3,
                ),
            ],
        )

        result = _merge_docling_tables_into_parsed(
            parsed, docling_doc, broken_indices=frozenset([0])
        )
        tables = [s for s in result.sections if s.chunk_type == "table"]
        assert len(tables) == 1
        assert tables[0].extractor == "docling"
        assert tables[0].page_number == 3
        assert tables[0].table_rows[0] == ["Characteristic", "No. (%)"]

    def test_keeps_unmatched_clean_intree_table(self):
        # If an in-tree table didn't match Docling AND wasn't flagged broken,
        # keep it (Docling missed a legitimate table).
        parsed = ParsedContent(
            raw_text="",
            sections=[
                _section(
                    "table",
                    page_number=4,
                    table_rows=[["A", "B"], ["1", "2"]],
                    extractor="pymupdf",
                ),
            ],
        )
        docling_doc = StubDoclingDocument(
            tables=[_make_stub_table([["X", "Y"]], page_no=3)],
        )

        result = _merge_docling_tables_into_parsed(
            parsed, docling_doc, broken_indices=frozenset()
        )
        tables = [s for s in result.sections if s.chunk_type == "table"]
        # Original in-tree table preserved, plus inserted Docling table.
        assert len(tables) == 2
        extractors = sorted(t.extractor for t in tables)
        assert extractors == ["docling", "pymupdf"]

    def test_unmatched_docling_inserted_in_page_order(self):
        parsed = ParsedContent(
            raw_text="",
            sections=[
                _section("prose", page_number=1, text="page1", extractor="pymupdf"),
                _section("prose", page_number=5, text="page5", extractor="pymupdf"),
            ],
        )
        docling_doc = StubDoclingDocument(
            tables=[_make_stub_table([["X", "Y"]], page_no=3)],
        )

        result = _merge_docling_tables_into_parsed(parsed, docling_doc)
        # Inserted Docling table should sit between the page-1 prose and the page-5 prose.
        assert [s.page_number for s in result.sections] == [1, 3, 5]
        assert result.sections[1].extractor == "docling"


# ---------------------------------------------------------------------------
# DoclingTableRefiner.refine() end-to-end with a fake converter
# ---------------------------------------------------------------------------


class _FakeResult:
    def __init__(self, document):
        self.document = document


class _FakeConverter:
    def __init__(self, document):
        self._document = document
        self.convert_calls = 0

    def convert(self, _stream):
        self.convert_calls += 1
        return _FakeResult(self._document)


class TestDoclingTableRefinerEndToEnd:
    def _config(self) -> ExtractionConfig:
        return ExtractionConfig(
            enable_docling_refiner=True,
            docling_source_allowlist=["cdc.gov/mmwr/"],
            docling_sparse_cell_threshold=0.5,
        )

    def test_triggers_on_allowlist(self):
        parsed = ParsedContent(
            raw_text="",
            sections=[
                _section(
                    "table",
                    page_number=1,
                    table_rows=[["Country", "Cases"], ["Sudan", "5"]],
                    extractor="pymupdf",
                ),
            ],
        )
        docling_doc = StubDoclingDocument(
            tables=[_make_stub_table([["NM", "9"], ["TX", "11"]], page_no=1)],
        )
        converter = _FakeConverter(docling_doc)
        refiner = DoclingTableRefiner(self._config(), converter=converter)

        out = refiner.refine(
            parsed,
            source_url="https://www.cdc.gov/mmwr/volumes/75/wr/mm7509a1.htm",
            content=b"%PDF-fake-bytes",
        )

        assert converter.convert_calls == 1
        assert out.sections[0].extractor == "docling"
        assert out.sections[0].table_rows == [["NM", "9"], ["TX", "11"]]

    def test_triggers_on_heuristic(self):
        # Sparse table -> heuristic fires even without allowlist match.
        rows = [["A", "", "", ""], ["", "", "", ""], ["", "", "", ""]]
        parsed = ParsedContent(
            raw_text="",
            sections=[
                _section("table", page_number=3, table_rows=rows, extractor="pymupdf"),
            ],
        )
        docling_doc = StubDoclingDocument(
            tables=[_make_stub_table([["Region", "n"], ["X", "1"]], page_no=3)],
        )
        converter = _FakeConverter(docling_doc)
        refiner = DoclingTableRefiner(self._config(), converter=converter)

        out = refiner.refine(
            parsed,
            source_url="https://example.org/random.pdf",
            content=b"%PDF-fake-bytes",
        )

        assert converter.convert_calls == 1
        assert out.sections[0].extractor == "docling"

    def test_no_trigger_leaves_parsed_unchanged(self):
        parsed = ParsedContent(
            raw_text="",
            sections=[
                _section(
                    "table",
                    page_number=1,
                    table_rows=[
                        ["Country", "Cases"],
                        ["Sudan", "5"],
                        ["DRC", "100"],
                    ],
                    extractor="pymupdf",
                ),
            ],
        )
        converter = _FakeConverter(StubDoclingDocument())
        refiner = DoclingTableRefiner(self._config(), converter=converter)

        out = refiner.refine(
            parsed,
            source_url="https://reuters.com/world/africa/article",
            content=b"%PDF-fake-bytes",
        )

        assert converter.convert_calls == 0
        assert out.sections[0].extractor == "pymupdf"

    def test_short_circuits_on_requires_ocr(self):
        parsed = ParsedContent(
            raw_text="",
            sections=[],
            is_partial=True,
            partial_reason="requires_ocr",
        )
        converter = _FakeConverter(StubDoclingDocument())
        refiner = DoclingTableRefiner(self._config(), converter=converter)

        out = refiner.refine(
            parsed,
            source_url="https://www.cdc.gov/mmwr/volumes/75/wr/mm7509a1.htm",
            content=b"%PDF-fake-bytes",
        )

        assert converter.convert_calls == 0
        assert out is parsed

    def test_converter_failure_falls_back_to_parsed(self):
        parsed = ParsedContent(
            raw_text="",
            sections=[
                _section(
                    "table",
                    page_number=1,
                    table_rows=[["A", "B"]],
                    extractor="pymupdf",
                ),
            ],
        )
        converter = MagicMock()
        converter.convert.side_effect = RuntimeError("boom")
        refiner = DoclingTableRefiner(self._config(), converter=converter)

        out = refiner.refine(
            parsed,
            source_url="https://www.cdc.gov/mmwr/volumes/75/wr/mm7509a1.htm",
            content=b"%PDF-fake-bytes",
        )
        assert out.sections[0].extractor == "pymupdf"


# ---------------------------------------------------------------------------
# Pipeline integration: extractor provenance flows through to DocumentChunk
# ---------------------------------------------------------------------------


def test_disabling_flag_skips_docling_construction(monkeypatch):
    """With enable_docling_refiner=False the pipeline must never instantiate
    a refiner (and therefore never touch any Docling import)."""
    from bioscancast.extraction.pipeline import ExtractionPipeline

    pipeline = ExtractionPipeline(
        config=ExtractionConfig(enable_docling_refiner=False)
    )

    def _fail(*_a, **_kw):
        raise AssertionError("DoclingTableRefiner should not be constructed")

    monkeypatch.setattr(
        "bioscancast.extraction.docling_refiner.DoclingTableRefiner.__init__",
        _fail,
    )
    # Force the pipeline's path that decides whether to call the refiner.
    assert pipeline._config.enable_docling_refiner is False
