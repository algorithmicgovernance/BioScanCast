"""Hybrid eval: run the in-tree PdfParser + DoclingTableRefiner combo
against the same 5 PDFs Docling/in-tree have already been benchmarked on.

This exercises the new code path from issue #16: in-tree parses, refiner
inspects and (conditionally) replaces table sections with Docling's
rendering when the source URL is on the allowlist OR the in-tree tables
look broken.

Reads from   data/docling_eval/sources/*.pdf
Writes:
  - data/docling_eval/hybrid_pdf/{name}.md        Markdown re-emitted from refined ParsedContent
  - data/docling_eval/hybrid_pdf/{name}.json      Full refined ParsedContent
  - data/docling_eval/hybrid_pdf/run_log.json     Per-source metrics + trigger info

Run from repo root (uses the docling venv since it imports docling):

    .venv-docling/Scripts/python.exe -u scripts/eval_hybrid_pdf.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bioscancast.stages.extraction.config import ExtractionConfig  # noqa: E402
from bioscancast.stages.extraction.docling_refiner import (  # noqa: E402
    DoclingTableRefiner,
    _broken_table_reasons,
    _should_refine_by_url,
)
from bioscancast.stages.extraction.parsers.pdf_parser import PdfParser  # noqa: E402

SRC_DIR = REPO_ROOT / "data" / "docling_eval" / "sources"
OUT_DIR = REPO_ROOT / "data" / "docling_eval" / "hybrid_pdf"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (source basename, plausible publisher URL).  URLs are constructed so that
# allowlist patterns fire exactly where the issue says they should:
#   - MMWR  -> matches `cdc.gov/mmwr/`
#   - WHO cholera sitrep -> matches the situation-reports path
#   - WHO mpox sitrep (this particular one) -> does NOT match
#   - ECDC CDTR -> not on allowlist
#   - Africa CDC weekly -> not on allowlist (will short-circuit on requires_ocr anyway)
SOURCES: list[tuple[str, str]] = [
    (
        "who_mpox_sitrep64",
        "https://cdn.who.int/media/docs/default-source/documents/emergencies/outbreak-reports/2025-mpox-external-sitrep-64.pdf",
    ),
    (
        "who_cholera_epi34",
        "https://cdn.who.int/media/docs/default-source/documents/emergencies/situation-reports/who-cholera-epi-update-34.pdf",
    ),
    (
        "cdc_mmwr_nm_measles",
        "https://www.cdc.gov/mmwr/volumes/75/wr/mm7509a1.htm",
    ),
    (
        "ecdc_cdtr_week16",
        "https://www.ecdc.europa.eu/sites/default/files/documents/communicable-disease-threats-report-week-16-2025.pdf",
    ),
    (
        "africa_cdc_weekly_apr2026",
        "https://africacdc.org/download/weekly-event-based-surveillance-report-april-2026/",
    ),
]


# ---------- markdown rendering (mirrors eval_intree_pdf.py) ----------


def _table_to_md(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    n_cols = max(len(r) for r in rows)
    norm = [r + [""] * (n_cols - len(r)) for r in rows]
    header = norm[0]
    body = norm[1:]
    lines = ["| " + " | ".join(c.replace("\n", " ").strip() for c in header) + " |"]
    lines.append("| " + " | ".join(["---"] * n_cols) + " |")
    for row in body:
        lines.append("| " + " | ".join(c.replace("\n", " ").strip() for c in row) + " |")
    return "\n".join(lines)


def _emit_markdown(parsed) -> str:
    lines: list[str] = []
    if parsed.title:
        lines.append(f"# {parsed.title}\n")
    if parsed.published_date:
        lines.append(f"*Published: {parsed.published_date.date()}*\n")
    if parsed.page_count:
        lines.append(f"*Pages: {parsed.page_count}*\n")

    last_path: str | None = None
    for s in parsed.sections:
        path = s.section_path or ""
        if path != last_path and path:
            depth = path.count(" > ") + 2
            depth = min(depth, 6)
            lines.append(f"\n{'#' * depth} {path.split(' > ')[-1]}\n")
            last_path = path
        if s.chunk_type == "table" and s.table_rows:
            if s.page_number:
                lines.append(
                    f"\n*Table on page {s.page_number} "
                    f"(extractor: {s.extractor or 'unknown'}):*\n"
                )
            lines.append(_table_to_md(s.table_rows) + "\n")
        elif s.text:
            lines.append(s.text + "\n")
    return "\n".join(lines)


def _section_summary(parsed) -> dict[str, Any]:
    table_sections = [s for s in parsed.sections if s.chunk_type == "table"]
    prose_sections = [s for s in parsed.sections if s.chunk_type == "prose"]
    table_cells = sum(
        len(s.table_rows or [])
        * (len((s.table_rows or [[]])[0]) if s.table_rows else 0)
        for s in table_sections
    )
    table_shapes = [
        f"{len(s.table_rows or [])}x{(len((s.table_rows or [[]])[0]) if s.table_rows else 0)}"
        for s in table_sections
    ]
    extractors = [s.extractor for s in table_sections]
    docling_tables = sum(1 for e in extractors if e == "docling")
    return {
        "n_sections": len(parsed.sections),
        "n_prose": len(prose_sections),
        "n_tables": len(table_sections),
        "table_shapes": table_shapes,
        "table_cells_total": table_cells,
        "table_extractors": extractors,
        "n_tables_docling": docling_tables,
        "raw_text_chars": len(parsed.raw_text),
        "is_partial": parsed.is_partial,
        "partial_reason": parsed.partial_reason,
    }


# ---------- main ----------


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = PdfParser()
    config = ExtractionConfig()  # default: refiner enabled, default allowlist
    refiner = DoclingTableRefiner(config)  # converter lazily built on first trigger

    results: list[dict[str, Any]] = []

    for name, source_url in SOURCES:
        pdf_path = SRC_DIR / f"{name}.pdf"
        if not pdf_path.exists():
            print(f"SKIP  {name}: file not found", flush=True)
            continue

        print(f"\n=== {name} ===", flush=True)
        print(f"  source_url: {source_url}", flush=True)
        content = pdf_path.read_bytes()

        # ---- parse with in-tree ----
        start = time.monotonic()
        try:
            parsed = parser.parse(content, source_url=source_url)
        except Exception as exc:
            elapsed = time.monotonic() - start
            print(
                f"PARSE ERROR after {elapsed:.2f}s: {type(exc).__name__}: {exc}",
                flush=True,
            )
            results.append(
                {
                    "name": name,
                    "source_url": source_url,
                    "status": "parse_error",
                    "elapsed_sec": round(elapsed, 2),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        intree_elapsed = time.monotonic() - start

        # ---- predict triggers (so we record *why* the refiner runs or doesn't) ----
        would_trigger_url = _should_refine_by_url(
            source_url, config.docling_source_allowlist
        )
        broken_reasons = _broken_table_reasons(
            parsed, threshold=config.docling_sparse_cell_threshold
        )
        ocr_short_circuit = (
            parsed.is_partial and parsed.partial_reason == "requires_ocr"
        )

        # Count tables before refinement (for diff after).
        in_tree_table_shapes = [
            f"{len(s.table_rows or [])}x{(len((s.table_rows or [[]])[0]) if s.table_rows else 0)}"
            for s in parsed.sections
            if s.chunk_type == "table"
        ]

        # ---- run refiner ----
        refine_start = time.monotonic()
        try:
            refined = refiner.refine(parsed, source_url=source_url, content=content)
        except Exception as exc:
            refine_elapsed = time.monotonic() - refine_start
            print(
                f"REFINE ERROR after {refine_elapsed:.2f}s: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            results.append(
                {
                    "name": name,
                    "source_url": source_url,
                    "status": "refine_error",
                    "intree_elapsed_sec": round(intree_elapsed, 2),
                    "refine_elapsed_sec": round(refine_elapsed, 2),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        refine_elapsed = time.monotonic() - refine_start

        # ---- write artefacts ----
        md_path = OUT_DIR / f"{name}.md"
        md_path.write_text(_emit_markdown(refined), encoding="utf-8")

        json_dump = {
            "source_url": source_url,
            "title": refined.title,
            "published_date": (
                refined.published_date.isoformat() if refined.published_date else None
            ),
            "page_count": refined.page_count,
            "is_partial": refined.is_partial,
            "partial_reason": refined.partial_reason,
            "raw_text_chars": len(refined.raw_text),
            "sections": [
                {
                    "section_path": s.section_path,
                    "page_number": s.page_number,
                    "chunk_type": s.chunk_type,
                    "text": s.text,
                    "table_rows": s.table_rows,
                    "extractor": s.extractor,
                }
                for s in refined.sections
            ],
        }
        (OUT_DIR / f"{name}.json").write_text(
            json.dumps(json_dump, indent=2, default=str), encoding="utf-8"
        )

        summary = _section_summary(refined)
        rec = {
            "name": name,
            "source_url": source_url,
            "status": "ok",
            "intree_elapsed_sec": round(intree_elapsed, 2),
            "refine_elapsed_sec": round(refine_elapsed, 2),
            "trigger": {
                "url_match": would_trigger_url,
                "broken_reasons": broken_reasons,
                "ocr_short_circuit": ocr_short_circuit,
            },
            "title": refined.title,
            "published_date": (
                refined.published_date.isoformat() if refined.published_date else None
            ),
            "page_count": refined.page_count,
            "in_tree_table_shapes": in_tree_table_shapes,
            **summary,
        }
        results.append(rec)
        print(
            f"OK  intree={rec['intree_elapsed_sec']}s "
            f"refine={rec['refine_elapsed_sec']}s "
            f"pages={rec['page_count']} sections={rec['n_sections']} "
            f"tables={rec['n_tables']} "
            f"docling_tables={rec['n_tables_docling']} "
            f"shapes={rec['table_shapes']} "
            f"chars={rec['raw_text_chars']}",
            flush=True,
        )
        print(
            f"  trigger: url_match={would_trigger_url} "
            f"broken={len(broken_reasons)} "
            f"ocr_short_circuit={ocr_short_circuit}",
            flush=True,
        )
        if broken_reasons:
            for r in broken_reasons:
                print(f"    - {r}", flush=True)
        if rec.get("partial_reason"):
            print(f"  partial: {rec['partial_reason']}", flush=True)

    (OUT_DIR / "run_log.json").write_text(
        json.dumps({"results": results}, indent=2, default=str), encoding="utf-8"
    )

    print("\n=== SUMMARY (hybrid: in-tree + Docling refiner) ===", flush=True)
    for r in results:
        if r["status"] == "ok":
            trig = r["trigger"]
            trigger_label = (
                "URL"
                if trig["url_match"]
                else "HEURISTIC"
                if trig["broken_reasons"]
                else "OCR-SKIP"
                if trig["ocr_short_circuit"]
                else "NONE"
            )
            print(
                f"  [   ok] {r['name']:30s} "
                f"intree={r['intree_elapsed_sec']:>6.2f}s "
                f"refine={r['refine_elapsed_sec']:>6.2f}s  "
                f"trigger={trigger_label:9s} "
                f"tables={r['n_tables']} ({r['n_tables_docling']} docling) "
                f"shapes={r['table_shapes']}",
                flush=True,
            )
        else:
            print(
                f"  [{r['status']:>6s}] {r['name']:30s}  {r.get('error')}",
                flush=True,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
