"""Head-to-head: run the in-tree PdfParser against the same 5 PDFs Docling
already converted, so we can eyeball Markdown output and compare metrics.

Reads from data/docling_eval/sources/*.pdf, writes:
- data/docling_eval/intree_pdf/{name}.md   — Markdown re-emitted from ParsedContent
- data/docling_eval/intree_pdf/{name}.json — full ParsedContent (sections + metadata)
- data/docling_eval/intree_pdf/run_log.json — aggregate metrics

Run from repo root:

    .venv-docling/Scripts/python.exe -u scripts/eval_intree_pdf.py
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # for `import bioscancast.*`

from bioscancast.stages.extraction.parsers.pdf_parser import PdfParser  # noqa: E402

SRC_DIR = REPO_ROOT / "data" / "docling_eval" / "sources"
OUT_DIR = REPO_ROOT / "data" / "docling_eval" / "intree_pdf"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = [
    "who_mpox_sitrep64",
    "who_cholera_epi34",
    "cdc_mmwr_nm_measles",
    "ecdc_cdtr_week16",
    "africa_cdc_weekly_apr2026",
]


def _table_to_md(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    # Normalise widths
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
    """Render ParsedContent as Markdown so it's directly comparable to Docling's."""
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
        # Emit a heading marker when the section_path changes
        if path != last_path and path:
            depth = path.count(" > ") + 2  # h2 minimum
            depth = min(depth, 6)
            lines.append(f"\n{'#' * depth} {path.split(' > ')[-1]}\n")
            last_path = path
        if s.chunk_type == "table" and s.table_rows:
            if s.page_number:
                lines.append(f"\n*Table on page {s.page_number}:*\n")
            lines.append(_table_to_md(s.table_rows) + "\n")
        elif s.text:
            lines.append(s.text + "\n")
    return "\n".join(lines)


def _section_summary(parsed) -> dict[str, Any]:
    table_sections = [s for s in parsed.sections if s.chunk_type == "table"]
    prose_sections = [s for s in parsed.sections if s.chunk_type == "prose"]
    table_cells = sum(
        len(s.table_rows or []) * (len((s.table_rows or [[]])[0]) if s.table_rows else 0)
        for s in table_sections
    )
    table_shapes = [
        f"{len(s.table_rows or [])}x{(len((s.table_rows or [[]])[0]) if s.table_rows else 0)}"
        for s in table_sections
    ]
    return {
        "n_sections": len(parsed.sections),
        "n_prose": len(prose_sections),
        "n_tables": len(table_sections),
        "table_shapes": table_shapes,
        "table_cells_total": table_cells,
        "raw_text_chars": len(parsed.raw_text),
        "is_partial": parsed.is_partial,
        "partial_reason": parsed.partial_reason,
    }


def main() -> int:
    parser = PdfParser()
    results: list[dict[str, Any]] = []

    for name in SOURCES:
        pdf_path = SRC_DIR / f"{name}.pdf"
        if not pdf_path.exists():
            print(f"SKIP  {name}: file not found", flush=True)
            continue

        print(f"\n=== {name} ===", flush=True)
        content = pdf_path.read_bytes()
        start = time.monotonic()
        try:
            parsed = parser.parse(content, source_url=str(pdf_path))
            elapsed = time.monotonic() - start
        except Exception as exc:
            elapsed = time.monotonic() - start
            print(f"ERROR after {elapsed:.2f}s: {type(exc).__name__}: {exc}", flush=True)
            results.append({
                "name": name,
                "status": "error",
                "elapsed_sec": round(elapsed, 2),
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue

        # Save MD + JSON
        md_path = OUT_DIR / f"{name}.md"
        md_path.write_text(_emit_markdown(parsed), encoding="utf-8")

        json_dump = {
            "title": parsed.title,
            "published_date": parsed.published_date.isoformat() if parsed.published_date else None,
            "page_count": parsed.page_count,
            "is_partial": parsed.is_partial,
            "partial_reason": parsed.partial_reason,
            "raw_text_chars": len(parsed.raw_text),
            "sections": [
                {
                    "section_path": s.section_path,
                    "page_number": s.page_number,
                    "chunk_type": s.chunk_type,
                    "text": s.text,
                    "table_rows": s.table_rows,
                }
                for s in parsed.sections
            ],
        }
        (OUT_DIR / f"{name}.json").write_text(
            json.dumps(json_dump, indent=2, default=str), encoding="utf-8"
        )

        summary = _section_summary(parsed)
        rec = {
            "name": name,
            "status": "ok",
            "elapsed_sec": round(elapsed, 2),
            "title": parsed.title,
            "published_date": parsed.published_date.isoformat() if parsed.published_date else None,
            "page_count": parsed.page_count,
            **summary,
        }
        results.append(rec)
        print(
            f"OK  elapsed={rec['elapsed_sec']}s pages={rec['page_count']} "
            f"sections={rec['n_sections']} prose={rec['n_prose']} tables={rec['n_tables']} "
            f"shapes={rec['table_shapes']} chars={rec['raw_text_chars']} "
            f"pub_date={rec['published_date']}",
            flush=True,
        )
        if rec.get("partial_reason"):
            print(f"  partial: {rec['partial_reason']}", flush=True)

    (OUT_DIR / "run_log.json").write_text(
        json.dumps({"results": results}, indent=2, default=str), encoding="utf-8"
    )

    print("\n=== SUMMARY (in-tree PdfParser) ===", flush=True)
    for r in results:
        if r["status"] == "ok":
            print(
                f"  [   ok] {r['name']:30s} {r['elapsed_sec']:>6.2f}s  "
                f"pages={r['page_count']} sections={r['n_sections']} "
                f"tables={r['n_tables']} {r['table_shapes']} chars={r['raw_text_chars']} "
                f"pub={r['published_date']}",
                flush=True,
            )
        else:
            print(f"  [error] {r['name']:30s} {r['elapsed_sec']:>6.2f}s  {r.get('error')}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
