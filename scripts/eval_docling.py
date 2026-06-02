"""Standalone Docling evaluation against real biosecurity sources.

Converts a curated set of WHO/CDC/ECDC/Africa-CDC PDFs plus a few HTML news
articles using Docling, saves Markdown + JSON output per source, runs
HybridChunker (max_tokens=512), and writes a summary log.

Not part of the BioScanCast package; uses its own venv (.venv-docling).
Run from the repo root:

    .venv-docling/Scripts/python.exe scripts/eval_docling.py

Outputs: data/docling_eval/
"""
from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Force unbuffered stdout so the progress log streams in real time.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Route docling's own loggers to stdout so progress downloads are visible.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

print("Importing docling...", flush=True)

# Docling imports
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

print("docling imported OK.", flush=True)


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "docling_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Hard per-source timeout so a hanging conversion never blocks the whole run.
# Docling itself has no convert timeout; we enforce it by tracking elapsed time
# and aborting at the end. (The task requirement is to flag >3 min runs.)
SOFT_TIMEOUT_SEC = 240  # flag anything over this as "slow"


@dataclass
class Source:
    name: str          # file-safe slug used for output files
    category: str      # who_don | cdc_mmwr | ecdc_cdtr | africa_cdc | reuters | cidrap | promed
    url: str
    notes: str = ""    # any caveats/expectations


# Curated list of publicly accessible biosecurity sources (verified URLs).
# NOTE: WHO "Disease Outbreak News" items themselves are HTML-only on who.int,
# so we use WHO outbreak situation-report PDFs (mpox + cholera) which are the
# table-heavy, multi-section PDFs WHO publishes for the same outbreaks.
SOURCES: list[Source] = [
    Source(
        name="who_mpox_sitrep64",
        category="who_don",
        url="https://cdn.who.int/media/docs/default-source/_sage-2026/multi-country-outbreak-of-mpox--external-situation-report_64.pdf?sfvrsn=10400a6e_4&download=true",
        notes="WHO multi-country mpox external situation report #64 (table-heavy).",
    ),
    Source(
        name="who_cholera_epi34",
        category="who_don",
        url="https://cdn.who.int/media/docs/default-source/documents/emergencies/situation-reports/20260221_multi-country_outbreak-of-cholera_epidemiological_update_34.pdf?sfvrsn=c367355_4&download=true",
        notes="WHO multi-country cholera epidemiological update #34 (21 Feb 2026).",
    ),
    Source(
        name="cdc_mmwr_nm_measles",
        category="cdc_mmwr",
        url="https://www.cdc.gov/mmwr/volumes/75/wr/pdfs/mm7509a1-H.pdf",
        notes="MMWR Vol 75 No 9 (Mar 12 2026) — Measles Outbreak New Mexico 2025.",
    ),
    Source(
        name="ecdc_cdtr_week16",
        category="ecdc_cdtr",
        url="https://www.ecdc.europa.eu/sites/default/files/documents/Communicable-disease-threats-report-week-16-2026.pdf",
        notes="ECDC Communicable Disease Threats Report week 16 (12-18 Apr 2026).",
    ),
    Source(
        name="africa_cdc_weekly_apr2026",
        category="africa_cdc",
        url="https://africacdc.org/download/africa-cdc-epidemic-intelligence-weekly-report-april-2026/?wpdmdl=24028",
        notes="Africa CDC Epidemic Intelligence Weekly Report, April 2026.",
    ),
    Source(
        name="reuters_bird_flu",
        category="reuters",
        # Reuters uses Cloudflare bot protection; docling's default httpx fetch
        # typically returns 401/403. We include this source precisely to measure
        # whether Docling can handle a hardened HTML source out of the box.
        url="https://www.reuters.com/business/healthcare-pharmaceuticals/",
        notes="Reuters healthcare section front page (tests bot-protected HTML).",
    ),
    Source(
        name="cidrap_utah_measles",
        category="cidrap",
        url="https://www.cidrap.umn.edu/measles/utah-measles-outbreak-tops-600-cases-now-most-active-us",
        notes="CIDRAP news article — Utah measles outbreak tops 600 cases.",
    ),
    Source(
        name="promed_latest",
        category="promed",
        # ProMED's public homepage lists recent posts; individual post permalinks
        # are behind JS. We feed the list page to exercise HTML handling.
        url="https://promedmail.org/promed-post/",
        notes="ProMED recent-posts listing page (JS-heavy).",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_chunk_meta(chunk: Any) -> dict[str, Any]:
    """Extract heading path + page refs from a chunk.meta, robust to schema."""
    meta: dict[str, Any] = {}
    chunk_meta = getattr(chunk, "meta", None)
    if chunk_meta is None:
        return meta

    # Heading path: docling exposes chunk.meta.headings (list[str]) in hierarchical chunker.
    headings = getattr(chunk_meta, "headings", None)
    if headings:
        meta["headings"] = list(headings)

    # Page references: docling stores doc_items each with prov -> list[ProvenanceItem(page_no,...)]
    pages: set[int] = set()
    doc_items = getattr(chunk_meta, "doc_items", None) or []
    for item in doc_items:
        provs = getattr(item, "prov", None) or []
        for p in provs:
            page_no = getattr(p, "page_no", None)
            if isinstance(page_no, int):
                pages.add(page_no)
    if pages:
        meta["pages"] = sorted(pages)

    # Origin (source filename) if available
    origin = getattr(chunk_meta, "origin", None)
    if origin is not None:
        origin_filename = getattr(origin, "filename", None)
        if origin_filename:
            meta["origin_filename"] = origin_filename

    return meta


def _count_tables(doc: Any) -> int:
    tables = getattr(doc, "tables", None) or []
    try:
        return len(tables)
    except TypeError:
        return 0


def _count_pages(doc: Any) -> int:
    pages = getattr(doc, "pages", None)
    if pages is None:
        return 0
    try:
        return len(pages)
    except TypeError:
        return 0


def _extract_pub_date(doc: Any) -> str | None:
    """Best-effort: docling rarely exposes publication metadata for PDFs.
    We look at doc.origin (filename/mimetype/binary_hash) and any top-level meta.
    """
    origin = getattr(doc, "origin", None)
    if origin is not None:
        for attr in ("publication_date", "date", "created"):
            val = getattr(origin, attr, None)
            if val:
                return str(val)
    # Some converters attach metadata via doc.meta or doc.properties — try both.
    meta = getattr(doc, "meta", None)
    if isinstance(meta, dict):
        for key in ("publication_date", "date", "created", "creationDate"):
            if key in meta and meta[key]:
                return str(meta[key])
    return None


def convert_one(source: Source, converter: DocumentConverter,
                chunker: HybridChunker) -> dict[str, Any]:
    """Convert a single source, save outputs, return a metrics record."""
    record: dict[str, Any] = {
        "name": source.name,
        "category": source.category,
        "url": source.url,
        "notes": source.notes,
        "status": "pending",
        "elapsed_sec": None,
        "pages": None,
        "tables": None,
        "chunks": None,
        "pub_date": None,
        "error": None,
        "markdown_path": None,
        "doc_json_path": None,
        "chunks_json_path": None,
        "slow": False,
    }

    print(f"\n=== {source.name} ({source.category}) ===", flush=True)
    print(f"URL: {source.url}", flush=True)
    start = time.monotonic()
    try:
        result = converter.convert(source.url)
        elapsed = time.monotonic() - start
        record["elapsed_sec"] = round(elapsed, 2)
        record["slow"] = elapsed > SOFT_TIMEOUT_SEC

        doc = result.document

        # Counts
        record["pages"] = _count_pages(doc)
        record["tables"] = _count_tables(doc)
        record["pub_date"] = _extract_pub_date(doc)

        # Save Markdown
        md_path = OUT_DIR / f"{source.name}.md"
        md_path.write_text(doc.export_to_markdown(), encoding="utf-8")
        record["markdown_path"] = str(md_path.relative_to(REPO_ROOT))

        # Save full document JSON
        doc_json_path = OUT_DIR / f"{source.name}.json"
        doc_json_path.write_text(
            json.dumps(doc.export_to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        record["doc_json_path"] = str(doc_json_path.relative_to(REPO_ROOT))

        # Chunk
        chunks_list: list[dict[str, Any]] = []
        for chunk in chunker.chunk(dl_doc=doc):
            contextualized = chunker.contextualize(chunk=chunk)
            # Token count (using chunker's tokenizer)
            try:
                token_count = chunker.tokenizer.count_tokens(text=contextualized)
            except Exception:
                token_count = None
            entry: dict[str, Any] = {
                "text": chunk.text,
                "contextualized_text": contextualized,
                "token_count": token_count,
            }
            entry.update(_safe_chunk_meta(chunk))
            chunks_list.append(entry)

        record["chunks"] = len(chunks_list)
        chunks_json_path = OUT_DIR / f"{source.name}_chunks.json"
        chunks_json_path.write_text(
            json.dumps(chunks_list, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        record["chunks_json_path"] = str(chunks_json_path.relative_to(REPO_ROOT))

        record["status"] = "ok"
        print(
            f"OK  elapsed={record['elapsed_sec']}s pages={record['pages']} "
            f"tables={record['tables']} chunks={record['chunks']} "
            f"pub_date={record['pub_date']}",
            flush=True,
        )
        if record["slow"]:
            print(f"WARNING: conversion took >{SOFT_TIMEOUT_SEC}s (marked slow).", flush=True)
    except Exception as exc:
        elapsed = time.monotonic() - start
        record["elapsed_sec"] = round(elapsed, 2)
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        print(f"ERROR after {record['elapsed_sec']}s: {record['error']}", flush=True)
        traceback.print_exc(limit=2)

    return record


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    started_at = datetime.now(timezone.utc).isoformat()
    print(f"Docling eval — started {started_at}", flush=True)
    print(f"Output dir: {OUT_DIR}", flush=True)

    # Single converter reused across all sources.
    # Disable OCR: every source in SOURCES is a born-digital PDF or HTML page,
    # so OCR just burns 5-10 minutes per PDF on CPU without improving extraction.
    # Use FAST TableFormer mode — the accurate model is roughly 3x slower.
    print("Constructing DocumentConverter (first run downloads layout models)...", flush=True)
    pdf_opts = PdfPipelineOptions()
    pdf_opts.do_ocr = False
    pdf_opts.do_table_structure = True
    pdf_opts.table_structure_options.mode = TableFormerMode.FAST
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
    )
    print("Converter ready.", flush=True)

    # HybridChunker with max_tokens=512 on the all-MiniLM-L6-v2 tokenizer
    # (the docling default, matches typical embedding contexts).
    print("Loading HuggingFace tokenizer (all-MiniLM-L6-v2)...", flush=True)
    hf_tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    tokenizer = HuggingFaceTokenizer(tokenizer=hf_tokenizer, max_tokens=512)
    chunker = HybridChunker(tokenizer=tokenizer)
    print("Chunker ready.", flush=True)

    records: list[dict[str, Any]] = []
    for source in SOURCES:
        rec = convert_one(source, converter, chunker)
        records.append(rec)

    # Summary
    finished_at = datetime.now(timezone.utc).isoformat()
    summary = {
        "started_at": started_at,
        "finished_at": finished_at,
        "total_sources": len(records),
        "ok": sum(1 for r in records if r["status"] == "ok"),
        "errors": sum(1 for r in records if r["status"] == "error"),
        "slow": sum(1 for r in records if r.get("slow")),
        "records": records,
    }
    (OUT_DIR / "run_log.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print("\n=== SUMMARY ===")
    print(f"ok={summary['ok']} errors={summary['errors']} slow={summary['slow']}")
    for r in records:
        status = r["status"]
        extra = (
            f"pages={r['pages']} tables={r['tables']} chunks={r['chunks']}"
            if status == "ok"
            else r["error"]
        )
        print(f"  [{status:>5}] {r['name']:35s} {r['elapsed_sec']}s  {extra}")

    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
