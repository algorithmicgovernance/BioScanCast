"""OCR-on re-conversion of the two PDFs that OCR-off couldn't handle well.

- ECDC CDTR week 16: OCR-off detected 4 tables but all came back 0x0.
- Africa CDC weekly (April 2026): OCR-off yielded 0 chunks — pure image PDF.

Reads the locally-downloaded copies under data/docling_eval/sources/ (from
the earlier run) to avoid re-fetching and any publisher-side drift, and
writes OCR-on outputs to data/docling_eval/ocr/ so the OCR-off outputs at
data/docling_eval/*.md stay intact for comparison.

Run from the repo root:

    .venv-docling/Scripts/python.exe -u scripts/eval_docling_ocr.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

print("Importing docling...", flush=True)
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
print("docling imported OK.", flush=True)


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "data" / "docling_eval" / "sources"
OUT_DIR = REPO_ROOT / "data" / "docling_eval" / "ocr"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Source:
    name: str
    path: Path
    notes: str


SOURCES: list[Source] = [
    Source(
        name="ecdc_cdtr_week16",
        path=SRC_DIR / "ecdc_cdtr_week16.pdf",
        notes="OCR-off: 4 tables detected but 0x0 — recover case-count tables via OCR.",
    ),
    Source(
        name="africa_cdc_weekly_apr2026",
        path=SRC_DIR / "africa_cdc_weekly_apr2026.pdf",
        notes="OCR-off: 0 chunks extracted — entire PDF is scanned images.",
    ),
]


def _safe_chunk_meta(chunk: Any) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    chunk_meta = getattr(chunk, "meta", None)
    if chunk_meta is None:
        return meta
    headings = getattr(chunk_meta, "headings", None)
    if headings:
        meta["headings"] = list(headings)
    pages: set[int] = set()
    for item in getattr(chunk_meta, "doc_items", None) or []:
        for p in getattr(item, "prov", None) or []:
            page_no = getattr(p, "page_no", None)
            if isinstance(page_no, int):
                pages.add(page_no)
    if pages:
        meta["pages"] = sorted(pages)
    return meta


def convert_one(source: Source, converter: DocumentConverter,
                chunker: HybridChunker) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "name": source.name,
        "path": str(source.path.relative_to(REPO_ROOT)),
        "notes": source.notes,
        "status": "pending",
        "elapsed_sec": None,
        "pages": None,
        "tables": None,
        "table_shapes": None,
        "chunks": None,
        "markdown_path": None,
        "doc_json_path": None,
        "chunks_json_path": None,
    }
    print(f"\n=== {source.name} (OCR=on) ===", flush=True)
    print(f"Path: {source.path}", flush=True)
    start = time.monotonic()
    try:
        result = converter.convert(str(source.path))
        elapsed = time.monotonic() - start
        rec["elapsed_sec"] = round(elapsed, 2)
        doc = result.document

        pages = getattr(doc, "pages", None) or {}
        try:
            rec["pages"] = len(pages)
        except TypeError:
            rec["pages"] = 0

        tables = getattr(doc, "tables", None) or []
        rec["tables"] = len(tables)
        shapes: list[str] = []
        for t in tables:
            data = getattr(t, "data", None)
            n_rows = getattr(data, "num_rows", None) if data is not None else None
            n_cols = getattr(data, "num_cols", None) if data is not None else None
            shapes.append(f"{n_rows}x{n_cols}")
        rec["table_shapes"] = shapes

        md_path = OUT_DIR / f"{source.name}.md"
        md_path.write_text(doc.export_to_markdown(), encoding="utf-8")
        rec["markdown_path"] = str(md_path.relative_to(REPO_ROOT))

        doc_json_path = OUT_DIR / f"{source.name}.json"
        doc_json_path.write_text(
            json.dumps(doc.export_to_dict(), indent=2, default=str), encoding="utf-8"
        )
        rec["doc_json_path"] = str(doc_json_path.relative_to(REPO_ROOT))

        chunks_list: list[dict[str, Any]] = []
        for chunk in chunker.chunk(dl_doc=doc):
            contextualized = chunker.contextualize(chunk=chunk)
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
        rec["chunks"] = len(chunks_list)

        chunks_json_path = OUT_DIR / f"{source.name}_chunks.json"
        chunks_json_path.write_text(
            json.dumps(chunks_list, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        rec["chunks_json_path"] = str(chunks_json_path.relative_to(REPO_ROOT))

        rec["status"] = "ok"
        print(
            f"OK  elapsed={rec['elapsed_sec']}s pages={rec['pages']} "
            f"tables={rec['tables']} shapes={shapes} chunks={rec['chunks']}",
            flush=True,
        )
    except Exception as exc:
        rec["elapsed_sec"] = round(time.monotonic() - start, 2)
        rec["status"] = "error"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        print(f"ERROR after {rec['elapsed_sec']}s: {rec['error']}", flush=True)
        traceback.print_exc(limit=2)
    return rec


def main() -> int:
    print(f"Output dir: {OUT_DIR}", flush=True)
    print("Constructing DocumentConverter (OCR=on, RapidOCR, TableFormer FAST)...", flush=True)
    pdf_opts = PdfPipelineOptions()
    pdf_opts.do_ocr = True
    pdf_opts.do_table_structure = True
    pdf_opts.table_structure_options.mode = TableFormerMode.FAST
    # Leave force_full_page_ocr off: Docling's default is to OCR only bitmap
    # regions on otherwise-text pages, which is exactly what we want for
    # ECDC (image-embedded tables on text pages). For the fully-scanned
    # Africa CDC PDF, every page is one big bitmap so it'll get OCR'd anyway.
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
    )
    print("Converter ready.", flush=True)

    print("Loading HuggingFace tokenizer (all-MiniLM-L6-v2)...", flush=True)
    hf_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    tokenizer = HuggingFaceTokenizer(tokenizer=hf_tokenizer, max_tokens=512)
    chunker = HybridChunker(tokenizer=tokenizer)
    print("Chunker ready.", flush=True)

    records = [convert_one(s, converter, chunker) for s in SOURCES]
    (OUT_DIR / "run_log.json").write_text(
        json.dumps({"records": records}, indent=2, default=str), encoding="utf-8"
    )

    print("\n=== SUMMARY (OCR=on) ===", flush=True)
    for r in records:
        extra = (
            f"pages={r['pages']} tables={r['tables']} shapes={r.get('table_shapes')} chunks={r['chunks']}"
            if r["status"] == "ok" else r.get("error")
        )
        print(f"  [{r['status']:>5}] {r['name']:30s} {r['elapsed_sec']}s  {extra}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
