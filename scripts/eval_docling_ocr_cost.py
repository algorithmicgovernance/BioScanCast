"""Per-page OCR cost measurement on the ECDC CDTR PDF.

Earlier run: OCR-on on the full 12-page ECDC CDTR didn't finish in 42 min
before we killed it, which tells us the wall-clock is prohibitive but not
the per-page rate. This script measures cost with `page_range`:

- OCR-off baseline on pages 1, 5, 10 (3 samples across the doc)
- OCR-on (bitmap-only, default) on the same three pages
- OCR-on (forced full-page) on one page, for upper bound

Writes each per-page Markdown to data/docling_eval/ocr/per_page_<mode>_p<N>.md
and logs timings to stdout. Reads from the local source PDF downloaded earlier.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s")

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PDF = REPO_ROOT / "data" / "docling_eval" / "sources" / "ecdc_cdtr_week16.pdf"
OUT_DIR = REPO_ROOT / "data" / "docling_eval" / "ocr"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _make_converter(do_ocr: bool, force_full_page: bool = False) -> DocumentConverter:
    opts = PdfPipelineOptions()
    opts.do_ocr = do_ocr
    opts.do_table_structure = True
    opts.table_structure_options.mode = TableFormerMode.FAST
    if do_ocr and force_full_page:
        opts.ocr_options.force_full_page_ocr = True
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def _time_single_page(converter: DocumentConverter, page: int, label: str) -> dict:
    start = time.monotonic()
    result = converter.convert(str(SRC_PDF), page_range=(page, page))
    elapsed = time.monotonic() - start
    doc = result.document
    tables = getattr(doc, "tables", []) or []
    shapes = [
        f"{getattr(getattr(t,'data',None),'num_rows',None)}x{getattr(getattr(t,'data',None),'num_cols',None)}"
        for t in tables
    ]
    md = doc.export_to_markdown()
    out_path = OUT_DIR / f"per_page_{label}_p{page:02d}.md"
    out_path.write_text(md, encoding="utf-8")
    n_tokens_approx = len(md.split())
    return {
        "label": label,
        "page": page,
        "elapsed_sec": round(elapsed, 2),
        "tables": len(tables),
        "shapes": shapes,
        "md_bytes": len(md),
        "md_words_approx": n_tokens_approx,
        "md_path": str(out_path.relative_to(REPO_ROOT)),
    }


def main() -> int:
    print(f"Source: {SRC_PDF}", flush=True)
    print(f"Output: {OUT_DIR}", flush=True)

    pages_to_test = [1, 5, 10]

    results: list[dict] = []

    print("\n--- OCR OFF (baseline) ---", flush=True)
    conv_off = _make_converter(do_ocr=False)
    for p in pages_to_test:
        r = _time_single_page(conv_off, p, "ocroff")
        print(f"  page {p:>2}: elapsed={r['elapsed_sec']:>6.2f}s tables={r['tables']} shapes={r['shapes']} md_bytes={r['md_bytes']}", flush=True)
        results.append(r)

    print("\n--- OCR ON (bitmap-only, default) ---", flush=True)
    conv_on = _make_converter(do_ocr=True, force_full_page=False)
    for p in pages_to_test:
        r = _time_single_page(conv_on, p, "ocron")
        print(f"  page {p:>2}: elapsed={r['elapsed_sec']:>6.2f}s tables={r['tables']} shapes={r['shapes']} md_bytes={r['md_bytes']}", flush=True)
        results.append(r)

    print("\n--- OCR ON (force_full_page), page 5 only ---", flush=True)
    conv_full = _make_converter(do_ocr=True, force_full_page=True)
    r = _time_single_page(conv_full, 5, "ocrfull")
    print(f"  page {5:>2}: elapsed={r['elapsed_sec']:>6.2f}s tables={r['tables']} shapes={r['shapes']} md_bytes={r['md_bytes']}", flush=True)
    results.append(r)

    (OUT_DIR / "per_page_cost.json").write_text(
        json.dumps({"results": results}, indent=2, default=str), encoding="utf-8"
    )

    # Summarise
    off_times = [r["elapsed_sec"] for r in results if r["label"] == "ocroff"]
    on_times = [r["elapsed_sec"] for r in results if r["label"] == "ocron"]
    full_times = [r["elapsed_sec"] for r in results if r["label"] == "ocrfull"]
    print("\n=== SUMMARY ===", flush=True)
    print(f"OCR OFF mean/page: {sum(off_times)/len(off_times):>6.1f}s  (pages={pages_to_test})", flush=True)
    print(f"OCR ON  mean/page: {sum(on_times)/len(on_times):>6.1f}s  (pages={pages_to_test})", flush=True)
    print(f"OCR ON marginal cost/page: {(sum(on_times)/len(on_times)) - (sum(off_times)/len(off_times)):>6.1f}s", flush=True)
    print(f"OCR FULL-PAGE page 5: {full_times[0]:>6.1f}s (vs bitmap-only {[r['elapsed_sec'] for r in results if r['label']=='ocron' and r['page']==5][0]:.1f}s)", flush=True)
    total_12p_on = (sum(on_times)/len(on_times)) * 12
    print(f"Projected OCR-on total for 12-page ECDC CDTR: ~{total_12p_on/60:.1f} min", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
