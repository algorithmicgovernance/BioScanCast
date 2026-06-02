# Docling Evaluation — Biosecurity Sources

Ran `scripts/eval_docling.py` against 8 real biosecurity sources (5 PDFs + 3 HTML). Full per-source metrics are in [`run_log.json`](run_log.json); this file summarises what the outputs look like and where Docling struggles for our use case (ingesting WHO/CDC/ECDC/Africa-CDC outbreak documents).

Environment: Docling 2.90.0 + docling-core 2.74.0 in a fresh `.venv-docling` (Python 3.13, Windows, CPU-only). OCR disabled (`do_ocr=False`) and `TableFormerMode.FAST` — with OCR on, the first source alone took >11 minutes and still hadn't finished, so the reported timings are the "fast-path" numbers.

## Summary

| Source | Category | Pages | Tables | Chunks | Elapsed | Status |
| --- | --- | ---:| ---:| ---:| ---:| --- |
| WHO Mpox Sitrep #64 | PDF (WHO sitrep) | 15 | 1 | 38 | **274.6 s** (slow) | ok |
| WHO Cholera Epi Update #34 | PDF (WHO sitrep) | 8 | 1 | 17 | 197.7 s | ok |
| CDC MMWR — NM Measles (mm7509a1) | PDF (MMWR) | 5 | 1 | 20 | 110.5 s | ok |
| ECDC CDTR Week 16 | PDF (ECDC) | 12 | 4 (all empty) | 28 | **324.6 s** (slow) | ok |
| Africa CDC Weekly (April 2026) | PDF (Africa CDC) | 15 | 2 (all empty) | **0** | **523.4 s** (slow) | ok† |
| Reuters — healthcare/pharma landing | HTML | — | — | — | 13.3 s | **error** (401) |
| CIDRAP — Utah measles | HTML | 0 | 0 | 16 | 13.8 s | ok |
| ProMED recent-posts listing | HTML | 0 | 1 | 17 | 13.8 s | ok |

† Africa CDC returned 0 chunks — the PDF is fully image-based and yielded no extractable text with OCR off.

7/8 succeeded, 3/8 breached the 240 s "slow" threshold, and 1 hard failure (Reuters, bot-protected). Total wall-clock for the 8 sources was ~25 minutes; first-run model download added ~40 MB and ~60 s on top.

## What's in the Markdown

### Tables — row/column structure and readable case counts

| Source | Tables in doc | `num_rows × num_cols` | Readable from MD? |
| --- | ---:| --- | --- |
| WHO mpox sitrep | 1 | 9×4 | **Yes** — country / cases / deaths / reporting-countries readable: e.g. "Madagascar \| 368 \| 1 \| -". |
| WHO cholera update | 1 | 21×8 | **Yes** — full cholera-by-region table (country, cases, deaths, CFR, cases-per-100k, monthly % change). "Democratic Republic of the Congo \| 6 543 \| 148 \| 2.3 \| 5 \| 39 \| 66". |
| CDC MMWR | 1 | 17×2 | **Yes** — demographic/characteristic table rendered. |
| ECDC CDTR | 4 detected | **all 0×0** | **No** — TableFormer flagged the table regions but returned empty cells. Case numbers that sit inside the tables are missing from the markdown; in the body text, inline counts ("Italy (63), Spain (36), France (16)") do come through. |
| Africa CDC | 2 detected | **all 0×0** | **No** — image-only PDF, see below. |
| CIDRAP | 0 | — | n/a (article doesn't have tables). |
| ProMED listing | 1 | 157×2 | **Yes** — table of recent post titles by date renders cleanly. |

Takeaway: Docling produces clean Markdown tables **when the PDF has native text tables** (the three WHO/CDC reports do). It silently degrades to empty cells when the tables are embedded as images or rely on OCR, and ECDC's CDTR layout falls into that bucket. For BioScanCast, this means case-count tables from WHO/CDC/MMWR are usable as-is, but ECDC/Africa-CDC tables will need either OCR-on fallback or an external data pipe.

### Section headings

Heading counts after conversion: mpox 40, cholera 15, MMWR 21, ECDC 30, CIDRAP 15, ProMED 7, Africa CDC 0. Order is preserved in all text-extractable sources:

- WHO mpox: "## Highlights" → "## Epidemiological update" → "## Global monkeypox virus (MPXV) distribution" → "## Update on mpox outbreak transmission dynamics by virus clade" → "## Clade Ia MPXV" → "## Clade Ib MPXV" → … (matches the PDF's hierarchy).
- CDC MMWR: "## Abstract" → "## Introduction" → "## Investigation and Outcomes" → "## Notification of Confirmed Measles Cases in Texas" → "## Characteristics of Outbreak-Related Measles Cases" → "## Public Health Response" → "## Discussion" → "## Limitations" → "## Implications for Public Health Practice".
- ECDC CDTR: "## This week's topics" → "## Executive summary" → per-disease sections in order.

Chunk `meta.headings` is populated, so the chunk exposes the full heading path (e.g. `['Measles Outbreak - New Mexico, 2025']`, `['Highlights']`). One caveat: the very first chunk of each doc has `headings=None` because it precedes the first `##` marker.

### Reading order on multi-column PDFs

MMWR is classic 2-column journal layout. The output reads correctly: paragraphs in column 1 flow into column 2 without interleaving, footnote markers (`*`, `†`, `§`) stay attached, and footnote bodies are placed near their markers. One quirk: the "INSIDE" sidebar (which lives in column 2 of page 1) gets spliced between body paragraphs rather than being lifted out — annoying for reading but not a correctness issue.

### HTML: nav / ads / footers stripping

Docling's HTML pipeline does **not** strip boilerplate.

- CIDRAP article: lines 1–47 are the site nav (Topics & Projects, Podcasts, About, Search, …), line 48 is the actual article H1, the article body runs ~lines 48–77, and the remaining ~260 lines are other articles, "Choose newsletters" CTAs, and footer. The first chunk's heading is `['Main navigation']`, and chunk 15's headings include `['Tetanus still occurs among all ages in US, mainly in undervaccinated', 'Choose newsletters']` — so **unwanted content is definitely in the chunk stream**. For BioScanCast, HTML news articles will need a `trafilatura`-style pre-pass (we already use it in the existing extraction stage) before handing text to Docling, or a post-pass to filter chunks whose heading path contains "navigation" / "newsletters" / etc.
- ProMED recent-posts: by coincidence the listing page is mostly a `<table>` of recent posts, which Docling preserves as a clean 157-row Markdown table. Good for headlines (MEASLES — ROMANIA, AVIAN INFLUENZA — INDIA (19), …) but not actual post bodies — those live at permalinks we didn't probe.

### JavaScript-rendered sources

- Reuters (`https://www.reuters.com/business/healthcare-pharmaceuticals/`): **fails** with `HTTPError: 401 Client Error: HTTP Forbidden` in 13 s. Docling uses a default `requests`/`httpx` fetch that doesn't pass a browser-like user-agent, and Reuters' Cloudflare front rejects it. Any Reuters or AP-equivalent source will need an out-of-band fetch (Playwright, explicit UA, or a news API).
- ProMED listing page: the latest-posts table does render into the initial HTML, so Docling captured it fine. The individual post bodies behind each permalink are likely JS-rendered and would need a different approach.
- CIDRAP: fully server-rendered, docling converted without issue.

### Publication dates from metadata

`pub_date` came back `None` for every source. Docling exposes `DoclingDocument.origin` but its only fields are `filename`, `mimetype`, `binary_hash` — no publication / creation date. Dates exist in the body text ("published 26 March 2026", "Week 16, 11–17 April 2026") but have to be extracted with a regex / LLM pass, not from document metadata. For BioScanCast, assume Docling won't give us a publication date; we need a separate parser over the first page.

### Failures, timeouts, >3-minute runs

- **Failure**: Reuters 401. Expected for any Cloudflare-fronted news site.
- **>3 min (slow)**: WHO mpox 274.6 s, ECDC CDTR 324.6 s, Africa CDC 523.4 s. Average PDF ran at ~18 s/page with OCR off and TableFormer FAST on CPU; the mpox + ECDC PDFs are layout-dense (figures + tables + multi-column), and the Africa CDC PDF is pure images so the pipeline still runs layout detection on every page.
- **No hangs, no timeouts** — just slow.
- First-run model cost: ~40 MB of downloads the first time (RapidOCR det/rec models — still downloaded even with `do_ocr=False`, but not used; layout-heron, tableformer). Once cached, subsequent runs skip the download.

### Africa CDC failure mode (0 chunks)

The markdown for `africa_cdc_weekly_apr2026.md` is 266 bytes — 15 lines, each `<!-- image -->`. The PDF has 15 pages but Docling extracted zero text because it's published as a scanned/rasterised document rather than native-text PDF. OCR would be needed to recover anything; see the OCR cost section below for why that's not viable on this hardware.

## OCR cost evaluation (ECDC CDTR week 16)

Follow-up run via `scripts/eval_docling_ocr_cost.py`, using `convert(page_range=...)` to time individual pages and project full-doc cost. Results in [data/docling_eval/ocr/per_page_cost.json](ocr/per_page_cost.json):

| Mode | Mean per page | Projected 12-page doc | Extra bytes vs OCR-off baseline |
| --- | ---:| ---:| --- |
| `do_ocr=False` (baseline) | 22.5 s | ~4.5 min | — (3214 B on p5, 3721 B on p10) |
| `do_ocr=True`, bitmap-only (default) | 132.6 s | ~26.5 min | **+57 B on p1, +0 B on p5/p10** |
| `do_ocr=True`, `force_full_page_ocr=True` | 1055.8 s on p5 alone | ~3.5 hours | **less** content (2753 B vs 3214 B) — OCR overwrote the clean text layer |

The earlier full-doc OCR-on run was killed at 42 min before ECDC even finished — that was `force_full_page_ocr=True`. Even the saner default (~26.5 min projected) returns essentially nothing for ECDC because the "4 tables detected but 0×0" in the OCR-off run are layout-detection **false positives on chart/figure regions**, not real tables. The case counts ECDC actually publishes ("Italy (63), Spain (36), France (16) and Poland (five)") are already in the text-flow prose that OCR-off captures. Africa CDC was skipped — full-page OCR projection ≈3.5 hours per 15-page doc is unworkable on CPU.

Practical conclusion: **don't enable Docling OCR on this hardware**. Use OCR-off everywhere. For genuinely scanned PDFs like Africa CDC, route to a different ingestion path (external OCR service, GPU host, or simply skip).

## Recommendations for the BioScanCast pipeline

1. **Keep OCR off everywhere** (`do_ocr=False`). The OCR cost evaluation above showed bitmap-only OCR adds ~110 s/page of CPU work and recovers near-zero content on ECDC; full-page OCR is worse. For scanned-only PDFs (Africa CDC), OCR is the only path but the wall-clock makes it infeasible on CPU — handle out-of-band.
2. **HTML pre-filter**. Keep the existing `trafilatura` main-content extraction in the pipeline; hand Docling the cleaned article HTML rather than raw URLs, or drop Docling for HTML entirely and use the current HTML path. Nav/footer chunks from Docling's HTML pipeline are not useful.
3. **Reuters/AP**: Docling's default fetcher can't bypass Cloudflare (401). Feed it pre-fetched HTML from a UA-spoofing fetcher (the `curl` test in [data/docling_eval/sources/](sources/) showed that path works for CIDRAP), or skip news HTML sources in the Docling path.
4. **Publication date**: plan a separate extractor; Docling doesn't expose it. Tier as: HTML `<meta property="article:published_time">` / JSON-LD via trafilatura → PDF `/CreationDate` via PyMuPDF (noisy) → regex over the first chunk's body text.
5. **Budget wall-clock**: expect 2-5 minutes/PDF on CPU even with OCR off; mpox sitrep was 4.5 min, ECDC 5.4 min, Africa CDC 8.7 min (and useless without OCR). A cron-driven BioScanCast scan that touches 10+ PDFs will want a worker pool or a GPU host; don't put this behind a synchronous API call.
6. **Tables**: the WHO/MMWR tables we care about (country/case/death matrices) come through cleanly as Markdown — downstream code can parse them with a simple Markdown-table reader. ECDC's "tables" are charts/figures and need to be read from the surrounding prose instead.

## Head-to-head: Docling vs. in-tree `PdfParser`

Run via `scripts/eval_intree_pdf.py` against the same 5 local PDFs in `data/docling_eval/sources/`. In-tree stack: PyMuPDF + pdfplumber-fallback + font-size heading heuristic + `<time>`/metadata date extraction. Outputs in [data/docling_eval/intree_pdf/](intree_pdf/).

### Speed

| Doc | Docling (s) | In-tree (s) | In-tree speedup |
| --- | ---:| ---:| ---:|
| WHO mpox sitrep | 274.6 | 1.71 | **160×** |
| WHO cholera | 197.7 | 1.07 | **185×** |
| CDC MMWR | 110.5 | 1.95 | **57×** |
| ECDC CDTR | 324.6 | 1.27 | **256×** |
| Africa CDC | 523.4 | 0.02 | **fail-fast** (flagged `requires_ocr` immediately) |
| **Total** | **1430.8** (~24 min) | **6.0** | **~240×** |

### Quality (per source)

| Aspect | Docling | In-tree | Winner |
| --- | --- | --- | --- |
| **WHO mpox table** | 9×4 with wrapped headers collapsed into single cells (cleaner semantics) | 11×4 — preserves visual line breaks as separate header rows ("Number of" / "reported confirmed" / "cases" each as their own row) | **Docling** for downstream parsing |
| **WHO cholera table** | 21×8 clean — every country / cases / deaths / CFR aligned in fixed columns | 23×20 — over-segmented; numeric values shift columns row-to-row, padding cells everywhere | **Docling** clearly — in-tree's table is technically lossless but a pain to parse (column position not stable per row) |
| **CDC MMWR borderless table** | 17×2 — perfect recovery: every demographic row (sex, age, vaccination doses) with counts and percentages | 6×13 — entirely empty body, only the table caption leaked into one cell | **Docling clearly** — borderless table defeats both PyMuPDF and pdfplumber (the in-tree fallback also returns 0 tables on this page). No post-processor can recover what wasn't extracted. |
| **ECDC CDTR tables** | 4 detected, all 0×0 (false positives on chart regions) | 0 tables detected | **In-tree** — doesn't claim what isn't there |
| **Africa CDC (scanned)** | Silently produced 266-byte empty doc, took 523 s | Detected zero text → `requires_ocr` flag in 0.02 s | **In-tree** — same outcome (no usable data without OCR) but informative and instant |
| **Heading count** | 40/15/21/30/0 (mpox/cholera/MMWR/ECDC/Africa) | 40/11/17/34/1 | **Push** — comparable counts |
| **Heading hierarchy depth** | Full nested path on each chunk (`meta.headings=['Highlights']`, etc.) | Single-level stack, leaf only in `section_path`; sometimes misclassifies long body sentences as headings ("## Global surveillance data are updated monthly…") | **Docling** for clean hierarchy |
| **Multi-column reading order (MMWR)** | Page-1 "INSIDE" sidebar interleaved between body paragraphs | Page-1 Editorial Board block dropped between section 1 and section 2 | **Both have issues**, in different places |
| **Prose flow** | Paragraphs reflowed (newlines removed, soft-hyphens joined) | Preserves PDF line breaks and `­` soft hyphens (e.g. "complica­\ntions") — would need a `re.sub(r'­\s*\n', '', ...)` post-pass | **Docling** ready-to-use; in-tree needs cleanup |
| **Publication date** | `None` for all 5 | **4/5** extracted from PDF metadata (mpox 2026-03-26, cholera 2026-02-21, MMWR 2026-03-11, Africa CDC 2026-04-09); only ECDC missing | **In-tree** clearly |
| **Title from metadata** | Not surfaced | Pulled from PDF `/Title` ("Measles Outbreak — New Mexico, 2025" etc.) | **In-tree** |
| **Failure mode for scanned PDFs** | Silent empty doc, 9 minutes wasted | `is_partial=True, partial_reason='requires_ocr'` flag, instant | **In-tree** |
| **First-run model download** | ~40 MB | none | **In-tree** |
| **Memory footprint** | 1.5-2 GB at peak (transformer models loaded) | ~100 MB | **In-tree** |

### What Docling actually buys you on PDFs

The consistent, real win is **table extraction on tables that defeat rule-based heuristics**. Two distinct failure modes appeared in just 5 PDFs:

1. **Over-segmentation on dense merged-cell tables** (WHO cholera, 21×8 vs 23×20). In-tree gets the data but spreads it across noisy padding columns. **Recoverable with a ~30-line post-processor.**
2. **Total extraction failure on borderless tables** (CDC MMWR, 17×2 vs 6×13-of-empty). PyMuPDF's `find_tables()` *and* pdfplumber's `extract_tables()` both return nothing usable; Docling's TableFormer recovers the full structure. **Not recoverable on top of in-tree without bringing in a learned model** (Docling, gmft, table-transformer/TATR — same class of solution).

The first is patchable; the second is a hard ceiling. If your downstream tasks need structured table data from borderless layouts (which CDC/MMWR uses heavily for demographic breakdowns), in-tree alone won't get you there.

The other Docling advantages are smaller than expected:
- **Heading hierarchy**: technically nicer to have nested paths, but in-tree captures the same headings, just flat.
- **Reading order**: both are imperfect on multi-column MMWR.
- **OCR**: turned out unworkable on CPU regardless (see OCR cost section above) — in-tree's "fail-fast and flag" is arguably the better behaviour for batch pipelines.

In-tree advantages: **240× faster, gets the publication date, gets the title, fails fast on scanned PDFs, no model download, ~15× less memory**.

### Verdict for BioScanCast

**Keep in-tree as the default PDF path** for prose + headings + dates: it's 240× faster, gets publication dates, fails fast on scanned PDFs, and produces comparable text quality. Across the 5-PDF benchmark in-tree is good enough for the body content of every doc.

**Add Docling (or a TableFormer-class model) as a targeted fallback specifically for table extraction**, because two distinct in-tree failure modes appeared in just 5 PDFs and one of them is uncloseable without a learned model. Borderless tables in particular — common in CDC/MMWR demographic breakdowns — are unrecoverable through PyMuPDF or pdfplumber.

A practical hybrid policy:

```
PDF arrives
 ├── always run in-tree PdfParser (fast, gets metadata, prose, headings)
 ├── if downstream task consumes structured tables AND
 │   any of: (a) in-tree returns table sections with >50% empty cells,
 │           (b) page is detected to contain a table region but extraction yielded
 │               <N non-empty cells, or
 │           (c) source is from a known borderless-table family (MMWR, some WHO sitreps)
 │      → re-run that PDF through Docling for the table only
 └── if is_partial and partial_reason='requires_ocr'
        → route to external OCR service (not Docling — too slow on CPU)
```

The "trigger Docling fallback" heuristic could even be: `Docling-in-CI flagged this source family as a table-extraction failure on a previous doc → automatically use it`. That keeps the speed advantage on the 80% case while letting the maturity-of-Docling argument apply where it actually pays off.

For HTML, no change — in-tree's trafilatura + BS4 + meta-tag-date pipeline beats Docling-on-HTML across the board. Add `curl_cffi` to the fetcher when you start hitting Cloudflare-fronted sources.

## Artifacts

| What | Path |
| --- | --- |
| Docling eval script (8 sources, OCR off) | [scripts/eval_docling.py](../../scripts/eval_docling.py) |
| Docling OCR cost script (ECDC, 3 modes × 3 pages) | [scripts/eval_docling_ocr_cost.py](../../scripts/eval_docling_ocr_cost.py) |
| Docling OCR full-doc script (abandoned, kept for reference) | [scripts/eval_docling_ocr.py](../../scripts/eval_docling_ocr.py) |
| In-tree head-to-head script | [scripts/eval_intree_pdf.py](../../scripts/eval_intree_pdf.py) |
| Docling per-source Markdown / JSON / chunks | `data/docling_eval/{name}.md` / `.json` / `_chunks.json` |
| Source PDFs/HTML for side-by-side | [data/docling_eval/sources/](sources/) |
| Docling per-page OCR Markdown + cost JSON | [data/docling_eval/ocr/](ocr/) |
| In-tree per-source Markdown + JSON | [data/docling_eval/intree_pdf/](intree_pdf/) |
| Aggregate run logs | [run_log.json](run_log.json) (Docling), [intree_pdf/run_log.json](intree_pdf/run_log.json) (in-tree) |
