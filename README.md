# BioScanCast

BioScanCast is an open source pipeline for biosecurity forecasting using LLMs and automated web retrieval.

The system retrieves internet sources, filters relevant documents, extracts structured information, and is intended to support probabilistic forecasting and evaluation against human forecasters.

Current repository contents include:

* modular pipeline stages
* shared schemas and LLM abstractions
* retrieval and extraction tooling
* benchmarking and evaluation infrastructure
* smoke-test and operational scripts

The forecasting stage is not yet implemented.

---

# Project Goals

1. Build an open source forecasting system for biosecurity questions.
2. Benchmark model forecasts against human forecasters.
3. Provide a reproducible research pipeline suitable for publication.
4. Produce accessible technical and public-facing outputs.

---

# Pipeline Status

Implemented:

```text
Search -> Filtering -> Extraction -> Insight
```

Planned:

```text
Forecasting
```

Current capabilities include:

* LLM query decomposition
* web/news retrieval via Tavily
* heuristic + optional LLM filtering
* HTML/PDF extraction and chunking
* hybrid BM25 + embedding retrieval
* structured fact extraction with provenance tracking

---

# Pipeline Overview

## 1. Search Stage

Collect candidate internet sources.

Features:

* LLM query decomposition
* Tavily retrieval backend
* source tier scoring
* dashboard injection
* URL normalization + deduplication

Output:

```python
List[SearchResult]
```

---

## 2. Filtering Stage

Identify credible and relevant sources.

Features:

* heuristic relevance scoring
* source credibility scoring
* duplicate removal
* optional LLM review
* extraction-priority assignment

Output:

```python
List[FilteredDocument]
```

---

## 3. Extraction Stage

Fetch and normalize source content.

Features:

* HTML/PDF fetching
* HTML/PDF parsing
* table extraction
* chunk normalization
* metadata extraction

Output:

```python
List[Document]
```

---

## 4. Insight Stage

Convert extracted text into structured facts.

Features:

* BM25 retrieval
* embedding retrieval
* hybrid reranking
* structured fact extraction
* provenance tracking
* hallucination filtering
* cross-document deduplication

Output:

```python
List[InsightRecord]
```

Design principle:

```text
one chunk -> one extraction call
```

Each fact must include:

* supporting quote
* source chunk
* source URL

Facts failing substring verification are discarded.

Current limitations:

* disconnected from extraction outputs in smoke tests
* no temporal reasoning layer
* no forecasting integration

---

## 5. Forecasting Stage

Planned but not implemented.

Intended responsibilities:

* probabilistic forecasting
* calibration
* confidence estimation
* forecast evaluation

---

# Repository Structure

```text
bioscancast/
├── bioscancast/
│   ├── datasets/      Source registries and tier definitions
│   ├── extraction/    Fetching, parsing, and chunking
│   ├── filtering/     Relevance filtering and reranking
│   ├── insight/       Retrieval and insight extraction
│   ├── llm/           LLM client abstractions
│   ├── schemas/       Shared data models
│   ├── stages/
│   │   ├── search_stage/
│   │   └── eval_stage/
│   └── tests/         Unit and integration tests
├── data/
│   ├── raw/
│   ├── processed/
│   └── docling_eval/
├── evaluation/
├── scripts/
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

# Core Modules

| Module                 | Purpose                                    |
| ---------------------- | ------------------------------------------ |
| `datasets/`            | curated source registries and source tiers |
| `extraction/`          | fetching, parsing, chunking                |
| `filtering/`           | source filtering and ranking               |
| `insight/`             | retrieval and fact extraction              |
| `llm/`                 | model abstractions                         |
| `schemas/`             | shared structured contracts                |
| `stages/search_stage/` | retrieval stage                            |
| `stages/eval_stage/`   | evaluation tooling                         |

---

# Stage Details

## Search Stage

```text
bioscancast/stages/search_stage/
```

Implemented modules:

| File                         | Purpose                    |
| ---------------------------- | -------------------------- |
| `pipeline.py`                | orchestration              |
| `query_decomposition.py`     | LLM sub-query generation   |
| `tier_resolution.py`         | source credibility scoring |
| `dashboard_lookup.py`        | dashboard injection        |
| `url_normalization.py`       | canonicalization + dedup   |
| `backends/tavily_backend.py` | Tavily backend             |

Current features:

* 5–8 LLM-generated subqueries
* backend abstraction via `SearchBackend`
* source tier + freshness scoring
* aggregator-domain flagging
* non-content URL filtering

Known limitations:

* English-only retrieval
* hardcoded dashboard mappings
* no multilingual retrieval

---

## Filtering Stage

```text
bioscancast/filtering/
```

Implemented modules:

| File               | Purpose                        |
| ------------------ | ------------------------------ |
| `pipeline.py`      | orchestration                  |
| `heuristics.py`    | heuristic scoring              |
| `llm_filter.py`    | LLM adjudication               |
| `reranker.py`      | borderline reranking           |
| `deduplication.py` | duplicate handling             |
| `postprocess.py`   | extraction-priority assignment |

Current features:

* heuristic relevance scoring
* source credibility scoring
* optional LLM review
* domain caps
* extraction-mode assignment

---

## Extraction Stage

```text
bioscancast/extraction/
```

Implemented modules:

| File                     | Purpose                   |
| ------------------------ | ------------------------- |
| `pipeline.py`            | orchestration             |
| `fetcher.py`             | network retrieval         |
| `chunking.py`            | chunk normalization       |
| `parsers/html_parser.py` | HTML extraction           |
| `parsers/pdf_parser.py`  | PDF extraction            |
| `docling_refiner.py`     | optional table refinement |

Current features:

* browser-fingerprinted fetching via `curl_cffi`
* BeautifulSoup + trafilatura HTML parsing
* PyMuPDF PDF parsing
* pdfplumber table fallback
* chunk normalization
* metadata extraction
* document-level provenance tracking

### PDF Table Extraction (Docling Refiner)

The default PDF pipeline uses PyMuPDF + pdfplumber with an optional Docling TableFormer refinement pass.

The first refinement run downloads Docling models (~40 MB) into:

```text
~/.cache/huggingface/
```

Models remain resident in memory (~1.5 GB) for the process lifetime.

Controlled via:

```python
ExtractionConfig.enable_docling_refiner
```

When disabled, no Docling imports occur.

Current limitations:

* OCR not implemented
* scanned PDFs return `requires_ocr`
* no persistent document store
* extraction is currently in-memory only

---

## Insight Stage

```text
bioscancast/insight/
```

Implemented modules:

| File                            | Purpose             |
| ------------------------------- | ------------------- |
| `pipeline.py`                   | orchestration       |
| `retrieval/bm25.py`             | lexical retrieval   |
| `retrieval/embeddings.py`       | embedding retrieval |
| `retrieval/hybrid.py`           | hybrid reranking    |
| `extraction/chunk_extractor.py` | fact extraction     |

Current features:

* BM25 retrieval
* embedding similarity retrieval
* hybrid scoring
* keyword reranking
* chunk-level extraction
* quote-based hallucination guards
* provenance linking
* cross-document deduplication

---

# Evaluation

```text
bioscancast/stages/eval_stage/
```

Implemented modules:

| File               | Purpose                   |
| ------------------ | ------------------------- |
| `evaluator.py`     | orchestration             |
| `scoring.py`       | forecast scoring          |
| `calibration.py`   | calibration metrics       |
| `compare.py`       | model vs human comparison |
| `visualisation.py` | plots and reporting       |

Repository datasets:

```text
bioscancast_forecasts.csv
bioscancast_questions.csv
```

---

# Schemas

```text
bioscancast/schemas/
```

Shared stage contracts.

Key schemas:

| File                | Purpose                      |
| ------------------- | ---------------------------- |
| `document.py`       | extracted documents + chunks |
| `insight_record.py` | extracted facts              |

Additional filtering models live in:

```text
bioscancast/filtering/models.py
```

including:

* `ForecastQuestion`
* `SearchResult`
* `FilteredDocument`

Stages should communicate through schemas rather than raw dictionaries.

---

# LLM Integration

```text
bioscancast/llm/
```

Current files:

| File               | Purpose                            |
| ------------------ | ---------------------------------- |
| `base.py`          | shared protocol + token accounting |
| `client.py`        | legacy/simple OpenAI wrapper       |
| `openai_client.py` | structured extraction client       |
| `fake_client.py`   | testing client                     |

The repository currently contains two partially overlapping interfaces:

```text
bioscancast/llm/base.py
bioscancast/llm/client.py
```

These should eventually be unified.

## Historical-replay mode (benchmarking against human forecasters)

When benchmarking the pipeline against human forecasters on past questions,
the model must not be allowed to see sources that didn't exist (or contained
different content) at the time the human forecasted. Historical-replay mode
enforces this by reading a single per-question field, `ForecastQuestion.as_of_date`:

- When `as_of_date` is `None` (default), the pipeline behaves exactly as in
  live mode. No code paths change.
- When `as_of_date` is set, the search backend receives `end_date=as_of_date`,
  the cache key incorporates the cutoff, post-retrieval filtering drops any
  result dated after the cutoff (and any undated result whose date cannot be
  cheaply recovered), dashboard URLs are rewritten to the closest Wayback
  snapshot at or before the cutoff (or suppressed if none exists), and the
  extraction stage fetches from Wayback. Wayback fallback to live is logged
  at INFO and recorded in `Document.fetch_strategy`, never silent.

The LLM "historical roleplay" prompt is *not* automatically enabled by
`as_of_date`; it lives behind a separate `historical_roleplay=True` flag on
`SearchStagePipeline` because its effect on query quality is harder to
predict. Turn it on for the benchmark and off for production.

What this mode does NOT fix: the LLMs themselves were trained on data that
postdates many of our benchmark questions. Retrieval fairness ≠ model
fairness. The `retrieval_free_baseline_forecast` metric in
`bioscancast/stages/eval_stage/contamination.py` reports how well the LLM
forecasts with no evidence at all; a small gap between that and the full
pipeline is itself evidence of training-data leakage and must be reported
alongside the headline Brier/log scores.

`filter_caught_contamination_rate` is also exposed by the same module. It
is a **lower bound** on contamination — it only counts post-cutoff results
whose `published_date` is known. Undated results and results whose content
changed post-cutoff are invisible to it. Reports MUST surface this caveat;
the metric's docstring repeats it for the same reason.

---

# Datasets

```text
bioscancast/datasets/
```

Curated source definitions and credibility tiers.

| File                     | Purpose                  |
| ------------------------ | ------------------------ |
| `biosecurity_sources.py` | curated source registry  |
| `source_tiers.py`        | source credibility tiers |

---

# Scripts

```text
scripts/
```

Operational and smoke-test utilities.

| Script                | Purpose                     |
| --------------------- | --------------------------- |
| `run_search_stage.py` | run search stage            |
| `run_filtering.py`    | run filtering stage         |
| `run_extraction.py`   | run extraction stage        |
| `run_insight.py`      | run insight smoke test      |
| `eval_docling.py`     | Docling evaluation          |
| `eval_hybrid_pdf.py`  | PDF extraction benchmarking |

Scripts are intended for operational workflows rather than reusable library APIs.

---

# Running the Pipeline

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Additional packages:

```bash
pip install openai tavily-python python-dotenv
```

---

## Environment Variables

Create `.env`:

```bash
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

---

## Search Stage

```bash
python scripts/run_search_stage.py \
  "Will H5N1 cause more than 100 human cases in the US by December 2026?" \
  --pathogen h5n1 \
  --region "United States"
```

Optional JSON output:

```bash
python scripts/run_search_stage.py \
  "How many mpox cases will be reported globally by June 2026?" \
  --pathogen mpox \
  --output data/search_results.json
```

---

## Filtering Stage

```bash
python scripts/run_filtering.py
```

Current limitation:

* uses hardcoded sample inputs rather than automatic search-stage ingestion

Output:

```text
data/filtered_results.json
```

---

## Extraction Stage

Smoke-test mode:

```bash
python scripts/run_extraction.py
```

Using filtered-document JSON:

```bash
python scripts/run_extraction.py \
  --input data/filtered_results.json
```

Output:

```text
data/extraction_results.json
```

---

## Insight Stage

```bash
python scripts/run_insight.py
```

Current limitation:

* uses synthetic documents rather than extraction outputs

---

# Closest Current End-to-End Flow

```bash
python scripts/run_search_stage.py \
  "Will mpox cases increase in Uganda in 2026?" \
  --pathogen mpox \
  --region Uganda \
  --output data/search_results.json && \
python scripts/run_filtering.py && \
python scripts/run_extraction.py \
  --input data/filtered_results.json
```

`bioscancast/main.py` currently contains pseudocode only and is not yet a runnable orchestrator.

---

# Tests

```text
bioscancast/tests/
```

Includes:

* extraction tests
* retrieval tests
* pipeline tests
* schema validation
* search-stage integration tests

Run all tests:

```bash
pytest
```

Run selected tests:

```bash
pytest bioscancast/tests/test_extraction_pipeline.py
pytest bioscancast/tests/test_insight_pipeline.py
```

Live fetch tests are marked:

```python
@pytest.mark.live
```

and skipped by default.

Run with:

```bash
pytest --live
```

---

# Dependencies

Important dependencies:

| Dependency   | Usage                               |
| ------------ | ----------------------------------- |
| `curl_cffi`  | browser-fingerprinted HTTP fetching |
| `rank_bm25`  | lexical retrieval                   |
| `PyMuPDF`    | primary PDF parsing                 |
| `pdfplumber` | fallback PDF table extraction       |

`curl_cffi` is used in:

```text
bioscancast/extraction/fetcher.py
```

The impersonation profile is configurable via:

```python
ExtractionConfig.impersonate
```

---

# Development Principles

1. Keep pipeline stages modular.
2. Use schemas between stages.
3. Prefer structured interfaces over raw dictionaries.
4. Keep experimental workflows in scripts or notebooks.
5. Prioritize reproducibility.
6. Treat provenance and auditability as first-class concerns.

---

# Known Architectural Gaps

Major missing components:

1. unified end-to-end orchestrator
2. extraction → insight integration
3. OCR support
4. forecasting stage implementation
5. persistent storage/vector DB layer
6. unified LLM abstraction
7. multilingual retrieval
