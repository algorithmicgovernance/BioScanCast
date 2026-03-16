# BioScanCast

BioScanCast is an open source pipeline that uses large language models and automated web retrieval to produce forecasts for biosecurity related questions.

The system gathers information from the internet, filters relevant sources, extracts structured insights, and produces probabilistic forecasts with confidence scores. The project also evaluates model forecasts against human expert forecasts.

This repository contains the full pipeline implementation, benchmarking framework, and tooling required to reproduce experiments.

---

# Project Goals

1. Build an open source forecasting system for biosecurity questions.
2. Benchmark model forecasts against human expert forecasters.
3. Provide a fully reproducible research pipeline suitable for publication.
4. Produce accessible outputs including technical documentation and public explanations.

---

# High Level Pipeline

The system follows a modular pipeline with five stages.

1. Search stage
   Collect candidate sources from the internet.

2. Filtering stage
   Identify credible and relevant sources.

3. Extraction stage
   Retrieve and clean content from selected sources.

4. Insight stage
   Extract structured information such as events and timelines.

5. Forecasting stage
   Use structured information to generate forecasts and confidence scores.

Each stage is implemented as an independent module so developers can work on components without affecting the rest of the system.

---

# Repository Structure

```
bioscancast/
│
├── bioscancast/
│   ├── pipeline/
│   ├── stages/
│   ├── schemas/
│   ├── llm/
│   ├── retrieval/
│   ├── evaluation/
│   ├── datasets/
│   └── utils/
│
├── configs/
├── data/
├── scripts/
├── notebooks/
├── tests/
└── docs/
```

The sections below describe the purpose of each directory.

---

# Core Package

The `bioscancast/` directory contains the main application code.

```
bioscancast/
```

This package implements the forecasting pipeline and supporting modules.

---

# Pipeline

```
bioscancast/pipeline/
```

Responsible for coordinating execution across stages.

Files:

**orchestrator.py**
Controls pipeline flow. Calls each stage sequentially.

**pipeline_runner.py**
Entry point for running the full pipeline.

**pipeline_types.py**
Shared data types used to pass outputs between stages.

Developers modifying pipeline order or execution logic should work here.

---

# Pipeline Stages

```
bioscancast/stages/
```

Each stage of the forecasting pipeline is implemented in its own folder.

Stages should remain independent and communicate only through defined schemas.

---

## Search Stage

```
bioscancast/stages/search_stage/
```

Purpose:

Generate candidate sources relevant to a forecasting question.

Expected tasks:

• build search queries
• call search APIs
• retrieve top results

Expected outputs:

```
List[SearchResult]
```

Example modules:

search_engine.py
query_builder.py
search_clients/

---

## Filtering Stage

```
bioscancast/stages/filtering_stage/
```

Purpose:

Identify credible and relevant sources.

Expected tasks:

• LLM relevance classification
• source credibility checks
• removal of duplicate or low quality URLs

Expected outputs:

```
List[FilteredURL]
```

Example modules:

url_ranker.py
source_validator.py
relevance_model.py

---

## Extraction Stage

```
bioscancast/stages/extraction_stage/
```

Purpose:

Retrieve and normalize content from selected sources.

Expected tasks:

• scrape HTML pages
• download PDFs
• parse documents
• clean text

Expected outputs:

```
List[Document]
```

Example modules:

scraper.py
html_parser.py
pdf_parser.py
text_cleaner.py

---

## Insight Stage

```
bioscancast/stages/insight_stage/
```

Purpose:

Extract structured information from text.

Expected tasks:

• event extraction
• timeline construction
• key insight identification

Expected outputs:

```
DataFrame[InsightRecord]
```

Example modules:

information_extractor.py
event_parser.py
timeline_builder.py
dataframe_builder.py

---

## Forecasting Stage

```
bioscancast/stages/forecasting_stage/
```

Purpose:

Produce probabilistic forecasts based on extracted insights.

Expected tasks:

• generate model prompts
• apply reasoning models
• calibrate probabilities
• produce confidence scores

Expected outputs:

```
ForecastOutput
```

Example modules:

forecaster.py
prompt_templates.py
confidence_calibration.py

---

# Schemas

```
bioscancast/schemas/
```

Defines structured data objects shared between pipeline stages.

Examples:

search_result.py
document.py
extracted_event.py
forecast_output.py

All stages should communicate using these schemas. Do not pass raw dictionaries between stages.

---

# LLM Integration

```
bioscancast/llm/
```

Provides abstraction layers for language models.

Supported providers may include:

• OpenAI
• Anthropic
• Local models

Example files:

llm_client.py
openai_client.py
anthropic_client.py

Stages should call these clients rather than directly interacting with APIs.

---

# Retrieval Utilities

```
bioscancast/retrieval/
```

Tools for document retrieval and embedding.

Examples:

document_store.py
embedding_model.py
chunking.py

Used by extraction and insight stages.

---

# Evaluation

```
bioscancast/evaluation/
```

Contains benchmarking and evaluation logic.

Examples:

benchmark_loader.py
scoring.py
brier_score.py
calibration_metrics.py
human_comparison.py

Used to compare model forecasts against human forecasts.

---

# Datasets

```
bioscancast/datasets/
```

Contains definitions for forecasting datasets and curated source lists.

Examples:

forecast_questions.py
biosecurity_sources.py

---

# Utilities

```
bioscancast/utils/
```

General purpose helpers used throughout the codebase.

Examples:

logging utilities
configuration loading
rate limiting
caching

---

# Configurations

```
configs/
```

Configuration files for models, scraping behavior, and pipeline parameters.

Examples:

model configuration
API settings
scraping limits
LLM prompt settings

These files allow experimentation without modifying code.

---

# Data

```
data/
```

Stores intermediate and benchmark data.

Subdirectories:

raw
original scraped data

processed
cleaned datasets

benchmarks
forecasting evaluation datasets

---

# Scripts

```
scripts/
```

Command line tools used to run experiments and pipelines.

Examples:

run_pipeline.py
Runs the full forecasting pipeline.

run_benchmark.py
Evaluates model forecasts against benchmark datasets.

scrape_sources.py
Bulk scraping utility for collecting documents.

evaluate_forecasts.py
Computes evaluation metrics.

Scripts are intended for operational tasks rather than reusable code.

---

# Notebooks

```
notebooks/
```

Used for exploratory analysis and experimentation.

Examples:

model experiments
prompt exploration
benchmark analysis

Notebook code should not be required for the main pipeline.

---

# Tests

```
tests/
```

Unit and integration tests for pipeline components.

Examples:

stage level tests
pipeline execution tests
schema validation tests

Each pipeline stage should include test coverage.

---

# Documentation

```
docs/
```

Project documentation and architecture notes.

Examples:

system architecture
pipeline design
benchmark methodology
API documentation

These documents support research publication and developer onboarding.

---

# Development Principles

1. Pipeline stages must remain modular.
2. Data passed between stages must use schemas.
3. Stages should not import logic from other stages.
4. Experimental code should live in notebooks or scripts.
5. Reproducibility is a core requirement.

---

# Running the Pipeline

Example:

```
python scripts/run_pipeline.py
```

Example benchmark run:

```
python scripts/run_benchmark.py
```

---

# Contributing

Developers should work within a single pipeline stage whenever possible. Changes that affect data contracts or schemas should be discussed before merging.

All contributions should include tests.