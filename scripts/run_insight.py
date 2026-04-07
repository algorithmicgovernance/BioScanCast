"""Manual smoke test for the insight pipeline with a real OpenAI client.

Usage:
    OPENAI_API_KEY=sk-... python scripts/run_insight.py

This is NOT part of CI.  It uses synthetic documents so it can run
independently of the extraction stage.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime

# Use the real client (requires openai package and API key)
from bioscancast.llm.openai_client import OpenAILLMClient
from bioscancast.insight.pipeline import InsightPipeline
from bioscancast.insight.config import InsightConfig

# Synthetic test data
from bioscancast.tests.fixtures.insight.synthetic_documents import (
    DOC_WHO_SUDAN,
    DOC_CDC_H5N1,
    DOC_REUTERS_MPOX,
    QUESTION_SUDAN,
    QUESTION_H5N1,
)


def main() -> None:
    print("=== BioScanCast Insight Stage — Smoke Test ===\n")

    client = OpenAILLMClient()
    config = InsightConfig(
        retrieval_top_k=5,
        max_chunks_per_document=5,
        max_input_tokens_per_run=50_000,
    )
    pipeline = InsightPipeline(llm_client=client, config=config)

    question = QUESTION_SUDAN
    documents = [DOC_WHO_SUDAN, DOC_CDC_H5N1]

    print(f"Question: {question.text}")
    print(f"Documents: {len(documents)}")
    print()

    result = pipeline.run(question, documents)

    print(f"Documents processed: {result.documents_processed}")
    print(f"Documents skipped:   {result.documents_skipped}")
    print(f"Records extracted:   {len(result.records)}")
    print()

    for i, record in enumerate(result.records, 1):
        print(f"--- Record {i} ---")
        print(f"  Type:       {record.event_type}")
        print(f"  Confidence: {record.confidence:.2f}")
        print(f"  Location:   {record.location}")
        print(f"  ISO Code:   {record.iso_country_code}")
        print(f"  Pathogen:   {record.pathogen}")
        print(f"  Metric:     {record.metric_name} = {record.metric_value} {record.metric_unit or ''}")
        print(f"  Date:       {record.event_date}")
        print(f"  Summary:    {record.summary}")
        print(f"  Sources:    {len(record.sources)}")
        for src in record.sources:
            print(f"    - {src.document_id} / {src.chunk_id}")
            print(f"      Quote: {src.quote[:80]}...")
        print()

    print("=== Budget Summary ===")
    print(json.dumps(result.budget_summary, indent=2))

    if result.notes:
        print("\n=== Notes ===")
        for note in result.notes:
            print(f"  - {note}")


if __name__ == "__main__":
    main()
