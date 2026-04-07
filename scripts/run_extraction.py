"""Smoke-test script for the extraction stage.

Usage:
    python scripts/run_extraction.py [--input data/filtered_results.json]

Without --input, runs against a handful of built-in sample URLs.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime

from bioscancast.extraction.pipeline import ExtractionPipeline
from bioscancast.filtering.models import FilteredDocument


def _build_sample_docs() -> list[FilteredDocument]:
    """A small set of sample FilteredDocuments for manual smoke testing."""
    return [
        FilteredDocument(
            result_id="smoke-1",
            question_id="q1",
            url="https://www.who.int/emergencies/disease-outbreak-news",
            canonical_url="https://www.who.int/emergencies/disease-outbreak-news",
            domain="who.int",
            title="Disease Outbreak News",
            snippet="WHO disease outbreak news and updates.",
            published_date=None,
            file_type=None,
            relevance_score=0.9,
            credibility_score=1.0,
            final_score=0.95,
            source_tier="official",
            is_official_domain=True,
            selection_reasons=["official_domain"],
            extraction_priority=1,
            extraction_mode="html",
            expected_value="high",
        ),
        FilteredDocument(
            result_id="smoke-2",
            question_id="q1",
            url="https://www.cdc.gov/bird-flu/situation-summary/index.html",
            canonical_url="https://www.cdc.gov/bird-flu/situation-summary/index.html",
            domain="cdc.gov",
            title="H5N1 Bird Flu: Current Situation",
            snippet="CDC summary of H5N1 situation.",
            published_date=None,
            file_type=None,
            relevance_score=0.85,
            credibility_score=1.0,
            final_score=0.92,
            source_tier="official",
            is_official_domain=True,
            selection_reasons=["official_domain"],
            extraction_priority=2,
            extraction_mode="html",
            expected_value="high",
        ),
    ]


def _load_from_json(path: str) -> list[FilteredDocument]:
    with open(path) as f:
        data = json.load(f)
    docs = []
    for item in data:
        docs.append(
            FilteredDocument(
                result_id=item["result_id"],
                question_id=item.get("question_id", "q1"),
                url=item["url"],
                canonical_url=item.get("canonical_url", item["url"]),
                domain=item["domain"],
                title=item["title"],
                snippet=item.get("snippet", ""),
                published_date=None,
                file_type=item.get("file_type"),
                relevance_score=item.get("relevance_score", 0.5),
                credibility_score=item.get("credibility_score", 0.5),
                final_score=item.get("final_score", 0.5),
                source_tier=item.get("source_tier", "unknown"),
                is_official_domain=item.get("is_official_domain", False),
                selection_reasons=item.get("selection_reasons", []),
                extraction_priority=item.get("extraction_priority", 1),
                extraction_mode=item.get("extraction_mode", "html"),
                expected_value=item.get("expected_value", "medium"),
            )
        )
    return docs


def main():
    parser = argparse.ArgumentParser(description="Run the extraction pipeline.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to a JSON file with filtered results (output of run_filtering.py).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/extraction_results.json",
        help="Output path for extraction results.",
    )
    args = parser.parse_args()

    if args.input:
        filtered_docs = _load_from_json(args.input)
        print(f"Loaded {len(filtered_docs)} documents from {args.input}")
    else:
        filtered_docs = _build_sample_docs()
        print(f"Using {len(filtered_docs)} built-in sample documents")

    pipeline = ExtractionPipeline()
    documents = pipeline.run(filtered_docs)

    print(f"\nExtraction complete: {len(documents)} documents processed")
    for doc in documents:
        status_icon = "[OK]" if doc.status == "success" else "[FAIL]"
        chunk_info = f"{len(doc.chunks)} chunks" if doc.chunks else "no chunks"
        print(f"  {status_icon} {doc.source_url} -> {chunk_info}")
        if doc.error_message:
            print(f"         Error: {doc.error_message}")

    output = [asdict(d) for d in documents]
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
