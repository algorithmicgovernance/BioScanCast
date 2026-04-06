"""End-to-end demo script for the Search Stage.

Usage:
    python scripts/run_search_stage.py \
        "Will H5N1 cause more than 100 human cases in the US by December 2026?" \
        --pathogen h5n1 \
        --region "United States"

    # With output file:
    python scripts/run_search_stage.py \
        "How many mpox cases will be reported globally by June 2026?" \
        --pathogen mpox \
        --output data/search_results.json

Requires TAVILY_API_KEY and OPENAI_API_KEY in environment (or .env file).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; keys must be in environment directly

from bioscancast.filtering.models import ForecastQuestion
from bioscancast.llm.client import OpenAIClient
from bioscancast.stages.search_stage.backends.tavily_backend import TavilyBackend
from bioscancast.stages.search_stage.cache import SearchCache
from bioscancast.stages.search_stage.pipeline import SearchStagePipeline


def _serialize(obj):
    """JSON serializer for datetime and other non-serializable types."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def main():
    parser = argparse.ArgumentParser(description="Run BioScanCast Search Stage")
    parser.add_argument("question", help="The forecast question text")
    parser.add_argument("--pathogen", default=None, help="Pathogen name (e.g. h5n1, mpox)")
    parser.add_argument("--region", default=None, help="Geographic region")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file path")
    parser.add_argument("--no-cache", action="store_true", help="Disable search cache")
    args = parser.parse_args()

    question = ForecastQuestion(
        id="demo_001",
        text=args.question,
        created_at=datetime.now(timezone.utc),
        pathogen=args.pathogen,
        region=args.region,
    )

    llm_client = OpenAIClient()
    search_backend = TavilyBackend()
    cache = None if args.no_cache else SearchCache()

    pipeline = SearchStagePipeline(
        search_backend=search_backend,
        llm_client=llm_client,
        cache=cache,
        backend_name="tavily",
    )

    print(f"Running search stage for: {question.text}")
    print(f"  Pathogen: {question.pathogen or 'not specified'}")
    print(f"  Region:   {question.region or 'not specified'}")
    print()

    results = pipeline.run(question)

    print(f"Search stage returned {len(results)} results\n")

    output = json.dumps(
        [asdict(r) for r in results],
        indent=2,
        default=_serialize,
    )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Results written to {args.output}")
    else:
        print(output)

    if cache:
        cache.close()


if __name__ == "__main__":
    main()
