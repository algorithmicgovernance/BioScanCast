"""End-to-end demo script for the Search Stage.

Usage:
    python scripts/run_searching.py \
        "Will H5N1 cause more than 100 human cases in the US by December 2026?" \
        --pathogen h5n1 \
        --region "United States"

    # With output file:
    python scripts/run_searching.py \
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
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; keys must be in environment directly
from bioscancast.llm.openai_client import OpenAILLMClient

from bioscancast.stages.filtering.models import ForecastQuestion
from bioscancast.stages.searching.backends.tavily_backend import TavilyBackend
from bioscancast.stages.searching.cache import SearchCache
from bioscancast.stages.searching.pipeline import SearchStagePipeline


def _load_pathogen_choices() -> list[str]:
    """Load canonical pathogen keys from the YAML source catalog."""
    sources_yaml = Path(__file__).resolve().parents[1] / "bioscancast" / "datasets" / "sources.yaml"
    with open(sources_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    families = cfg.get("specific_pathogen_sources", {})
    if not isinstance(families, dict):
        return []

    pathogens: set[str] = set()
    for family_block in families.values():
        if not isinstance(family_block, dict):
            continue
        pathogens.update(str(k) for k in family_block.keys())
    return sorted(pathogens)


def _serialize(obj):
    """JSON serializer for datetime and other non-serializable types."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def main():
    try:
        pathogen_choices = _load_pathogen_choices()
    except Exception:
        # Keep CLI usable even if YAML is unavailable/malformed.
        pathogen_choices = []

    parser = argparse.ArgumentParser(description="Run BioScanCast Search Stage")
    parser.add_argument("question", help="The forecast question text")
    parser.add_argument(
        "--pathogen",
        default=None,
        choices=pathogen_choices if pathogen_choices else None,
        help=(
            "Pathogen name"
            + (
                "; allowed values are: " + ", ".join(pathogen_choices)
                if pathogen_choices
                else ""
            )
        ),
    )
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

    llm_client = OpenAILLMClient()
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
