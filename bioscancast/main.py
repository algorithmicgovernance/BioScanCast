"""End-to-end pipeline orchestrator: search -> filter -> extract -> insight.

Runs all four stages against a single forecast question, persisting each
stage's output and a running manifest under ``data/runs/{qid}/{run_id}/``
so a crashed run still has partial artifacts for debugging.

Usage:

    python -m bioscancast.main QUESTION_ID [--csv PATH] [--as-of-date Y-M-D] ...

See ``--help`` for the full flag list.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from bioscancast.llm.base import LLMResponse
from bioscancast.llm.openai_client import OpenAILLMClient
from bioscancast.llm.pricing import (
    UnknownModelError,
    estimate_cost,
)
from bioscancast.orchestration import persistence

from bioscancast.stages.evaluation.loaders import load_question_by_id
from bioscancast.stages.searching.backends.tavily_backend import TavilyBackend
from bioscancast.stages.searching.cache import SearchCache
from bioscancast.stages.searching.pipeline import SearchStagePipeline
from bioscancast.stages.extraction.config import ExtractionConfig
from bioscancast.stages.extraction.pipeline import ExtractionPipeline
from bioscancast.stages.filtering.config import FILTER_CONFIG
from bioscancast.stages.filtering.models import ForecastQuestion
from bioscancast.stages.filtering.pipeline import FilteringPipeline
from bioscancast.stages.insight.config import InsightConfig
from bioscancast.stages.insight.pipeline import InsightPipeline, InsightRunResult


DEFAULT_CSV = "bioscancast/stages/eval_stage/bioscancast_questions.csv"
DEFAULT_OUT_ROOT = "data/runs"

logger = logging.getLogger("bioscancast.main")


class PipelineError(RuntimeError):
    """Raised when a stage fails; wraps the underlying exception with stage name."""

    def __init__(self, stage: str, original: BaseException) -> None:
        super().__init__(f"Pipeline failed in stage {stage!r}: {original}")
        self.stage = stage
        self.original = original


class _UsageTrackingClient:
    """Wraps an LLMClient and accumulates per-model token usage.

    Used for the search and filter stages so the orchestrator can include
    their cost in the final estimate. The insight stage already tracks its
    own usage via BudgetTracker — passing the raw client to insight avoids
    double-counting.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.per_model: dict[str, dict[str, int]] = defaultdict(
            lambda: {"input_tokens": 0, "output_tokens": 0, "calls": 0}
        )

    def generate_json(self, **kwargs) -> LLMResponse:
        response = self._inner.generate_json(**kwargs)
        bucket = self.per_model[response.model]
        bucket["input_tokens"] += response.input_tokens
        bucket["output_tokens"] += response.output_tokens
        bucket["calls"] += 1
        return response

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        # The OpenAI embed() call doesn't expose usage today; the insight
        # pipeline doesn't currently track embedding cost either. If
        # embeddings become a material cost line, hook tokenizer-based
        # estimation in here.
        return self._inner.embed(texts, model=model)

    def snapshot(self) -> dict[str, dict[str, int]]:
        """Return a plain-dict copy of cumulative usage for delta-ing
        between stages."""
        return {model: dict(counts) for model, counts in self.per_model.items()}


# ----------------------------------------------------------------------------
# CLI parsing and question construction
# ----------------------------------------------------------------------------


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="bioscancast.main",
        description="Run the BioScanCast pipeline end-to-end for one question.",
    )
    parser.add_argument(
        "question_id",
        nargs="?",
        default="demo",
        help="ID of the question in the CSV (e.g. q7).",
    )
    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help=f"Path to the question CSV. Default: {DEFAULT_CSV}",
    )
    parser.add_argument(
        "--out-root",
        default=DEFAULT_OUT_ROOT,
        help=f"Run-artifact root directory. Default: {DEFAULT_OUT_ROOT}",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Override the UTC-timestamp run directory name.",
    )
    parser.add_argument(
        "--as-of-date",
        default=None,
        help="Historical-replay cutoff (YYYY-MM-DD). Omit for live mode.",
    )
    parser.add_argument(
        "--target-date",
        default=None,
        help="Override CSV-derived target_date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--question",
        default=None,
        help="Override question text. If omitted, load from CSV by question_id.",
    )
    parser.add_argument("--region", default=None, help="Override region field.")
    parser.add_argument("--pathogen", default=None, help="Override pathogen field.")
    parser.add_argument(
        "--event-type", default=None, help="Override event_type field."
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the search-stage cache.",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=None,
        help="Override InsightConfig.max_input_tokens_per_run.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Set log level to INFO.",
    )
    return parser.parse_args(argv)


def _parse_date(arg: Optional[str]) -> Optional[datetime]:
    if arg is None:
        return None
    return datetime.strptime(arg, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _apply_overrides(q: ForecastQuestion, args: argparse.Namespace) -> ForecastQuestion:
    target_date = _parse_date(args.target_date)
    return ForecastQuestion(
        id=q.id,
        text=q.text,
        created_at=q.created_at,
        target_date=target_date if target_date is not None else q.target_date,
        region=args.region or q.region,
        pathogen=args.pathogen or q.pathogen,
        event_type=args.event_type or q.event_type,
        resolution_criteria=q.resolution_criteria,
        as_of_date=q.as_of_date,
    )


# ----------------------------------------------------------------------------
# Stage timing and console output
# ----------------------------------------------------------------------------


@contextmanager
def _stage_timer(manifest: dict, stage: str) -> Iterator[None]:
    manifest["stage_timings"].setdefault(stage, None)
    manifest["current_stage"] = stage
    print(f"[{stage}] starting...", flush=True)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        manifest["stage_timings"][stage] = round(elapsed, 3)


def _log_summary(stage: str, summary: str, elapsed: Optional[float]) -> None:
    tail = f"({elapsed:.1f}s)" if elapsed is not None else ""
    print(f"[{stage}] {summary} {tail}", flush=True)


# ----------------------------------------------------------------------------
# Cost estimation
# ----------------------------------------------------------------------------


def _merge_usage(*usage_dicts: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    """Combine per-model usage dicts from different stages."""
    merged: dict[str, dict[str, int]] = defaultdict(
        lambda: {"input_tokens": 0, "output_tokens": 0, "calls": 0}
    )
    for usage in usage_dicts:
        for model, counts in (usage or {}).items():
            for k in ("input_tokens", "output_tokens", "calls"):
                merged[model][k] += int(counts.get(k, 0))
    return dict(merged)


def _usage_delta(
    before: dict[str, dict[str, int]],
    after: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    """Per-model usage that accrued between two snapshots of the same
    tracker. Models with no change are omitted."""
    delta: dict[str, dict[str, int]] = {}
    for model, counts in after.items():
        b = before.get(model, {})
        d = {
            k: int(counts.get(k, 0)) - int(b.get(k, 0))
            for k in ("input_tokens", "output_tokens", "calls")
        }
        if any(d.values()):
            delta[model] = d
    return delta


def _estimate_total_cost(per_model: dict[str, dict[str, int]]) -> tuple[float, list[str]]:
    """Return (usd_total, list_of_warnings)."""
    total = 0.0
    warnings: list[str] = []
    for model, counts in per_model.items():
        try:
            total += estimate_cost(
                model,
                input_tokens=int(counts.get("input_tokens", 0)),
                output_tokens=int(counts.get("output_tokens", 0)),
            )
        except UnknownModelError as exc:
            warnings.append(str(exc))
    return total, warnings


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------


def run_pipeline(args: argparse.Namespace) -> InsightRunResult:
    csv_path = Path(args.csv)

    if args.question is None:
        if not csv_path.exists():
            raise FileNotFoundError(f"Question CSV not found: {csv_path}")

        question = load_question_by_id(
            csv_path,
            args.question_id,
            as_of_date=_parse_date(args.as_of_date),
        )
    else:
        question = ForecastQuestion(
            id=args.question_id,
            text=args.question,
            created_at=datetime.now(timezone.utc),
            target_date=_parse_date(args.target_date),
            region=args.region,
            pathogen=args.pathogen,
            event_type=args.event_type,
            resolution_criteria=None,
            as_of_date=_parse_date(args.as_of_date),
        )
    question = _apply_overrides(question, args)

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root)
    run_dir = persistence.make_run_dir(out_root, question.id, run_id)

    print(f"=== Pipeline run: {question.id} ({run_id}) ===")
    print(
        f"question:  {question.text}\n"
        f"pathogen:  {question.pathogen!r}\n"
        f"region:    {question.region!r}\n"
        f"target:    {question.target_date.date() if question.target_date else None}\n"
        f"as_of:     {question.as_of_date.date() if question.as_of_date else None}\n"
        f"artifacts: {run_dir}\n",
    )

    persistence.save_question(run_dir, question)

    insight_config = InsightConfig()
    if args.max_input_tokens:
        insight_config.max_input_tokens_per_run = args.max_input_tokens

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "question_id": question.id,
        "csv_path": str(csv_path),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "stage_timings": {},
        "current_stage": None,
        "errored_stage": None,
        "error_message": None,
        "config": {
            "filter": dict(FILTER_CONFIG),
            "extraction": asdict(ExtractionConfig()),
            "insight": asdict(insight_config),
        },
    }
    persistence.save_manifest(run_dir, manifest)

    shared_llm_raw = OpenAILLMClient()
    shared_llm = _UsageTrackingClient(shared_llm_raw)

    # Per-stage usage is captured by snapshotting the shared tracker
    # before/after each stage that uses it. Search and filter share one
    # client; insight reports its own budget_summary.
    stage_usage: dict[str, dict[str, dict[str, int]]] = {}

    try:
        with _stage_timer(manifest, "search"):
            backend = TavilyBackend()
            cache = None if args.no_cache else SearchCache()
            search_pipeline = SearchStagePipeline(
                search_backend=backend,
                llm_client=shared_llm,
                cache=cache,
                backend_name="tavily",
            )
            search_results = search_pipeline.run(question)
            persistence.save_search(run_dir, search_results)
            if cache:
                cache.close()
        usage_after_search = shared_llm.snapshot()
        stage_usage["search"] = usage_after_search
        _log_summary(
            "search", f"{len(search_results)} results",
            manifest["stage_timings"]["search"],
        )
        persistence.save_manifest(run_dir, manifest)

        with _stage_timer(manifest, "filter"):
            filter_pipeline = FilteringPipeline(llm_client=shared_llm)
            filtered_docs = filter_pipeline.run(question, search_results)
            persistence.save_filtered(run_dir, filtered_docs)
        stage_usage["filter"] = _usage_delta(usage_after_search, shared_llm.snapshot())
        _log_summary(
            "filter", f"{len(filtered_docs)} docs",
            manifest["stage_timings"]["filter"],
        )
        persistence.save_manifest(run_dir, manifest)

        with _stage_timer(manifest, "extract"):
            extraction_pipeline = ExtractionPipeline(as_of_date=question.as_of_date)
            documents = extraction_pipeline.run(filtered_docs)
            persistence.save_documents(run_dir, documents)
        n_ok = sum(1 for d in documents if d.status == "success")
        _log_summary(
            "extract", f"{n_ok}/{len(documents)} success",
            manifest["stage_timings"]["extract"],
        )
        persistence.save_manifest(run_dir, manifest)

        with _stage_timer(manifest, "insight"):
            insight_pipeline = InsightPipeline(
                llm_client=shared_llm_raw,
                config=insight_config,
            )
            insight_result = insight_pipeline.run(question, documents)
            persistence.save_insight(run_dir, insight_result)
        stage_usage["insight"] = insight_result.budget_summary.get("per_model") or {}
        budget = insight_result.budget_summary
        _log_summary(
            "insight",
            f"{len(insight_result.records)} records | "
            f"in={budget.get('total_input_tokens', 0):,} "
            f"out={budget.get('total_output_tokens', 0):,}",
            manifest["stage_timings"]["insight"],
        )

    except Exception as exc:
        manifest["errored_stage"] = manifest.get("current_stage")
        manifest["error_message"] = f"{type(exc).__name__}: {exc}"
        persistence.save_manifest(run_dir, manifest)
        raise PipelineError(manifest["current_stage"] or "unknown", exc) from exc

    # Per-stage cost from the captured usage snapshots.
    stage_costs: dict[str, float] = {}
    cost_warnings: list[str] = []
    for stage in ("search", "filter", "insight"):
        usage = stage_usage.get(stage) or {}
        cost, warns = _estimate_total_cost(usage)
        stage_costs[stage] = round(cost, 6)
        cost_warnings.extend(warns)

    # Combine usage across stages and estimate total cost.
    combined_usage = _merge_usage(
        dict(shared_llm.per_model),
        insight_result.budget_summary.get("per_model") or {},
    )
    cost_usd, _ = _estimate_total_cost(combined_usage)
    # Dedup warnings (same unknown model can surface in multiple stages).
    cost_warnings = list(dict.fromkeys(cost_warnings))

    manifest["current_stage"] = None
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["combined_usage"] = combined_usage
    manifest["stage_usage"] = stage_usage
    manifest["stage_costs_usd"] = stage_costs
    manifest["estimated_cost_usd"] = round(cost_usd, 6)
    if cost_warnings:
        manifest["cost_estimate_warnings"] = cost_warnings
    persistence.save_manifest(run_dir, manifest)

    print()
    print(f"=== Pipeline complete: {question.id} ===")
    for stage in ("search", "filter", "extract", "insight"):
        elapsed = manifest["stage_timings"].get(stage)
        if elapsed is None:
            continue
        stage_cost = stage_costs.get(stage)
        cost_str = f"  ${stage_cost:.4f}" if stage_cost is not None else ""
        print(f"  {stage:<8} {elapsed:>7.2f}s{cost_str}")
    print(f"  total cost:  ${cost_usd:.4f}")
    if cost_warnings:
        for w in cost_warnings:
            print(f"  ! {w}")
    print(f"  artifacts: {run_dir}")

    return insight_result


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        run_pipeline(args)
    except PipelineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except (FileNotFoundError, KeyError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
