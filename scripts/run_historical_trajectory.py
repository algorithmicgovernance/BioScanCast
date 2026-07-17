"""Multi-timepoint historical-replay trajectory driver.

For one benchmark question, runs the FULL pipeline independently at several
evenly-spaced as-of cutoffs between the question's open date (``created_at``)
and its resolution date (``target_date``), then collects every run's forecast
rows into a single eval-compatible CSV so the model's *trajectory* can be
scored and compared against the human group and the resolved outcome.

Design (locked): clean re-derivation + fresh forecast at each cutoff. Each
timepoint is an independent pipeline run; no insights and no prior forecast are
carried forward. This keeps every timepoint an independent measurement, so a
change in the forecast is attributable to new evidence rather than to
accumulation or anchoring.

The driver is idempotent: a cutoff whose run directory already has a
``forecast.json`` is reused (not re-run) unless ``--force`` is given, so an
interrupted trajectory can be resumed cheaply.

Usage:
    python scripts/run_historical_trajectory.py q1 \\
        --options "70-100,100-150,150-200,200+" \\
        --n-forecasts 5 -v

Requires OPENAI_API_KEY and TAVILY_API_KEY (loaded from .env by the
orchestrator).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd  # noqa: E402

from bioscancast import main as orchestrator  # noqa: E402
from bioscancast.stages.evaluation.loaders import (  # noqa: E402
    load_options_for_question,
    load_question_by_id,
)

# Windows consoles default to cp1252, which can't encode characters that turn
# up in source quotes and rationales. Re-encode stdout/stderr as UTF-8 so the
# trajectory summary never crashes mid-print.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover
        pass

DEFAULT_CSV = orchestrator.DEFAULT_CSV
DEFAULT_FORECASTS_CSV = orchestrator.DEFAULT_FORECASTS_CSV
DEFAULT_OUT_ROOT = orchestrator.DEFAULT_OUT_ROOT


def build_schedule(
    created_at: datetime, target_date: datetime, n: int
) -> List[datetime]:
    """Return ``n`` evenly-spaced as-of cutoffs over ``[created_at, target_date)``.

    The first cutoff is the open date and the last is strictly before
    resolution: cutoff ``i`` = ``created_at + i*(target-created)/n`` for
    ``i in 0..n-1``. Each is floored to midnight (a day-granularity cutoff);
    if a short window collapses two cutoffs onto the same day, the duplicate
    is dropped, so a trajectory may have fewer than ``n`` points.
    """
    if n < 1:
        raise ValueError(f"n-forecasts must be >= 1, got {n}")
    if target_date <= created_at:
        raise ValueError(
            f"target_date ({target_date.date()}) must be after created_at "
            f"({created_at.date()}); cannot build a replay schedule."
        )
    span = target_date - created_at
    seen: set = set()
    out: List[datetime] = []
    for i in range(n):
        cutoff = created_at + span * (i / n)
        cutoff = cutoff.replace(hour=0, minute=0, second=0, microsecond=0)
        if cutoff.date() not in seen:
            seen.add(cutoff.date())
            out.append(cutoff)
    return out


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a multi-timepoint historical-replay forecast trajectory."
    )
    p.add_argument("question_id", help="Question ID in the CSV (e.g. q1).")
    p.add_argument("--csv", default=DEFAULT_CSV, help=f"Question CSV. Default: {DEFAULT_CSV}")
    p.add_argument(
        "--forecasts-csv",
        default=DEFAULT_FORECASTS_CSV,
        help="Forecasts CSV used to resolve options when --options is omitted.",
    )
    p.add_argument(
        "--options",
        default=None,
        help="Comma-separated answer options (e.g. '70-100,100-150,150-200,200+'). "
        "If omitted, options are read from the forecasts CSV, falling back to YES/NO.",
    )
    p.add_argument(
        "--n-forecasts", type=int, default=5,
        help="Number of evenly-spaced cutoffs from open date to (just before) resolution.",
    )
    p.add_argument(
        "--target-date", default=None,
        help="Override the CSV-derived resolution date (YYYY-MM-DD). Required when "
        "the question text has no parseable 'by <Month> <day>, <year>' date.",
    )
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="Run-artifact root.")
    p.add_argument(
        "--out-csv", default=None,
        help="Where to write the collected model forecasts CSV. "
        "Default: data/investigations/{qid}_trajectory_forecasts.csv",
    )
    p.add_argument("--no-baseline", action="store_true", help="Skip the retrieval-free baseline.")
    p.add_argument("--no-cache", action="store_true", help="Disable the search-stage cache.")
    p.add_argument(
        "--history-context",
        action="store_true",
        help="Enable prior-run forecast/insight context during forecasting (off by default).",
    )
    p.add_argument(
        "--max-input-tokens", type=int, default=None,
        help="Override InsightConfig.max_input_tokens_per_run.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-run cutoffs even if a forecast.json already exists.",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Show per-stage logs.")
    return p.parse_args(argv)


def _resolve_options(args: argparse.Namespace) -> List[str]:
    if args.options:
        opts = [o.strip() for o in args.options.split(",") if o.strip()]
        if opts:
            return opts
    forecasts_csv = Path(args.forecasts_csv)
    if forecasts_csv.exists():
        try:
            opts = load_options_for_question(forecasts_csv, args.question_id)
        except Exception:  # noqa: BLE001
            opts = []
        if opts:
            return opts
    return ["YES", "NO"]


def _run_one_cutoff(
    args: argparse.Namespace,
    cutoff: datetime,
    target_date: datetime,
    run_dir: Path,
) -> None:
    """Drive the orchestrator for a single cutoff via its own CLI parser, so
    every stage and all persistence behave exactly as in production.

    Options are forwarded only when explicitly given on the driver CLI;
    otherwise the orchestrator resolves them from the forecasts CSV. This
    matters because some option labels contain commas (e.g. mpox case bands
    "124,831-126,000"), which the comma-delimited ``--options`` flag would
    mis-split — CSV resolution reads them intact.
    """
    orch_argv = [
        args.question_id,
        "--csv", args.csv,
        "--out-root", args.out_root,
        "--run-id", run_dir.name,
        "--as-of-date", cutoff.strftime("%Y-%m-%d"),
        "--target-date", target_date.strftime("%Y-%m-%d"),
        "--forecasts-csv", args.forecasts_csv,
    ]
    if args.options:
        orch_argv += ["--options", args.options]
    if args.no_baseline:
        orch_argv.append("--no-baseline")
    if args.no_cache:
        orch_argv.append("--no-cache")
    if args.history_context:
        orch_argv.append("--history-context")
    if args.max_input_tokens is not None:
        orch_argv += ["--max-input-tokens", str(args.max_input_tokens)]
    if args.verbose:
        orch_argv.append("-v")
    orchestrator.run_pipeline(orchestrator._parse_args(orch_argv))


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    question = load_question_by_id(args.csv, args.question_id)
    created_at = question.created_at

    if args.target_date:
        target_date = datetime.strptime(args.target_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
    elif question.target_date is not None:
        target_date = question.target_date
    else:
        print(
            f"ERROR: no resolution date for {args.question_id}. The question text "
            f"has no parseable 'by <Month> <day>, <year>' date; pass --target-date.",
            file=sys.stderr,
        )
        return 2

    schedule = build_schedule(created_at, target_date, args.n_forecasts)
    options = _resolve_options(args)  # for the summary banner only
    out_csv = Path(
        args.out_csv
        or f"data/investigations/{args.question_id}_trajectory_forecasts.csv"
    )

    print(f"=== Historical-replay trajectory: {args.question_id} ===")
    print(f"question : {question.text}")
    print(f"open date: {created_at.date()}   resolution: {target_date.date()}")
    print(f"options  : {options}")
    print(f"cutoffs  : {[c.date().isoformat() for c in schedule]}")
    print(f"out-csv  : {out_csv}\n")

    rows: List[dict] = []
    summary: List[str] = []
    for i, cutoff in enumerate(schedule, 1):
        run_dir = Path(args.out_root) / args.question_id / f"traj_{cutoff:%Y%m%d}"
        forecast_path = run_dir / "forecast.json"
        cutoff_str = cutoff.strftime("%Y-%m-%d")

        if forecast_path.exists() and not args.force:
            print(f"[{i}/{len(schedule)}] {cutoff_str}: reusing existing run {run_dir}")
        else:
            print(f"[{i}/{len(schedule)}] {cutoff_str}: running pipeline -> {run_dir}")
            _run_one_cutoff(args, cutoff, target_date, run_dir)

        if not forecast_path.exists():
            print(f"  ! no forecast.json produced for {cutoff_str}; skipping", file=sys.stderr)
            continue

        forecast = json.loads(forecast_path.read_text(encoding="utf-8"))
        for rec in forecast.get("records", []):
            rows.append(
                {
                    "question_id": rec["question_id"],
                    # Encode source + timepoint so the scorer treats each
                    # (source, cutoff) as its own distribution to score.
                    "forecast_version": f"{rec['forecast_source']}@{cutoff_str}",
                    "option": rec["option"],
                    "probability": float(rec["probability"]),
                }
            )
        # Per-cutoff top option for the console trajectory summary.
        for dist in forecast.get("distributions", []):
            if dist.get("forecast_source") == "bioscancast":
                probs = dist.get("probabilities", {})
                if probs:
                    top = max(probs, key=probs.get)
                    summary.append(f"  {cutoff_str}  ->  {top}  (p={probs[top]:.3f})")

    if not rows:
        print("No forecast rows collected; nothing written.", file=sys.stderr)
        return 1

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["question_id", "forecast_version", "option", "probability"])
    # Match the loader format: ';'-separated, cp1252, comma decimals. Write
    # fixed-point (not scientific) so the comma-decimal reader parses cleanly.
    df["probability"] = df["probability"].map(lambda p: f"{float(p):.12f}".replace(".", ","))
    df.to_csv(out_csv, sep=";", encoding="cp1252", index=False)

    print(f"\n=== Trajectory complete: {args.question_id} ===")
    print(f"informed (bioscancast) most-likely option by cutoff:")
    for line in summary:
        print(line)
    print(f"\nwrote {len(rows)} rows across {len(schedule)} cutoff(s) -> {out_csv}")
    print(
        "Score with:\n"
        f"  python -m bioscancast.stages.evaluation.main "
        f"--forecasts {out_csv} --questions {args.csv}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
