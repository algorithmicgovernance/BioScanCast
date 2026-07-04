"""Live forecasting demo.

Runs the full BioScanCast pipeline (search -> filter -> extract -> insight
-> forecast) for one question and prints a readable report: the evidence
the insight stage extracted, the informed forecast, the no-retrieval
baseline, a sample of the model's reasoning, and the token cost.

Examples:
    python scripts/demo_forecast.py                      # ongoing Ebola outbreak (default)
    python scripts/demo_forecast.py -v                   # show stage logs
    python scripts/demo_forecast.py q1 \\
        --csv bioscancast/stages/evaluation/bioscancast_questions.csv \\
        --as-of-date 2025-02-28                          # historical replay

Requires OPENAI_API_KEY and TAVILY_API_KEY (loaded from .env).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bioscancast import main as orchestrator  # noqa: E402
from bioscancast.stages.evaluation.loaders import load_questions  # noqa: E402

# Windows consoles default to cp1252, which can't encode characters like
# "≥" or en-dashes that turn up in source quotes and model rationales.
# Re-encode stdout/stderr as UTF-8 (replacing anything unmappable) so the
# report never crashes mid-print.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover - non-reconfigurable stream
        pass

# Demo questions shipped with the repo and their answer options. The
# default models the ongoing 2026 Bundibugyo (Ebola) outbreak in the DRC
# and Uganda — a current, unresolved question with strong live sources.
DEMO_CSV = "examples/demo_questions.csv"
DEMO_OPTIONS = {
    "ebola2026": ["Under 450", "450-650", "650-900", "900+"],
}

BAR_WIDTH = 36


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live BioScanCast forecasting demo.")
    p.add_argument("question_id", nargs="?", default="ebola2026")
    p.add_argument("--csv", default=None, help="Question CSV (defaults to the demo CSV).")
    p.add_argument("--options", default=None, help="Comma-separated answer options (override).")
    p.add_argument("--as-of-date", default=None, help="Historical-replay cutoff YYYY-MM-DD.")
    p.add_argument("--no-baseline", action="store_true", help="Skip the retrieval-free baseline.")
    p.add_argument("--max-insights", type=int, default=12, help="How many facts to show.")
    p.add_argument("--out-root", default="data/runs")
    p.add_argument("-v", "--verbose", action="store_true", help="Show per-stage logs.")
    return p.parse_args(argv)


def _bar(p: float) -> str:
    filled = int(round(p * BAR_WIDTH))
    return "|" + "#" * filled + " " * (BAR_WIDTH - filled) + "|"


def _fmt_date(iso: str | None, precision: str | None) -> str:
    if not iso:
        return "undated"
    if precision == "year":
        return iso[:4]
    if precision == "month":
        return iso[:7]
    return iso[:10]


def _print_distribution(probs: dict[str, float], options: list[str]) -> None:
    for opt in options:
        p = float(probs.get(opt, 0.0))
        print(f"  {opt:<12} {_bar(p)} {p:.2f}")


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _resolve_options(args: argparse.Namespace) -> list[str] | None:
    if args.options:
        return [o.strip() for o in args.options.split(",") if o.strip()]
    return DEMO_OPTIONS.get(args.question_id)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    csv = args.csv or DEMO_CSV
    options = _resolve_options(args)
    run_id = "demo-" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Drive the real orchestrator via its own CLI parser so every stage and
    # all persistence behave exactly as in production.
    orch_argv = [
        args.question_id, "--csv", csv, "--out-root", args.out_root,
        "--run-id", run_id,
    ]
    if options:
        orch_argv += ["--options", ",".join(options)]
    if args.as_of_date:
        orch_argv += ["--as-of-date", args.as_of_date]
    if args.no_baseline:
        orch_argv += ["--no-baseline"]
    if args.verbose:
        orch_argv += ["-v"]

    orch_args = orchestrator._parse_args(orch_argv)
    orchestrator.run_pipeline(orch_args)

    run_dir = Path(args.out_root) / args.question_id / run_id
    question = _load_json(run_dir / "question.json")
    insight = _load_json(run_dir / "insight.json")
    forecast = _load_json(run_dir / "forecast.json")
    manifest = _load_json(run_dir / "manifest.json")

    # Resolved answer, if any (current questions are unresolved).
    resolved, status = None, None
    try:
        qdf = load_questions(csv)
        row = qdf[qdf["question_id"].astype(str).str.strip() == args.question_id].iloc[0]
        status = str(row.get("question_status", "")).lower()
        resolved = str(row.get("resolved_option", "")).strip()
    except Exception:
        pass

    opts = forecast["options"]
    dists = {d["forecast_source"]: d["probabilities"] for d in forecast["distributions"]}

    line = "=" * 64
    print("\n" + line)
    print("  BioScanCast - Forecasting demo")
    print(line)
    print(f"Question [{question['id']}]")
    print(f"  {question['text']}")
    mode = f"AS-OF {args.as_of_date}" if args.as_of_date else "LIVE"
    print(
        f"  pathogen={question.get('pathogen')}  region={question.get('region')}  "
        f"target={(question.get('target_date') or '')[:10]}  mode={mode}"
    )
    print(f"  options: {' | '.join(opts)}")
    if status and status == "resolved" and resolved:
        print(f"  resolved answer (dataset): {resolved}")
    else:
        print("  status: unresolved (genuine forward-looking forecast)")

    # --- Evidence ---
    records = insight["records"]
    print(f"\nEvidence (insight stage): {len(records)} facts from "
          f"{insight.get('documents_processed', 0)} documents"
          f" (showing up to {args.max_insights}, most recent first)")

    def _key(r):
        return (r.get("event_date") or "", r.get("confidence") or 0.0)
    for r in sorted(records, key=_key, reverse=True)[: args.max_insights]:
        date = _fmt_date(r.get("event_date"), r.get("event_date_precision"))
        loc = r.get("location") or r.get("iso_country_code") or ""
        if r.get("metric_value") is not None:
            mv = r["metric_value"]
            mv = int(mv) if float(mv).is_integer() else mv
            metric = f"{r.get('metric_name') or 'value'}={mv} {r.get('metric_unit') or ''}".strip()
        else:
            metric = (r.get("summary") or "").strip()
        print(f"  [{date}] {r.get('pathogen') or ''} {loc}  {metric}  (conf {r.get('confidence', 0):.2f})")
        srcs = r.get("sources") or []
        if srcs:
            print(f"        \"{(srcs[0].get('quote') or '').strip()}\"")
            print(f"        source: {srcs[0].get('source_url')}")

    # --- Informed forecast ---
    primary_src = next((d["forecast_source"] for d in forecast["distributions"]
                        if d["forecast_source"] != "bioscancast_baseline"), None)
    if primary_src and primary_src in dists:
        print(f"\nInformed forecast  (source: {primary_src})")
        _print_distribution(dists[primary_src], opts)
        top = max(opts, key=lambda o: dists[primary_src].get(o, 0.0))
        print(f"  most likely: {top}  (p={dists[primary_src].get(top, 0.0):.2f})")

    # --- Baseline + shift ---
    if "bioscancast_baseline" in dists:
        base = dists["bioscancast_baseline"]
        print("\nNo-retrieval baseline  (source: bioscancast_baseline)")
        _print_distribution(base, opts)
        if primary_src in dists:
            print("  shift from evidence (informed - baseline):")
            for opt in opts:
                delta = dists[primary_src].get(opt, 0.0) - base.get(opt, 0.0)
                print(f"    {opt:<12} {delta:+.2f}")
        if forecast.get("baseline_rationale"):
            print(f"  baseline reasoning: {forecast['baseline_rationale']}")

    # --- Sample reasoning ---
    samples = [s for s in forecast.get("samples", []) if s.get("ok")]
    if samples:
        s = samples[0]
        print(f"\nSample reasoning (1 of {len(samples)} usable, model={s.get('model')})")
        if s.get("reference_class"):
            print(f"  reference class : {s['reference_class']}")
        if s.get("base_rate") is not None:
            print(f"  base rate       : {s['base_rate']}")
        for label, key in (("drivers up", "drivers_up"), ("drivers down", "drivers_down")):
            for d in (s.get(key) or [])[:3]:
                print(f"  {label:<15}: {d}")
        if s.get("why_might_be_wrong"):
            print(f"  why might be wrong: {s['why_might_be_wrong']}")
        if s.get("rationale"):
            print(f"  rationale       : {s['rationale']}")

    if forecast.get("notes"):
        print("\nNotes:")
        for n in forecast["notes"]:
            print(f"  - {n}")

    # --- Cost ---
    print("\nCost & tokens")
    timings = manifest.get("stage_timings", {})
    costs = manifest.get("stage_costs_usd", {})
    for stage in ("search", "filter", "extract", "insight", "forecast"):
        t = timings.get(stage)
        if t is None:
            continue
        c = costs.get(stage)
        c_str = f"${c:.4f}" if c is not None else ""
        print(f"  {stage:<9} {t:>7.1f}s  {c_str}")
    print(f"  TOTAL estimated cost: ${manifest.get('estimated_cost_usd', 0.0):.4f}")
    print(f"  artifacts: {run_dir}")
    print(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
