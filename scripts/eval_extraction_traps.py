"""Diagnose the insight stage's "lurch" failure on a labeled trap-set.

Runs the real chunk extractor over the extraction trap-set
(``bioscancast/tests/fixtures/insight/extraction_traps.py``) for one or
more models and prints, per model: how each trap was handled, the records
it produced, and headline metrics (false-observed rate, mean confidence on
trap values, control recall). Running the cheap model against the strong
model is the A2 diagnosis *and* the cheap-vs-strong evidence issue #26
asks for.

Examples:
    python scripts/eval_extraction_traps.py                 # gpt-4o-mini vs gpt-4o
    python scripts/eval_extraction_traps.py --models gpt-4o-mini
    python scripts/eval_extraction_traps.py --show-records  # print every record

Requires OPENAI_API_KEY (loaded from .env). Cost is a few cents — one
chunk-extraction call per trap per model.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

# Windows consoles default to cp1252, which can't encode en-dashes / "≥"
# that appear in source quotes. Re-encode stdout/stderr as UTF-8 so the
# report never crashes mid-print (same guard as scripts/demo_forecast.py).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover
        pass

from bioscancast.stages.insight.text_extraction.chunk_extractor import (  # noqa: E402
    extract_facts_from_chunk,
)
from bioscancast.llm.openai_client import OpenAILLMClient  # noqa: E402
from bioscancast.tests.fixtures.insight.extraction_traps import (  # noqa: E402
    ALL_TRAPS,
    aggregate,
    score_extraction,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--models",
        default="gpt-4o-mini,gpt-4o",
        help="Comma-separated models to compare. Default: gpt-4o-mini,gpt-4o",
    )
    p.add_argument(
        "--max-tokens", type=int, default=4096,
        help="Per-call output-token cap (matches InsightConfig).",
    )
    p.add_argument(
        "--show-records", action="store_true",
        help="Print every extracted record (auditing why a trap tripped).",
    )
    return p.parse_args(argv)


def _fmt_record(r) -> str:
    basis = getattr(r, "value_basis", None)
    basis_str = f" basis={basis}" if basis is not None else ""
    value = "" if r.metric_value is None else f"{r.metric_value:g}"
    return (
        f"      [{r.event_type}] {r.metric_name or '-'}={value} "
        f"conf={r.confidence:.2f} loc={r.location or r.iso_country_code or '-'}"
        f"{basis_str}"
    )


def run_model(model: str, max_tokens: int, show_records: bool) -> dict:
    client = OpenAILLMClient()
    results = []
    print(f"\n=== model: {model} ===")
    for trap in ALL_TRAPS:
        records, _ = extract_facts_from_chunk(
            trap.chunk, trap.document, trap.question, client,
            model=model, max_tokens=max_tokens,
        )
        result = score_extraction(records, trap)
        results.append(result)
        flags = (
            ", ".join(
                f"{m.value:g}:{'/'.join(m.reasons)}" for m in result.mishandled
            )
            or "clean"
        )
        status = "OK " if result.ok else "!! "
        print(
            f"  {status}{trap.label:<28} mishandled=[{flags}] "
            f"controls={result.controls_recalled}/{result.controls_total}"
        )
        if show_records:
            for r in records:
                print(_fmt_record(r))

    agg = aggregate(results)
    print(f"  --- {model} aggregate ---")
    print(
        f"    false_observed={agg['false_observed_count']}  "
        f"mis_scoped={agg['mis_scoped_to_region']}  "
        f"qualifier_dropped={agg['qualifier_dropped']}  "
        f"mean_trap_conf={agg['mean_trap_confidence']:.2f}  "
        f"control_recall={agg['control_recall']:.2f}  "
        f"clean={agg['traps_clean']}/{agg['traps']}"
    )
    return agg


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set (shell or .env).", file=sys.stderr)
        return 2

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    summary: dict[str, dict] = {}
    for model in models:
        summary[model] = run_model(model, args.max_tokens, args.show_records)

    if len(models) > 1:
        print("\n=== comparison (lower mishandling is better) ===")
        header = f"  {'metric':<22}" + "".join(f"{m:>16}" for m in models)
        print(header)
        for key in (
            "false_observed_count", "mis_scoped_to_region", "qualifier_dropped",
            "mean_trap_confidence", "control_recall", "traps_clean",
        ):
            row = f"  {key:<22}" + "".join(
                f"{summary[m][key]:>16.2f}" if isinstance(summary[m][key], float)
                else f"{summary[m][key]:>16}"
                for m in models
            )
            print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
