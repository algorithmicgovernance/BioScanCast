"""Diagnose the extractor's handling of the epidemiological count-basis
distinction on a small labeled set.

Runs the real chunk extractor over minimal snippets that share a metric
name/value but differ in count basis (cumulative / incident / active), plus
ambiguity and data-quality probes, and reports per case:

- count_basis / time_window / surveillance_method / data_quality,
- two headline checks plus one observation:
    * DISTINCT   — the canonical 3-way snippets get 3 different count_basis
      values (the flattening the meeting flagged);
    * CAVEAT     — data_quality fires on explicit under-reporting/lag text and
      stays null on clean counts (no over-population);
    * BARE-LEAN  — informational: which basis a "reported N cases" sentence
      with no explicit cue gets. Not pass/fail — live runs show both models
      lean "cumulative" for "as of ... has reported" phrasing (defensible: it
      reads as a running total), so the "unknown" fallback rarely triggers on
      real epidemiological wording. Watch it, don't gate on it.

This is the live counterpart to the deterministic regression test in
bioscancast/tests/test_insight_count_basis.py, and the count-basis analogue of
scripts/eval_extraction_traps.py. Not part of CI.

Examples:
    python scripts/eval_count_basis.py                    # gpt-4o-mini
    python scripts/eval_count_basis.py --models gpt-4o-mini,gpt-4o

Requires OPENAI_API_KEY (loaded from .env). Cost is a fraction of a cent —
one small extraction call per case per model.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover
        pass

from bioscancast.stages.filtering.models import ForecastQuestion  # noqa: E402
from bioscancast.stages.insight.text_extraction.chunk_extractor import (  # noqa: E402
    extract_facts_from_chunk,
)
from bioscancast.llm.openai_client import OpenAILLMClient  # noqa: E402
from bioscancast.llm.pricing import estimate_cost  # noqa: E402
from bioscancast.schemas import Document, DocumentChunk  # noqa: E402


_QUESTION = ForecastQuestion(
    id="eval-count-basis",
    text="How many confirmed cases of mpox will be reported in Country X?",
    created_at=datetime(2026, 7, 3, tzinfo=timezone.utc),
    region="Country X",
    pathogen="mpox",
)

# label -> (text, kind). kind drives the headline checks.
_CASES: dict[str, tuple[str, str]] = {
    "cumulative": (
        "As of 3 July 2026, Country X has reported a cumulative total of 282 "
        "confirmed cases of mpox since the start of the outbreak.",
        "canonical",
    ),
    "incident_week": (
        "During the week of 27 June to 3 July 2026, Country X reported 282 new "
        "confirmed cases of mpox, according to laboratory surveillance.",
        "canonical",
    ),
    "active": (
        "As of 3 July 2026, Country X currently has 282 active confirmed cases "
        "of mpox under isolation.",
        "canonical",
    ),
    "ambiguous_bare": (
        "As of 3 July 2026, Country X has reported 282 confirmed cases of mpox.",
        "ambiguous",
    ),
    "caveat_underreport": (
        "Officials cautioned that the true number of mpox infections in Country "
        "X is likely far higher than the 282 confirmed cases, as testing "
        "capacity remains limited and many mild cases go unreported.",
        "caveat",
    ),
    "clean_no_caveat": (
        "As of 3 July 2026, Country X has reported a cumulative total of 400 "
        "confirmed cases of mpox since the start of the outbreak.",
        "clean",
    ),
}


def _chunk_doc(text: str) -> tuple[DocumentChunk, Document]:
    chunk = DocumentChunk(chunk_id="c0", chunk_index=0, text=text, chunk_type="prose")
    doc = Document(
        id="d0", result_id="r0", source_url="https://example.org/report",
        domain="example.org", fetched_at=datetime(2026, 7, 3, tzinfo=timezone.utc),
        document_type="html", status="success",
        title="Country X mpox situation report",
        published_date=datetime(2026, 7, 3, tzinfo=timezone.utc), chunks=[chunk],
    )
    return chunk, doc


def run_model(model: str, max_tokens: int) -> None:
    client = OpenAILLMClient()
    tin = tout = 0
    basis_by_label: dict[str, str | None] = {}
    dq_by_label: dict[str, str | None] = {}

    print(f"\n=== model: {model} ===")
    for label, (text, _kind) in _CASES.items():
        chunk, doc = _chunk_doc(text)
        records, resp = extract_facts_from_chunk(
            chunk, doc, _QUESTION, client, model=model, max_tokens=max_tokens
        )
        tin += resp.input_tokens
        tout += resp.output_tokens
        # Take the record matching the metric (first case_count with a value).
        rec = next(
            (r for r in records if r.metric_value is not None), records[0] if records else None
        )
        basis_by_label[label] = rec.count_basis if rec else None
        dq_by_label[label] = rec.data_quality if rec else None
        if rec is None:
            print(f"  {label:<20} (no record)")
        else:
            print(
                f"  {label:<20} basis={str(rec.count_basis):<10} "
                f"window={str(rec.time_window):<8} "
                f"surv={rec.surveillance_method or '-'} "
                f"data_quality={rec.data_quality!r}"
            )

    # ---- headline checks ----
    canon = [basis_by_label.get(k) for k in ("cumulative", "incident_week", "active")]
    distinct = len({b for b in canon if b}) == 3
    caveat_fires = bool(dq_by_label.get("caveat_underreport"))
    clean_null = dq_by_label.get("clean_no_caveat") in (None, "")

    def mark(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    print(f"  --- {model} checks ---")
    print(f"    DISTINCT (cumulative/incident/active differ): {mark(distinct)} {canon}")
    print(f"    CAVEAT   (fires on caveat, null on clean):    "
          f"{mark(caveat_fires and clean_null)} "
          f"(caveat={dq_by_label.get('caveat_underreport')!r}, "
          f"clean={dq_by_label.get('clean_no_caveat')!r})")
    print(f"    BARE-LEAN (info, bare count -> ?):            "
          f"{basis_by_label.get('ambiguous_bare')!r}")
    print(f"    est. cost: ${estimate_cost(model, tin, tout):.4f} "
          f"(in={tin} out={tout})")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--models", default="gpt-4o-mini",
                   help="Comma-separated models. Default: gpt-4o-mini")
    p.add_argument("--max-tokens", type=int, default=1024)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set (shell or .env).", file=sys.stderr)
        return 2
    for model in (m.strip() for m in args.models.split(",") if m.strip()):
        run_model(model, args.max_tokens)
    return 0


if __name__ == "__main__":
    sys.exit(main())
