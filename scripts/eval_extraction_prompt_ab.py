"""A/B-test a candidate extraction prompt against the production prompt.

The trap-set diagnosis showed gpt-4o-mini mis-files projections and
wrong-scope numbers as observed counts at high confidence. This script
tests the cheapest possible fix:
*better prompting alone*. It runs the trap-set on gpt-4o-mini with the
current prompt vs a candidate prompt (observed-vs-projected, scope-match,
and calibrated-confidence rules), averaged over N runs since extraction
temperature is non-zero.

If the candidate rescues the cheap model, the win is nearly free and no
strong-model pass is needed (issue #26). If not, that is the hard evidence
for a strong pass / model upgrade. Nothing in production changes — the
candidate prompt is monkeypatched in for the experiment only.

    python scripts/eval_extraction_prompt_ab.py                # gpt-4o-mini, 3 runs
    python scripts/eval_extraction_prompt_ab.py --runs 5 --model gpt-4o-mini

Requires OPENAI_API_KEY (loaded from .env). ~12*N cheap calls.
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

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover
        pass

import bioscancast.stages.insight.text_extraction.prompts as prompts_mod  # noqa: E402
from bioscancast.stages.insight.text_extraction.chunk_extractor import (  # noqa: E402
    extract_facts_from_chunk,
)
from bioscancast.llm.openai_client import OpenAILLMClient  # noqa: E402
from bioscancast.tests.fixtures.insight.extraction_traps import (  # noqa: E402
    ALL_TRAPS,
    aggregate,
    score_extraction,
)


# Candidate prompt: the production EXTRACTION_SYSTEM_PROMPT plus three rules
# targeting the diagnosed failures. Kept verbatim-compatible with the rest
# of the extractor (same quote rule, canonical metric_names, schema).
CANDIDATE_SYSTEM_PROMPT = """\
You are a biosecurity fact extractor.  Your job is to extract \
structured factual claims from a document chunk that are relevant \
to a specific forecast question.

RULES:
1. Extract ONLY facts that are directly stated in or clearly supported \
by the chunk text.  Do NOT infer, speculate, or use outside knowledge.
2. For each fact, provide a verbatim quote from the chunk (max 200 \
characters) that supports the claim.  The quote must be an exact \
substring of the chunk text and must contain the metric_value (as \
digits, a number-word, or a clear relative reference). A contextual \
sentence that mentions the topic but not the figure is NOT acceptable.
3. If the chunk contains no relevant facts, return an empty facts list.
4. Do NOT answer the forecast question.  Your job is fact extraction.
5. For event_date, use the most specific ISO date you can extract \
(``YYYY-MM-DD`` / ``YYYY-MM`` / ``YYYY``); do not invent a day.
6. For metric_name, prefer canonical snake_case values when applicable: \
confirmed_cases, suspected_cases, confirmed_or_probable_cases, deaths, \
suspected_deaths, hospitalizations, recoveries, vaccinations_administered, \
vaccine_doses_distributed, affected_herds, affected_animals, \
new_outbreaks_declared, reproductive_number, case_fatality_ratio. Map \
"cases"/"reported cases"/"total cases" -> confirmed_cases; "suspected/\
probable/possible cases" -> suspected_cases; "deaths" -> deaths.

7. OBSERVED vs NON-OBSERVED.  event_type ``case_count`` and ``death_count`` \
are for OBSERVED, actually-reported counts ONLY.  NEVER assign them to a \
number that is projected, modeled, forecasted, simulated, estimated, \
hypothetical, or a target/threshold — even when the number is stated \
plainly (e.g. "could see as many as 10,000 cases by year-end", "one in 20 \
simulations projected ..."). For such a number, set event_type ``other``, \
leave metric_value null, and describe it in ``summary`` (say it is a \
projection/model output and give its horizon and scenario if stated).

8. SCOPE MATCH.  A number only pertains to the question if its metric, its \
geography, AND its time window all match the question's.  Do NOT relabel a \
number to the question's scope:
   - If the geography differs (e.g. a GLOBAL cumulative total against a \
United-States question), set ``location`` to the number's true scope \
(e.g. "Global") — never the question's region — and note the period (e.g. \
"cumulative since 2003") in ``summary``.
   - A sub-national figure (one state/district) keeps its sub-national \
``location``; it is NOT the national total.
   - A weekly / "new this week" increment must say so in ``summary`` or \
``metric_name``; it is NOT the cumulative total.

9. CONFIDENCE = SCOPE-MATCH, not textual presence.  Set ``confidence`` by \
how well the fact's metric/geography/time-window match the question, NOT by \
whether the number appears in the text.  A number present verbatim but of \
uncertain or mismatched scope gets LOW confidence (<= 0.5).  Reserve high \
confidence (> 0.85) for numbers whose metric, geography, and window clearly \
match the question.

10. Be aware of anchoring, availability, and overconfidence biases; when \
the chunk is ambiguous, lower your confidence.

OUTPUT: Return a JSON object with a "facts" array.  Each fact has the \
fields defined in the schema.  Return {"facts": []} if no relevant facts \
are found."""


def _run_once(model: str) -> dict:
    client = OpenAILLMClient()
    results = []
    for trap in ALL_TRAPS:
        records, _ = extract_facts_from_chunk(
            trap.chunk, trap.document, trap.question, client, model=model,
        )
        results.append(score_extraction(records, trap))
    return aggregate(results)


def _avg(dicts: list[dict], keys: list[str]) -> dict:
    return {k: sum(d[k] for d in dicts) / len(dicts) for k in keys}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--runs", type=int, default=3)
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set (shell or .env).", file=sys.stderr)
        return 2

    keys = [
        "false_observed_count", "mis_scoped_to_region", "qualifier_dropped",
        "mean_trap_confidence", "control_recall", "traps_clean",
    ]
    original = prompts_mod.EXTRACTION_SYSTEM_PROMPT

    print(f"Model: {args.model}   runs: {args.runs}\n")
    print("Running BASELINE (production prompt)...")
    baseline = _avg([_run_once(args.model) for _ in range(args.runs)], keys)

    print("Running CANDIDATE (augmented prompt)...")
    prompts_mod.EXTRACTION_SYSTEM_PROMPT = CANDIDATE_SYSTEM_PROMPT
    try:
        candidate = _avg([_run_once(args.model) for _ in range(args.runs)], keys)
    finally:
        prompts_mod.EXTRACTION_SYSTEM_PROMPT = original

    print(f"\n=== A/B over {args.runs} runs (means; lower mishandling better) ===")
    print(f"  {'metric':<22}{'baseline':>12}{'candidate':>12}")
    for k in keys:
        print(f"  {k:<22}{baseline[k]:>12.2f}{candidate[k]:>12.2f}")
    print(
        "\n  (traps_clean is out of "
        f"{len(ALL_TRAPS)}; control_recall and confidence are 0..1)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
