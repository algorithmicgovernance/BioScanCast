"""Probe Tavily configurations across the BioScanCast resolved corpus.

Originally a single-query script comparing topic="news" vs topic="general" on
q1 (H5N1 US, cutoff Feb 17 2025). Now generalized to iterate the corpus and
explore Tavily knobs (search_depth, include_domains, exact_match, etc.) under
the historical-replay cutoff machinery.

Investigation context: see ``specs/tavily-historical-coverage.md`` and the
plan at ``~/.claude/plans/i-d-like-you-to-wondrous-whale.md``.

Each (question x config) result is dumped to ``specs/probe-results/`` as JSON
so the analyzer (``analyze_tavily_probe.py``) can re-compute hit rates and
date-recovery coverage offline without re-paying the Tavily quota.

Examples:
    # All resolved questions, news topic, default settings
    python scripts/probe_tavily_topic.py --question-id all --topic news

    # Single question, advanced search_depth
    python scripts/probe_tavily_topic.py --question-id q1 --topic news \
        --knobs '{"search_depth": "advanced"}'

    # Synthetic backdated query (override question text + cutoff)
    python scripts/probe_tavily_topic.py --synthetic-query "MERS-CoV cases Saudi Arabia 2015" \
        --synthetic-cutoff 2017-01-01 --synthetic-tag mers2015 --topic news

    # Original q1/news+general behavior (legacy)
    python scripts/probe_tavily_topic.py --legacy
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from tavily import TavilyClient


REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_CSV = REPO_ROOT / "bioscancast" / "stages" / "eval_stage" / "bioscancast_questions.csv"
RESULTS_DIR = REPO_ROOT / "specs" / "probe-results"


def excel_serial_to_date(serial: int | str) -> date:
    """Excel epoch is 1899-12-30 (Lotus 1-2-3 leap-year bug correction)."""
    return (datetime(1899, 12, 30) + timedelta(days=int(serial))).date()


def load_resolved_questions() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with CORPUS_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            if row.get("question_status") == "resolved":
                row["cutoff_date"] = excel_serial_to_date(row["created_date"])
                out.append(row)
    return out


def get_question(qid: str) -> dict[str, Any]:
    for q in load_resolved_questions():
        if q["question_id"] == qid:
            return q
    raise SystemExit(f"Unknown or unresolved question_id: {qid}")


def _bucket(dstr: str | None, cutoff: date) -> str:
    if not dstr:
        return "no_date"
    try:
        d = date.fromisoformat(dstr[:10])
    except ValueError:
        try:
            from email.utils import parsedate_to_datetime
            d = parsedate_to_datetime(dstr).date()
        except (ValueError, TypeError):
            return "unparseable"
    if d <= cutoff:
        return "pre_cutoff"
    return f"post_cutoff_{d.year}"


def config_hash(query: str, cutoff: date, knobs: dict[str, Any]) -> str:
    payload = json.dumps({"query": query, "cutoff": cutoff.isoformat(), "knobs": knobs}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


def cache_path(tag: str, knobs: dict[str, Any]) -> Path:
    """Filename: <tag>__<config_hash>.json. Tag is question_id or synthetic-tag."""
    knob_summary = "_".join(f"{k}={v}" for k, v in sorted(knobs.items()) if k != "include_domains")
    if "include_domains" in knobs:
        knob_summary += "_domains=" + str(len(knobs["include_domains"]))
    knob_summary = knob_summary.replace("/", "_").replace(":", "_")[:60] or "default"
    h = hashlib.sha1(json.dumps(knobs, sort_keys=True).encode()).hexdigest()[:8]
    return RESULTS_DIR / f"{tag}__{knob_summary}__{h}.json"


def run_probe(
    client: TavilyClient,
    *,
    tag: str,
    query: str,
    cutoff: date,
    knobs: dict[str, Any],
    force: bool = False,
) -> dict[str, Any]:
    """Run one Tavily call (cached). Returns the cached payload."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(tag, knobs)
    if path.exists() and not force:
        with path.open(encoding="utf-8") as f:
            return json.load(f)

    kwargs: dict[str, Any] = {"query": query, "include_answer": False, **knobs}
    if "max_results" not in kwargs:
        kwargs["max_results"] = 20
    resp = client.search(**kwargs)

    payload = {
        "tag": tag,
        "query": query,
        "cutoff": cutoff.isoformat(),
        "knobs": knobs,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "response": resp,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def summarize(payload: dict[str, Any]) -> None:
    cutoff = date.fromisoformat(payload["cutoff"])
    results = payload["response"].get("results", []) or []
    buckets: Counter = Counter()
    dated = 0
    for r in results:
        d = r.get("published_date")
        if d:
            dated += 1
        buckets[_bucket(d, cutoff)] += 1
    pre = buckets.get("pre_cutoff", 0)
    knob_str = ", ".join(f"{k}={v}" for k, v in sorted(payload["knobs"].items()))[:80] or "(default)"
    n = len(results) or 1
    print(
        f"  {payload['tag']:>10}  cutoff={cutoff}  {knob_str:<82}  "
        f"-> pre={pre}/{len(results)} ({pre / n:.0%})  dated={dated}/{len(results)}"
    )


def add_year_hint(query: str, cutoff: date) -> str:
    """Mirror the pipeline's year-hint suffix so probes match pipeline behavior."""
    y = str(cutoff.year)
    if y in query:
        return query
    return f"{query} {y}"


def build_query_from_question(q: dict[str, Any], hint_year: bool = True) -> str:
    """Construct a search query from a corpus question. Strip framing words
    ("How many ... will be reported ... according to ...") to expose the
    topical noun phrase. Keep the topic prefix as a hint."""
    text = q["question_text"]
    base = f"{q['topic']} {text}"
    return add_year_hint(base, q["cutoff_date"]) if hint_year else base


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--question-id", help="Resolved question id (q1, q3, q7, q9) or 'all'.")
    p.add_argument("--topic", choices=["news", "general", "finance"], default="news")
    p.add_argument("--knobs", default="{}", help="JSON object of extra Tavily kwargs.")
    p.add_argument("--synthetic-query", help="Use a free-form query instead of corpus.")
    p.add_argument("--synthetic-cutoff", help="YYYY-MM-DD cutoff for synthetic query.")
    p.add_argument("--synthetic-tag", help="Short tag for cache filename (synthetic only).")
    p.add_argument("--no-year-hint", action="store_true", help="Skip the year-suffix hint.")
    p.add_argument("--force", action="store_true", help="Bypass cache and re-call Tavily.")
    p.add_argument("--legacy", action="store_true", help="Replicate original q1 news+general behavior.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        sys.exit("TAVILY_API_KEY missing")
    client = TavilyClient(api_key=api_key)

    if args.legacy:
        q = get_question("q1")
        query = build_query_from_question(q, hint_year=False)
        cutoff = q["cutoff_date"]
        for topic in ("news", "general"):
            payload = run_probe(
                client, tag="q1_legacy", query=query, cutoff=cutoff,
                knobs={"topic": topic, "max_results": 20}, force=args.force,
            )
            summarize(payload)
        return

    knobs = json.loads(args.knobs)
    knobs.setdefault("topic", args.topic)
    knobs.setdefault("max_results", 20)

    if args.synthetic_query:
        if not args.synthetic_cutoff:
            sys.exit("--synthetic-cutoff required with --synthetic-query")
        tag = args.synthetic_tag or "synth"
        cutoff = date.fromisoformat(args.synthetic_cutoff)
        query = args.synthetic_query if args.no_year_hint else add_year_hint(args.synthetic_query, cutoff)
        payload = run_probe(client, tag=tag, query=query, cutoff=cutoff, knobs=knobs, force=args.force)
        summarize(payload)
        return

    if not args.question_id:
        sys.exit("provide --question-id, --synthetic-query, or --legacy")

    qids = ["q1", "q3", "q7", "q9"] if args.question_id == "all" else [args.question_id]
    for qid in qids:
        q = get_question(qid)
        query = build_query_from_question(q, hint_year=not args.no_year_hint)
        payload = run_probe(
            client, tag=qid, query=query, cutoff=q["cutoff_date"], knobs=knobs, force=args.force,
        )
        summarize(payload)


if __name__ == "__main__":
    main()
