"""Manual smoke test for historical-replay mode.

Runs the search stage against `q1` (resolved H5N1 US, Feb 28 2025 deadline)
with as_of_date = question.created_date. Prints a digest of what the cutoff
machinery did. Does NOT push through filtering/extraction (issue #13 would
make that uninformative without an LLM and a relaxed threshold).

What this validates on the feat/as-of-date-replay branch:
  - Tavily backend receives end_date matching the cutoff
  - All returned SearchResult.published_date <= as_of_date
  - Dashboard URLs are Wayback-rewritten (or suppressed if no snapshot)
  - SearchResult.cutoff_applied is populated
  - The date-recovery chain fires for undated results

What this also gathers (for issue #5 — Tavily date reliability):
  - share of Tavily results that came with a published_date
  - share that needed the recovery chain (and which strategy won)
  - share that were dropped for no-date-available

Run:
  python scripts/test_historical_replay.py

Requires TAVILY_API_KEY and OPENAI_API_KEY in environment (or .env).
"""

from __future__ import annotations

import logging
import os
import sys
from collections import Counter
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from bioscancast.statesfiltering.models import ForecastQuestion
from bioscancast.llm.client import OpenAIClient
from bioscancast.stages.searching.backends.tavily_backend import TavilyBackend
from bioscancast.stages.searching.pipeline import SearchStagePipeline


# q1 from bioscancast_questions.csv. created_date is Excel serial 45705 =
# 2025-02-17. Resolution deadline is Feb 28, 2025. Cutoff for the human
# forecaster is the creation date.
Q1_TEXT = (
    "How many confirmed human cases of H5N1 will be reported in the US "
    "by February 28, 2025, according to the US dashboard?"
)
Q1_CREATED = datetime(2025, 2, 17, tzinfo=timezone.utc)


def _hr(title: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n{title}\n{bar}")


def main() -> None:
    # Surface the cutoff-filter and dashboard suppression INFO logs.
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # Hide curl_cffi/openai noise but keep our modules chatty.
    for noisy in ("urllib3", "httpx", "openai", "tavily"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    question = ForecastQuestion(
        id="q1",
        text=Q1_TEXT,
        created_at=Q1_CREATED,
        pathogen="h5n1",
        region="United States",
        target_date=datetime(2025, 2, 28, tzinfo=timezone.utc),
        as_of_date=Q1_CREATED,  # the cutoff
    )

    _hr("CONFIGURATION")
    print(f"Question text   : {question.text}")
    print(f"Pathogen        : {question.pathogen}")
    print(f"Created at      : {question.created_at.isoformat()}")
    print(f"Target date     : {question.target_date.isoformat()}")
    print(f"AS-OF (cutoff)  : {question.as_of_date.isoformat()}")
    print(f"Historical mode : YES  (as_of_date is set)")

    llm = OpenAIClient()
    # Wrap the Tavily backend so we can observe whether end_date was forwarded.
    base_backend = TavilyBackend()
    end_dates_seen: list = []
    _orig_search = base_backend.search

    def wrapped_search(query: str, max_results: int = 10, end_date=None):
        end_dates_seen.append(end_date)
        return _orig_search(query, max_results=max_results, end_date=end_date)

    base_backend.search = wrapped_search  # type: ignore[assignment]

    # NB: deliberately running without SearchCache so we hit Tavily fresh and
    # the test isn't influenced by entries from a previous (different-cutoff)
    # run. The cache key incorporates the cutoff so this is just paranoia.
    pipeline = SearchStagePipeline(
        search_backend=base_backend,
        llm_client=llm,
        cache=None,
        backend_name="tavily",
        # Leave historical_roleplay off — that's a separate opt-in.
    )

    _hr("RUNNING SEARCH STAGE")
    results = pipeline.run(question)

    _hr("BACKEND OBSERVATIONS")
    print(f"Sub-queries issued to Tavily: {len(end_dates_seen)}")
    distinct_end_dates = set(end_dates_seen)
    print(f"end_date values forwarded   : {distinct_end_dates}")
    if distinct_end_dates == {question.as_of_date.strftime('%Y-%m-%d')}:
        print(">> end_date correctly forwarded on every call.")
    else:
        print(">> WARNING: end_date forwarding looks wrong.")

    _hr("RESULT SUMMARY")
    print(f"Total results returned: {len(results)}")

    dashboards = [r for r in results if r.engine == "dashboard"]
    organic = [r for r in results if r.engine != "dashboard"]
    print(f"  Dashboards : {len(dashboards)}")
    print(f"  Organic    : {len(organic)}")

    # Cutoff sanity check
    leaks = [r for r in results if r.published_date and r.published_date > question.as_of_date]
    print(f"  Post-cutoff leaks: {len(leaks)}")
    if leaks:
        for r in leaks:
            print(f"    LEAK: {r.url}  pub={r.published_date.isoformat()}")
    else:
        print("  >> No post-cutoff results in the final list.")

    # cutoff_applied audit
    bad_cutoff = [r for r in results if r.cutoff_applied != question.as_of_date]
    print(f"  cutoff_applied mismatches: {len(bad_cutoff)}")

    _hr("DASHBOARDS (Wayback rewrite or suppression)")
    if not dashboards:
        print("No dashboards present — Wayback either had no snapshot or "
              "they were suppressed. Check the INFO log lines above.")
    for r in dashboards:
        in_wayback = "web.archive.org/web/" in r.url
        snap_date = r.published_date.isoformat() if r.published_date else "n/a"
        print(f"  [{('WAYBACK' if in_wayback else 'LIVE!')}] {r.url}")
        print(f"    snapshot_date={snap_date}  source={r.published_date_source}")

    _hr("DATA FOR ISSUE #5 (Tavily published_date reliability)")
    source_counter: Counter = Counter(r.published_date_source for r in organic)
    print("Per-result published_date_source distribution (organic only):")
    for src, n in source_counter.most_common():
        label = src if src is not None else "<none>"
        print(f"  {label:25s} {n}")
    n_backend = source_counter.get("backend", 0)
    n_recovered = sum(
        n for src, n in source_counter.items()
        if src in {"url_slug", "last_modified", "wayback_first_seen"}
    )
    n_unsourced = source_counter.get(None, 0)
    if organic:
        print(
            f"\nTavily-supplied date rate: {n_backend}/{len(organic)} = "
            f"{n_backend / len(organic):.0%}"
        )
        print(
            f"Recovery-chain saves     : {n_recovered}/{len(organic)} = "
            f"{n_recovered / len(organic):.0%}"
        )
        print(
            f"Unsourced (kept anyway)  : {n_unsourced}/{len(organic)} = "
            f"{n_unsourced / len(organic):.0%}  "
            "[expected ~0 in historical mode after the filter]"
        )

    _hr("TOP 15 RESULTS (sorted by search_stage_score)")
    for i, r in enumerate(results[:15], 1):
        pub = r.published_date.date().isoformat() if r.published_date else "—"
        src = r.published_date_source or "—"
        print(
            f"{i:2d}. score={r.search_stage_score:.3f}  "
            f"tier={r.source_tier:<13s} pub={pub:<10s} src={src:<20s} "
            f"{r.domain}"
        )
        print(f"    {r.title[:90]}")
        print(f"    {r.url}")


if __name__ == "__main__":
    main()
