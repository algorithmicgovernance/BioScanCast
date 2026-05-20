"""Compare Tavily ``topic="news"`` vs default (``general``) for one historical
query. No Wayback, no LLM — just two Tavily calls and a date distribution
so we can see whether dropping ``topic="news"`` would actually surface more
pre-cutoff results.

Run:
    python scripts/probe_tavily_topic.py
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from tavily import TavilyClient


QUERY = "H5N1 human cases United States 2025"
CUTOFF = date(2025, 2, 17)
MAX_RESULTS = 20


def _bucket(dstr: str | None) -> str:
    if not dstr:
        return "no_date"
    try:
        d = date.fromisoformat(dstr[:10])
    except ValueError:
        return "unparseable"
    if d <= CUTOFF:
        return "pre_cutoff"
    if d.year == 2025:
        return "post_cutoff_2025"
    if d.year == 2026:
        return "post_cutoff_2026"
    return f"post_cutoff_{d.year}"


def _run(client: TavilyClient, *, with_news_topic: bool) -> None:
    kwargs: dict = {
        "query": QUERY,
        "max_results": MAX_RESULTS,
        "include_answer": False,
    }
    if with_news_topic:
        kwargs["topic"] = "news"

    print(f"\n{'=' * 72}")
    print(f"Tavily topic = {'news' if with_news_topic else 'general (default)'}")
    print(f"Query        = {QUERY!r}")
    print(f"Cutoff       = {CUTOFF.isoformat()}")
    print("=" * 72)

    resp = client.search(**kwargs)
    results = resp.get("results", [])
    print(f"Returned {len(results)} results\n")

    buckets: Counter = Counter()
    for i, r in enumerate(results, 1):
        d = r.get("published_date")
        bucket = _bucket(d)
        buckets[bucket] += 1
        date_label = (d or "—")[:10] if d else "—"
        url = r.get("url", "")
        title = (r.get("title") or "")[:80]
        marker = "  " if bucket == "pre_cutoff" else "  "
        if bucket == "pre_cutoff":
            marker = "✓ "
        print(f"{marker}{i:2d}. {date_label:<10}  [{bucket:<20}]  {title}")
        print(f"      {url}")

    print(f"\nBucket totals:")
    for bucket, n in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"  {bucket:<20} {n}")
    pre = buckets.get("pre_cutoff", 0)
    if results:
        print(f"\nPre-cutoff hit rate: {pre}/{len(results)} = {pre / len(results):.0%}")


def main() -> None:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        sys.exit("TAVILY_API_KEY missing")
    client = TavilyClient(api_key=api_key)
    _run(client, with_news_topic=True)
    _run(client, with_news_topic=False)


if __name__ == "__main__":
    main()
