"""Consolidate Tavily probe-results JSON dumps into a hit-rate table.

Reads every ``specs/probe-results/*.json`` produced by ``probe_tavily_topic.py``,
applies the production cutoff filter + URL-slug date recovery, and prints a
markdown table suitable for pasting into the findings doc.

Also computes a "hybrid" row per question_id: union of news + general results
under matching knobs, deduped by URL.

No network calls. Safe to re-run any time.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bioscancast.stages.search_stage.date_recovery import date_from_url_slug

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "specs" / "probe-results"


def parse_published_date(dstr: str | None) -> date | None:
    if not dstr:
        return None
    try:
        return date.fromisoformat(dstr[:10])
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(dstr).date()
    except (ValueError, TypeError):
        return None


def classify_one(result: dict[str, Any], cutoff: date) -> dict[str, Any]:
    url = result.get("url", "")
    raw = result.get("published_date")
    pd = parse_published_date(raw)
    slug = date_from_url_slug(url)
    slug_d = slug.date() if slug else None
    # "effective" date: prefer native published_date, fall back to slug.
    effective = pd or slug_d
    return {
        "url": url,
        "title": result.get("title", ""),
        "raw_published_date": raw,
        "parsed_published_date": pd.isoformat() if pd else None,
        "slug_date": slug_d.isoformat() if slug_d else None,
        "effective_date": effective.isoformat() if effective else None,
        "native_pre_cutoff": pd is not None and pd <= cutoff,
        "effective_pre_cutoff": effective is not None and effective <= cutoff,
        "native_dated": pd is not None,
        "effective_dated": effective is not None,
    }


def analyze_payload(payload: dict[str, Any]) -> dict[str, Any]:
    cutoff = date.fromisoformat(payload["cutoff"])
    results = payload["response"].get("results", []) or []
    classified = [classify_one(r, cutoff) for r in results]
    n = len(classified) or 1
    return {
        "tag": payload["tag"],
        "query": payload["query"],
        "cutoff": payload["cutoff"],
        "knobs": payload["knobs"],
        "n_results": len(classified),
        "native_pre_cutoff": sum(1 for c in classified if c["native_pre_cutoff"]),
        "native_dated": sum(1 for c in classified if c["native_dated"]),
        "effective_pre_cutoff": sum(1 for c in classified if c["effective_pre_cutoff"]),
        "effective_dated": sum(1 for c in classified if c["effective_dated"]),
        "results": classified,
        "fetched_at": payload.get("fetched_at"),
    }


def knob_summary(knobs: dict[str, Any]) -> str:
    """Compact human-readable summary of the non-default knobs."""
    parts = []
    topic = knobs.get("topic", "news")
    parts.append(topic)
    for k, v in sorted(knobs.items()):
        if k in {"topic", "max_results", "include_answer"}:
            continue
        if k == "include_domains":
            parts.append(f"domains={len(v)}")
        else:
            parts.append(f"{k}={v}")
    return " ".join(parts)


def load_all() -> list[dict[str, Any]]:
    out = []
    if not RESULTS_DIR.exists():
        return out
    for path in sorted(RESULTS_DIR.glob("*.json")):
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
        out.append(analyze_payload(payload))
    return out


def emit_table(rows: Iterable[dict[str, Any]]) -> str:
    rows = list(rows)
    lines = []
    header = "| tag | config | n | native pre/dated | + slug pre/dated |"
    sep = "|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    for r in rows:
        cfg = knob_summary(r["knobs"])
        native = f"{r['native_pre_cutoff']}/{r['native_dated']}"
        eff = f"{r['effective_pre_cutoff']}/{r['effective_dated']}"
        lines.append(f"| {r['tag']} | {cfg} | {r['n_results']} | {native} | {eff} |")
    return "\n".join(lines)


def compute_hybrid(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For each tag, find the news-topic and general-topic rows under otherwise
    matching knobs and produce a unioned hybrid row."""
    by_tag: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        topic = r["knobs"].get("topic", "news")
        # Match by (tag, non-topic-knobs); store both topic variants.
        non_topic = {k: v for k, v in r["knobs"].items() if k != "topic"}
        key = json.dumps(non_topic, sort_keys=True)
        by_tag[r["tag"]][key].append(r)

    hybrid_rows = []
    for tag, by_knobs in by_tag.items():
        for key, group in by_knobs.items():
            if len({r["knobs"].get("topic") for r in group}) < 2:
                continue
            news = [r for r in group if r["knobs"].get("topic") == "news"]
            general = [r for r in group if r["knobs"].get("topic") == "general"]
            if not news or not general:
                continue
            news_r = news[0]
            general_r = general[0]
            seen_urls: set[str] = set()
            unioned = []
            for src in (news_r, general_r):
                for c in src["results"]:
                    if c["url"] in seen_urls:
                        continue
                    seen_urls.add(c["url"])
                    unioned.append(c)
            native_pre = sum(1 for c in unioned if c["native_pre_cutoff"])
            native_dated = sum(1 for c in unioned if c["native_dated"])
            eff_pre = sum(1 for c in unioned if c["effective_pre_cutoff"])
            eff_dated = sum(1 for c in unioned if c["effective_dated"])
            cfg_knobs = {**json.loads(key), "topic": "hybrid(news+general)"}
            hybrid_rows.append({
                "tag": tag,
                "query": news_r["query"],
                "cutoff": news_r["cutoff"],
                "knobs": cfg_knobs,
                "n_results": len(unioned),
                "native_pre_cutoff": native_pre,
                "native_dated": native_dated,
                "effective_pre_cutoff": eff_pre,
                "effective_dated": eff_dated,
                "results": unioned,
            })
    return hybrid_rows


def print_url_slug_coverage(rows: list[dict[str, Any]]) -> None:
    """Audit: for general-mode rows with no native dates, what fraction of URLs
    yield a date via the slug regex?"""
    print("\n## URL-slug recovery coverage (general-mode, no native date)\n")
    print("| tag | knobs | undated_urls | slug_recovered | recovery_rate |")
    print("|---|---|---|---|---|")
    for r in rows:
        if r["knobs"].get("topic") != "general":
            continue
        undated = [c for c in r["results"] if not c["native_dated"]]
        recovered = [c for c in undated if c["slug_date"] is not None]
        if not undated:
            continue
        print(
            f"| {r['tag']} | {knob_summary(r['knobs'])} | {len(undated)} | "
            f"{len(recovered)} | {len(recovered) / len(undated):.0%} |"
        )


def print_undated_url_sample(rows: list[dict[str, Any]], n: int = 30) -> None:
    """For Phase E: list a sample of undated, slug-non-matching URLs so we can
    eyeball what patterns Tavily-general returns."""
    print("\n## Undated URLs that the slug regex does NOT catch (sample)\n")
    seen: set[str] = set()
    count = 0
    for r in rows:
        if r["knobs"].get("topic") != "general":
            continue
        for c in r["results"]:
            if c["native_dated"] or c["slug_date"] is not None:
                continue
            if c["url"] in seen:
                continue
            seen.add(c["url"])
            print(f"- [{r['tag']}] {c['url']}")
            count += 1
            if count >= n:
                return


def main() -> None:
    rows = load_all()
    if not rows:
        print("No probe-results/*.json found. Run probe_tavily_topic.py first.")
        return

    print(f"# Tavily probe analysis ({len(rows)} runs)\n")
    print("## All runs\n")
    print(emit_table(rows))

    hybrid = compute_hybrid(rows)
    if hybrid:
        print("\n## Hybrid (news+general union)\n")
        print(emit_table(hybrid))

    print_url_slug_coverage(rows)
    print_undated_url_sample(rows)

    # Total call count = number of payloads (one Tavily call per cache entry).
    print(f"\n_Total cached Tavily calls: {len(rows)}_")


if __name__ == "__main__":
    main()
