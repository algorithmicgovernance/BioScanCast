"""Custom scraper for the CDC measles data page (``cdc_measles``).

The CDC "Measles Cases and Outbreaks" page renders its case/death/hospitalisation
figures client-side from a JSON data endpoint. A static scrape gets the case
*count* from prose (so q14 works) but the **deaths** table cells are empty in the
served HTML — they are JS-injected — so death questions (q16) extracted 0 records.

The data is published as plain JSON at ``/wcms/vizdata/measles/measles_hosp.json``
(no browser or Akamai challenge needed), carrying, per year, ``total_cases``,
``total_deaths`` and a ready-made ``deaths_sentence``. We fetch that and render a
compact HTML summary the existing HTML extraction pipeline consumes unchanged, so
the insight stage gets an unambiguous "N confirmed measles deaths in 2026" fact
(and the case count, corroborating the prose figure q14 already uses).

Returns ``None`` (fall back to the generic fetch) on any failure or in
historical-replay mode — the endpoint carries only current per-year aggregates
with no date history to cut on, so serving it under an ``as_of_date`` would leak.
"""

from __future__ import annotations

import html
import json
import logging
from datetime import datetime, timezone

from bioscancast.stages.extraction.config import ExtractionConfig
from bioscancast.stages.extraction.fetcher import FetchResult

logger = logging.getLogger(__name__)

# The page's client-side data feed. Absolute (host-pinned) because this scraper
# is specific to the CDC measles source; the hub ``url`` passed in is the
# human-facing data-research page.
_DATA_URL = "https://www.cdc.gov/wcms/vizdata/measles/measles_hosp.json"

# Years to surface, newest first. The current year answers q14/q16; the prior
# year is kept as corroborating context (and a base rate for the death question).
_YEARS = ("2026", "2025")


def _first(value) -> str | None:
    """CDC wraps each field as a single-element list (e.g. ``["0"]``)."""
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _get_json(url: str, cfg: ExtractionConfig):
    try:
        from curl_cffi import requests as curl_requests

        resp = curl_requests.get(
            url,
            timeout=max(cfg.fetch_timeout_seconds, 30.0),
            impersonate=cfg.impersonate,
            allow_redirects=True,
        )
    except Exception as exc:  # noqa: BLE001 - network best-effort
        logger.info("CDC measles JSON fetch failed for %s: %s", url, exc)
        return None
    if resp.status_code != 200 or not resp.content:
        return None
    try:
        return resp.json()
    except (ValueError, json.JSONDecodeError):
        return None


def fetch(
    url: str,
    *,
    config: ExtractionConfig | None = None,
    as_of_date: datetime | None = None,
    region: str | None = None,
    question_text: str | None = None,
    json_getter=None,
) -> FetchResult | None:
    # Live-only: the feed has no per-date history, so serving it in replay mode
    # would leak post-cutoff values. Fall back to the generic (Wayback) path.
    if as_of_date is not None:
        return None

    cfg = config or ExtractionConfig()
    fetched_at = datetime.now(timezone.utc)

    # ``json_getter`` is injectable for no-network tests; defaults to a live
    # curl_cffi fetch of the CDC data feed.
    getter = json_getter or _get_json
    data = getter(_DATA_URL, cfg)
    if not isinstance(data, dict):
        return None

    blocks: list[str] = []
    for year in _YEARS:
        rec = data.get(year)
        if not isinstance(rec, dict):
            continue
        cases = _first(rec.get("total_cases"))
        deaths = _first(rec.get("total_deaths"))
        deaths_sentence = _first(rec.get("deaths_sentence"))
        if cases is None and deaths is None:
            continue
        parts = [f"<h2>United States measles, {html.escape(year)}</h2>", "<p>"]
        if cases is not None:
            parts.append(
                f"As of the latest CDC update, a total of {html.escape(cases)} "
                f"confirmed measles cases were reported in the United States in "
                f"{html.escape(year)}. "
            )
        if deaths_sentence is not None:
            parts.append(html.escape(deaths_sentence) + " ")
        elif deaths is not None:
            parts.append(
                f"There have been {html.escape(deaths)} confirmed measles deaths "
                f"in the United States in {html.escape(year)}. "
            )
        parts.append("</p>")
        blocks.append("".join(parts))

    if not blocks:
        return None

    rendered = (
        "<html><head><meta charset='utf-8'>"
        "<title>CDC — Measles Cases and Outbreaks (United States)</title></head><body>"
        "<h1>CDC — Measles Cases and Outbreaks (United States)</h1>"
        f"<p>Source: {html.escape(url)} "
        f"(data feed {html.escape(_DATA_URL)}), retrieved "
        f"{fetched_at.date().isoformat()}.</p>"
        + "".join(blocks)
        + "<p>Note: this custom scraper renders the CDC measles JSON data feed "
        "into compact HTML because the page's case/death figures are injected "
        "client-side and are absent from the statically served table cells. "
        "Counts are cumulative year-to-date totals.</p>"
        "</body></html>"
    ).encode("utf-8")

    return FetchResult(
        url=_DATA_URL,
        final_url=_DATA_URL,
        status_code=200,
        content_type="text/html",
        content_bytes=rendered,
        fetched_at=fetched_at,
        error=None,
    )
