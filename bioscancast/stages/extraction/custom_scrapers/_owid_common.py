"""Shared Our World in Data CSV scraper core (issues #34 / #38).

OWID topic pages (``ourworldindata.org/mpox``, ``/coronavirus``) render their
charts client-side, so a static scrape captures only prose + a citations list —
no case numbers (issue #34). OWID also publishes the underlying data as plain
CSV time series (canonically on GitHub raw), carrying the full daily history
with ``location`` and ``date`` columns plus **cumulative** ``total_cases`` /
``total_deaths``. We fetch that instead, select the target entity, apply the
as-of cutoff as a row filter, and render a compact HTML summary the existing
HTML extraction pipeline consumes unchanged.

This module holds the logic shared by every OWID dataset. Each dataset gets a
thin ``custom_scrapers/<source_id>.py`` module (matching its ``sources.yaml``
id) that defines an :class:`OWIDDataset` and delegates to :func:`fetch_owid`.

Why cumulative, not 7-day incidence: the benchmark mpox questions (q7/q8)
resolve on cumulative counts (~130,000). The earlier explorer endpoint
(``Metric=Confirmed cases&Frequency=7-day average``) returns only
``new_cases_smoothed`` — a ~2-digit daily incidence number — which is
confidently wrong for a cumulative question (issue #38, finding 1). We surface
``total_cases`` / ``total_deaths`` explicitly rather than guessing the column.

Caveat (carried from #34): OWID revises historical values over time, so a
replay uses *today's CSV's* estimate of the cutoff-date value, not the value as
published on that date. That is a far smaller error than the zero-record
fallback it replaces, and is exactly correct for live forecasting. See
``data/investigations/issue-34-owid-dashboard-fetch.md``.
"""

from __future__ import annotations

import csv
import html
import io
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

from bioscancast.stages.extraction.config import ExtractionConfig
from bioscancast.stages.extraction.fetcher import FetchResult

logger = logging.getLogger(__name__)

# OWID CSVs can be large — the COVID file is ~70 MB, well past
# ExtractionConfig.fetch_max_bytes (25 MB). These are trusted, static publisher
# files, so the CSV fetch uses its own generous ceiling instead of the
# streaming byte cap the generic fetcher applies. Guards runaway memory only.
_OWID_CSV_MAX_BYTES = 250_000_000  # 250 MB

CsvFetcher = Callable[[str, ExtractionConfig], Optional[str]]


@dataclass(frozen=True)
class OWIDDataset:
    """Maps an OWID dashboard family to its structured CSV time series."""

    name: str
    csv_url: str
    label: str
    """Human label for the summary, e.g. ``"mpox"`` / ``"COVID-19"``."""
    entity: str = "World"
    """Default headline entity when the question names no specific location."""
    location_col: str = "location"
    date_col: str = "date"
    value_col: str = "total_cases"
    """Cumulative column that drives the series/snapshot/targeting ranking."""
    cumulative_metrics: tuple[tuple[str, str], ...] = (
        ("total_cases", "cumulative confirmed cases"),
        ("total_deaths", "cumulative deaths"),
    )
    """(csv_column, human_phrase) pairs surfaced in the prose summary."""
    trend_columns: tuple[str, ...] = ("date", "new_cases", "total_cases")
    trend_rows: int = 16


def _fetch_csv_text(url: str, cfg: ExtractionConfig) -> Optional[str]:
    """Fetch a plain CSV over HTTP. Returns text, or None on failure.

    Uses curl_cffi (the same client the rest of extraction uses) so we inherit
    browser-TLS impersonation; these are static files so it is mostly belt-and-
    braces. Unlike the generic fetcher this does not apply the 25 MB streaming
    cap — the COVID CSV legitimately exceeds it.
    """
    try:
        from curl_cffi import requests as curl_requests

        resp = curl_requests.get(
            url,
            timeout=max(cfg.fetch_timeout_seconds, 60.0),
            impersonate=cfg.impersonate,
            allow_redirects=True,
        )
    except Exception as exc:  # noqa: BLE001 - network best-effort
        logger.info("OWID CSV fetch failed for %s: %s", url, exc)
        return None

    if resp.status_code != 200 or not resp.content:
        logger.info("OWID CSV fetch returned status=%s for %s", resp.status_code, url)
        return None
    if len(resp.content) > _OWID_CSV_MAX_BYTES:
        logger.info(
            "OWID CSV too large (%d bytes > %d) for %s",
            len(resp.content), _OWID_CSV_MAX_BYTES, url,
        )
        return None
    return resp.content.decode("utf-8", errors="replace")


def _pick_col(fieldnames: list[str], options: tuple[str, ...]) -> str | None:
    lowered = {name.lower(): name for name in fieldnames}
    for opt in options:
        found = lowered.get(opt.lower())
        if found:
            return found
    return None


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    raw = value.strip().replace(",", "")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _fmt_number(raw: str | None) -> str | None:
    """Format an OWID numeric cell as a thousands-separated value.

    OWID writes cumulative counts as floats (e.g. ``129602.0``). Returns None
    for blank/non-numeric cells so absent metrics are omitted from the summary.
    """
    value = _parse_float(raw)
    if value is None:
        return None
    if value == int(value):
        return f"{int(value):,}"
    return f"{value:,.2f}"


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _as_utc(dt: datetime) -> datetime:
    """Make a datetime tz-aware (UTC) so cutoff comparisons never raise."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _infer_locations_from_question(
    *,
    question_text: str,
    known_locations: set[str],
) -> list[str]:
    q = question_text.lower()
    matches = [loc for loc in known_locations if loc.lower() in q]
    matches.sort(key=len, reverse=True)
    return matches[:3]


def _resolve_target_locations(
    *,
    region: str | None,
    question_text: str | None,
    known_locations: set[str],
) -> list[str]:
    if region:
        direct = [loc for loc in known_locations if loc.lower() == region.lower()]
        if direct:
            return direct

    if question_text:
        inferred = _infer_locations_from_question(
            question_text=question_text,
            known_locations=known_locations,
        )
        if inferred:
            return inferred

    for fallback in ("World", "Global"):
        if fallback in known_locations:
            return [fallback]
    return []


def _series_delta(series: list[tuple[datetime, float]], n_back: int) -> float | None:
    if len(series) <= n_back:
        return None
    return series[-1][1] - series[-(n_back + 1)][1]


def _cumulative_line(dataset: OWIDDataset, entity: str, dt: datetime, row: dict[str, str]) -> str:
    """One prose sentence stating an entity's cumulative figures as of a date."""
    parts = [f"As of {dt.date().isoformat()}, {html.escape(entity)}:"]
    for column, phrase in dataset.cumulative_metrics:
        value = _fmt_number(row.get(column))
        if value is not None:
            parts.append(f" {phrase} ({html.escape(entity)}): {value};")
    return " ".join(parts).rstrip(";")


def _trend_table_html(dataset: OWIDDataset, entity: str, ordered_rows: list[tuple[datetime, dict[str, str]]]) -> list[str]:
    cols = [c for c in dataset.trend_columns if ordered_rows and c in ordered_rows[-1][1]]
    if not cols:
        return []
    series = [(dt, row) for dt, row in ordered_rows if _parse_float(row.get(dataset.value_col)) is not None]
    value_series = [(dt, _parse_float(row.get(dataset.value_col))) for dt, row in series]
    d4 = _series_delta([(dt, v) for dt, v in value_series if v is not None], 4)
    d12 = _series_delta([(dt, v) for dt, v in value_series if v is not None], 12)

    lines = [f"<h3>{html.escape(entity)} — recent trend</h3>"]
    if value_series and value_series[-1][1] is not None:
        latest_dt, latest_val = value_series[-1]
        lines.append(
            f"<p>Latest {html.escape(dataset.value_col)} "
            f"({latest_dt.date().isoformat()}): {latest_val:,.0f}; "
            f"delta_4_rows: {d4 if d4 is not None else 'n/a'}; "
            f"delta_12_rows: {d12 if d12 is not None else 'n/a'}</p>"
        )
    header = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    lines.append(f"<table><thead><tr>{header}</tr></thead><tbody>")
    for _dt, row in ordered_rows[-dataset.trend_rows:]:
        cells = "".join(f"<td>{html.escape((row.get(c) or '').strip())}</td>" for c in cols)
        lines.append(f"<tr>{cells}</tr>")
    lines.append("</tbody></table>")
    return lines


def fetch_owid(
    url: str,
    dataset: OWIDDataset,
    *,
    config: ExtractionConfig | None = None,
    as_of_date: datetime | None = None,
    region: str | None = None,
    question_text: str | None = None,
    csv_fetcher: CsvFetcher | None = None,
) -> FetchResult | None:
    """Fetch an OWID dataset CSV and render it to a compact HTML summary.

    Returns None on any failure (network, empty/changed schema, no rows before
    the cutoff) so the dispatcher falls back to the generic fetch path rather
    than dropping the document.
    """
    cfg = config or ExtractionConfig()
    fetch_csv = csv_fetcher or _fetch_csv_text
    fetched_at = datetime.now(timezone.utc)

    text = fetch_csv(dataset.csv_url, cfg)
    if not text:
        return None

    reader = csv.DictReader(io.StringIO(text))
    fieldnames = list(reader.fieldnames or [])
    rows = list(reader)
    if not fieldnames or not rows:
        return None

    location_col = _pick_col(fieldnames, (dataset.location_col, "entity", "country", "location"))
    date_col = _pick_col(fieldnames, (dataset.date_col, "day", "date", "year"))
    value_col = _pick_col(fieldnames, (dataset.value_col,))
    if not location_col or not date_col or not value_col:
        return None

    cutoff = _as_utc(as_of_date) if as_of_date else fetched_at

    # Full per-location history (rows on-or-before the cutoff), date-ordered.
    rows_by_location: dict[str, list[tuple[datetime, dict[str, str]]]] = {}
    for row in rows:
        location = (row.get(location_col) or "").strip()
        if not location:
            continue
        dt = _parse_date(row.get(date_col))
        if dt is None or dt > cutoff:
            continue
        if _parse_float(row.get(value_col)) is None:
            continue
        rows_by_location.setdefault(location, []).append((dt, row))

    if not rows_by_location:
        return None

    for loc in rows_by_location:
        rows_by_location[loc].sort(key=lambda item: item[0])

    def latest_value(loc: str) -> float:
        dt, row = rows_by_location[loc][-1]
        return _parse_float(row.get(value_col)) or 0.0

    # Headline entity: the dataset default (World) when present, else the
    # largest-cumulative location so the summary is never empty.
    if dataset.entity in rows_by_location:
        headline = dataset.entity
    else:
        headline = max(rows_by_location, key=latest_value)

    target_locations = _resolve_target_locations(
        region=region,
        question_text=question_text,
        known_locations=set(rows_by_location.keys()),
    )
    # Always lead with the headline entity; then any question/region targets.
    ordered_entities: list[str] = [headline]
    for loc in target_locations:
        if loc not in ordered_entities:
            ordered_entities.append(loc)

    top_locations = sorted(
        (
            (loc, rows_by_location[loc][-1][0], latest_value(loc))
            for loc in rows_by_location
            if loc not in {"World", "Global"}
        ),
        key=lambda item: item[2],
        reverse=True,
    )[:30]

    summary_lines = [
        "<html><head><meta charset='utf-8'>"
        f"<title>Our World in Data — {html.escape(dataset.label)} summary</title></head><body>",
        f"<h1>Our World in Data — {html.escape(dataset.label)}</h1>",
        f"<p>CSV source: {html.escape(dataset.csv_url)}</p>",
        f"<p>Cutoff: {cutoff.date().isoformat()}; "
        f"locations with usable rows: {len(rows_by_location)}; "
        f"region hint: {html.escape(region or '(none)')}</p>",
    ]

    # Prose headline: the extractable, unambiguous cumulative figure(s).
    for entity in ordered_entities:
        dt, row = rows_by_location[entity][-1]
        summary_lines.append(f"<h2>{html.escape(entity)}</h2>")
        summary_lines.append(
            f"<p>Our World in Data — {html.escape(dataset.label)} ({html.escape(entity)}). "
            f"{_cumulative_line(dataset, entity, dt, row)}. "
            f"Source: {html.escape(dataset.csv_url)}.</p>"
        )
        summary_lines.extend(_trend_table_html(dataset, entity, rows_by_location[entity]))

    if top_locations:
        summary_lines.append("<h2>Top locations by latest cumulative value</h2>")
        summary_lines.append(
            "<table><thead><tr><th>Location</th><th>Date</th>"
            f"<th>{html.escape(dataset.value_col)}</th></tr></thead><tbody>"
        )
        for loc, dt, value in top_locations:
            summary_lines.append(
                "<tr>"
                f"<td>{html.escape(loc)}</td>"
                f"<td>{dt.date().isoformat()}</td>"
                f"<td>{value:,.0f}</td>"
                "</tr>"
            )
        summary_lines.append("</tbody></table>")

    summary_lines.append(
        "<p>Note: this custom scraper summarizes OWID CSV rows into compact HTML "
        "so the existing HTML extraction pipeline can process them "
        "deterministically. Figures are cumulative totals as of the cutoff.</p>"
    )
    summary_lines.append("</body></html>")

    rendered = "\n".join(summary_lines).encode("utf-8")
    return FetchResult(
        url=dataset.csv_url,
        final_url=dataset.csv_url,
        status_code=200,
        content_type="text/html",
        content_bytes=rendered,
        fetched_at=fetched_at,
        error=None,
    )
