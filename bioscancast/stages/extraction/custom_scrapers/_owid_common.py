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
import math
import statistics
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


def _r2(y: list[float], yhat: list[float]) -> float:
    if not y or len(y) != len(yhat):
        return 0.0
    y_mean = sum(y) / len(y)
    ss_res = sum((a - b) ** 2 for a, b in zip(y, yhat))
    ss_tot = sum((a - y_mean) ** 2 for a in y)
    if ss_tot <= 0.0:
        return 1.0 if ss_res <= 1e-12 else 0.0
    return max(0.0, min(1.0, 1.0 - (ss_res / ss_tot)))


def _fit_linear(values: list[float]) -> dict[str, float] | None:
    if len(values) < 2:
        return None
    n = len(values)
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(values) / n
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom <= 0.0:
        return None
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values)) / denom
    intercept = y_mean - (slope * x_mean)
    yhat = [intercept + slope * x for x in xs]
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "r2": _r2(values, yhat),
    }


def _fit_exponential(values: list[float]) -> dict[str, float] | None:
    if len(values) < 2:
        return None
    points = [(idx, v) for idx, v in enumerate(values) if v > 0]
    if len(points) < 2:
        return None

    xs = [float(idx) for idx, _ in points]
    ys = [float(v) for _, v in points]
    ln_ys = [math.log(v) for v in ys]

    n = len(xs)
    x_mean = sum(xs) / n
    y_mean = sum(ln_ys) / n
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom <= 0.0:
        return None

    b = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ln_ys)) / denom
    ln_a = y_mean - (b * x_mean)
    a = math.exp(ln_a)
    yhat = [a * math.exp(b * x) for x in xs]
    return {
        "a": float(a),
        "b": float(b),
        "r2": _r2(ys, yhat),
    }


def _fmt_num(value: float | None, *, decimals: int = 2) -> str:
    if value is None:
        return "n/a"
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,}"
    return f"{value:,.{decimals}f}"


def _stats_summary(values: list[float]) -> str:
    if not values:
        return "n=0"
    n = len(values)
    mean = sum(values) / n
    std = statistics.stdev(values) if n > 1 else 0.0
    min_v = min(values)
    med = statistics.median(values)
    max_v = max(values)
    return (
        f"n={n}, mean={_fmt_num(mean)}, std={_fmt_num(std)}, "
        f"min={_fmt_num(min_v)}, median={_fmt_num(med)}, max={_fmt_num(max_v)}"
    )


def _cumulative_line(dataset: OWIDDataset, entity: str, dt: datetime, row: dict[str, str]) -> str:
    """One prose sentence stating an entity's cumulative figures as of a date."""
    parts = [f"As of {dt.date().isoformat()}, {html.escape(entity)}:"]
    for column, phrase in dataset.cumulative_metrics:
        value = _fmt_number(row.get(column))
        if value is not None:
            parts.append(f" {phrase} ({html.escape(entity)}): {value};")
    return " ".join(parts).rstrip(";")


def _trend_table_html(
    dataset: OWIDDataset,
    entity: str,
    ordered_rows: list[tuple[datetime, dict[str, str]]],
    *,
    is_target_entity: bool,
) -> list[str]:
    cols = [c for c in dataset.trend_columns if ordered_rows and c in ordered_rows[-1][1]]
    if not cols:
        return []
    series = [(dt, row) for dt, row in ordered_rows if _parse_float(row.get(dataset.value_col)) is not None]
    value_series = [(dt, _parse_float(row.get(dataset.value_col))) for dt, row in series]
    d4 = _series_delta([(dt, v) for dt, v in value_series if v is not None], 4)
    d12 = _series_delta([(dt, v) for dt, v in value_series if v is not None], 12)
    d7 = _series_delta([(dt, v) for dt, v in value_series if v is not None], 7)
    d30 = _series_delta([(dt, v) for dt, v in value_series if v is not None], 30)

    lines = [f"<h3>{html.escape(entity)} — recent trend</h3>"]
    clean_series = [(dt, v) for dt, v in value_series if v is not None]
    if clean_series and clean_series[-1][1] is not None:
        latest_dt, latest_val = clean_series[-1]
        row_span_days = (
            (clean_series[-1][0] - clean_series[0][0]).days
            if len(clean_series) > 1
            else 0
        )
        lines.append(
            f"<p>Latest {html.escape(dataset.value_col)} "
            f"({latest_dt.date().isoformat()}): {latest_val:,.0f}; "
            f"delta_4_rows: {d4 if d4 is not None else 'n/a'}; "
            f"delta_12_rows: {d12 if d12 is not None else 'n/a'}</p>"
        )

        cumulative_values = [v for _, v in clean_series]
        cumulative_fit_window = cumulative_values[-30:]
        cumulative_linear = _fit_linear(cumulative_fit_window)
        increments = [
            cumulative_values[i] - cumulative_values[i - 1]
            for i in range(1, len(cumulative_values))
        ]

        lines.append(
            f"<p>Trend summary ({html.escape(dataset.value_col)}, {html.escape(entity)}): "
            f"points={len(cumulative_values)}, span_days={row_span_days}, "
            f"delta_7_rows={_fmt_num(d7)}, delta_30_rows={_fmt_num(d30)}, "
            f"recent_increment_stats={_stats_summary(increments)}.</p>"
        )

        if cumulative_linear is None:
            lines.append(
                "<p>Linear fit on recent cumulative trend: unavailable "
                "(insufficient points).</p>"
            )
        else:
            lines.append(
                "<p>Linear fit on recent cumulative trend "
                f"({html.escape(dataset.value_col)}, last {len(cumulative_fit_window)} rows): "
                f"intercept={cumulative_linear['intercept']:.6f}, "
                f"slope={cumulative_linear['slope']:.6f} per row, "
                f"R^2={cumulative_linear['r2']:.6f}.</p>"
            )

        if "new_cases" in cols:
            new_case_values = [
                _parse_float(row.get("new_cases"))
                for _dt, row in ordered_rows
            ]
            new_case_values = [v for v in new_case_values if v is not None]
            new_case_fit_window = new_case_values[-30:]
            exp_fit = _fit_exponential(new_case_fit_window)
            lines.append(
                f"<p>Incident trend stats (new_cases, {html.escape(entity)}): "
                f"{_stats_summary(new_case_fit_window)}.</p>"
            )
            if exp_fit is None:
                lines.append(
                    "<p>Exponential fit on recent new_cases: unavailable "
                    "(insufficient strictly-positive points).</p>"
                )
            else:
                lines.append(
                    "<p>Exponential fit on recent new_cases "
                    f"(last {len(new_case_fit_window)} rows): "
                    f"a={exp_fit['a']:.6f}, b={exp_fit['b']:.6f}, "
                    f"R^2={exp_fit['r2']:.6f}.</p>"
                )

        if is_target_entity:
            lines.append(
                f"<p>Region/question-target focus: {html.escape(entity)} trend "
                "statistics and model-fit outputs are included in this section.</p>"
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
        summary_lines.extend(
            _trend_table_html(
                dataset,
                entity,
                rows_by_location[entity],
                is_target_entity=entity in target_locations,
            )
        )

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
