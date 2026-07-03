from __future__ import annotations

import csv
import html
import io
from datetime import datetime, timezone
from typing import Optional

from curl_cffi import requests as curl_requests

from bioscancast.stages.extraction.config import ExtractionConfig
from bioscancast.stages.extraction.fetcher import FetchResult

_OWID_MPOX_CSV_URL = (
    "https://ourworldindata.org/explorers/monkeypox.csv?"
    "v=1&csvType=full&useColumnShortNames=true&"
    "Metric=Confirmed+cases&Frequency=7-day+average&Relative+to+population=false"
)


def _normalize_content_type(header_value: Optional[str]) -> Optional[str]:
    if not header_value:
        return None
    return header_value.split(";", 1)[0].strip().lower()


def _fetch_bytes(
    url: str,
    cfg: ExtractionConfig,
) -> tuple[Optional[bytes], str, Optional[int], Optional[str], Optional[str]]:
    try:
        response = curl_requests.get(
            url,
            stream=True,
            timeout=cfg.fetch_timeout_seconds,
            impersonate=cfg.impersonate,
            allow_redirects=True,
        )
        try:
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > cfg.fetch_max_bytes:
                return (
                    None,
                    str(response.url),
                    response.status_code,
                    _normalize_content_type(response.headers.get("content-type")),
                    f"Content-Length {content_length} exceeds max {cfg.fetch_max_bytes} bytes",
                )

            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_content():
                total += len(chunk)
                if total > cfg.fetch_max_bytes:
                    return (
                        None,
                        str(response.url),
                        response.status_code,
                        _normalize_content_type(response.headers.get("content-type")),
                        f"Response exceeded max {cfg.fetch_max_bytes} bytes during streaming",
                    )
                chunks.append(chunk)

            return (
                b"".join(chunks),
                str(response.url),
                response.status_code,
                _normalize_content_type(response.headers.get("content-type")),
                None,
            )
        finally:
            response.close()
    except Exception as exc:  # noqa: BLE001
        return None, url, None, None, str(exc)


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


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    # OWID dates are usually YYYY-MM-DD.
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _resolve_metric_col(fieldnames: list[str], rows: list[dict[str, str]]) -> str | None:
    preferred = (
        "Confirmed cases",
        "Confirmed cases (7-day average)",
        "cases",
        "value",
    )
    direct = _pick_col(fieldnames, preferred)
    if direct:
        return direct

    excluded = {
        "entity",
        "country",
        "location",
        "code",
        "iso code",
        "iso_code",
        "day",
        "date",
        "year",
    }
    candidates = [f for f in fieldnames if f.lower() not in excluded]
    best_col: str | None = None
    best_score = -1
    for col in candidates:
        numeric_count = 0
        for row in rows:
            if _parse_float(row.get(col)) is not None:
                numeric_count += 1
        if numeric_count > best_score:
            best_score = numeric_count
            best_col = col
    return best_col


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


def fetch(
    url: str,
    *,
    config: ExtractionConfig | None = None,
    as_of_date: datetime | None = None,
    region: str | None = None,
    question_text: str | None = None,
) -> FetchResult | None:
    cfg = config or ExtractionConfig()
    fetched_at = datetime.now(timezone.utc)
    csv_url = _OWID_MPOX_CSV_URL

    content_bytes, final_url, status_code, _content_type, err = _fetch_bytes(csv_url, cfg)
    if err is not None or content_bytes is None:
        return None

    text = content_bytes.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = list(reader.fieldnames or [])
    rows = list(reader)
    if not fieldnames or not rows:
        return None

    location_col = _pick_col(fieldnames, ("Entity", "Country", "Location"))
    date_col = _pick_col(fieldnames, ("Day", "Date", "Year"))
    metric_col = _resolve_metric_col(fieldnames, rows)
    if not location_col or not date_col or not metric_col:
        return None

    cutoff = as_of_date or fetched_at
    latest_by_location: dict[str, tuple[datetime, float]] = {}
    for row in rows:
        location = (row.get(location_col) or "").strip()
        if not location:
            continue
        dt = _parse_date(row.get(date_col))
        if dt is None or dt > cutoff:
            continue
        metric_val = _parse_float(row.get(metric_col))
        if metric_val is None:
            continue
        current = latest_by_location.get(location)
        if current is None or dt > current[0]:
            latest_by_location[location] = (dt, metric_val)

    if not latest_by_location:
        return None

    # Build full per-location time series for trend extraction.
    series_by_location: dict[str, list[tuple[datetime, float]]] = {}
    for row in rows:
        location = (row.get(location_col) or "").strip()
        if not location:
            continue
        dt = _parse_date(row.get(date_col))
        if dt is None or dt > cutoff:
            continue
        metric_val = _parse_float(row.get(metric_col))
        if metric_val is None:
            continue
        series_by_location.setdefault(location, []).append((dt, metric_val))

    for loc in list(series_by_location.keys()):
        series_by_location[loc].sort(key=lambda item: item[0])

    world_snapshot = None
    for key in ("World", "Global"):
        if key in latest_by_location:
            world_snapshot = (key, latest_by_location[key])
            break

    top_locations = sorted(
        (
            (loc, dt_val[0], dt_val[1])
            for loc, dt_val in latest_by_location.items()
            if loc not in {"World", "Global"}
        ),
        key=lambda item: item[2],
        reverse=True,
    )[:30]

    target_locations = _resolve_target_locations(
        region=region,
        question_text=question_text,
        known_locations=set(series_by_location.keys()),
    )

    summary_lines = [
        "<html><head><meta charset='utf-8'><title>OWID Mpox CSV Summary</title></head><body>",
        "<h1>Our World in Data - Mpox CSV Summary</h1>",
        f"<p>CSV source: {html.escape(final_url)}</p>",
        f"<p>Metric column: {html.escape(metric_col)}</p>",
        f"<p>Locations with usable rows: {len(latest_by_location)}</p>",
        f"<p>Region hint: {html.escape(region or '(none)')}</p>",
    ]

    if world_snapshot is not None:
        world_loc, (world_dt, world_value) = world_snapshot
        summary_lines.append(
            "<h2>Latest global snapshot</h2>"
            f"<p>{html.escape(world_loc)} on {world_dt.date().isoformat()}: {world_value:,.2f}</p>"
        )

    summary_lines.append("<h2>Top locations by latest value</h2>")
    summary_lines.append("<table><thead><tr><th>Location</th><th>Date</th><th>Value</th></tr></thead><tbody>")
    for loc, dt, value in top_locations:
        summary_lines.append(
            "<tr>"
            f"<td>{html.escape(loc)}</td>"
            f"<td>{dt.date().isoformat()}</td>"
            f"<td>{value:,.2f}</td>"
            "</tr>"
        )
    summary_lines.append("</tbody></table>")

    if target_locations:
        summary_lines.append("<h2>Target Region Time Series (recent)</h2>")
        for loc in target_locations:
            series = series_by_location.get(loc, [])
            if not series:
                continue
            latest_dt, latest_value = series[-1]
            d4 = _series_delta(series, 4)
            d12 = _series_delta(series, 12)
            summary_lines.append(f"<h3>{html.escape(loc)}</h3>")
            summary_lines.append(
                f"<p>Latest ({latest_dt.date().isoformat()}): {latest_value:,.2f}; "
                f"delta_4_points: {d4 if d4 is not None else 'n/a'}; "
                f"delta_12_points: {d12 if d12 is not None else 'n/a'}</p>"
            )
            summary_lines.append(
                "<table><thead><tr><th>Date</th><th>Value</th></tr></thead><tbody>"
            )
            for dt, value in series[-16:]:
                summary_lines.append(
                    "<tr>"
                    f"<td>{dt.date().isoformat()}</td>"
                    f"<td>{value:,.2f}</td>"
                    "</tr>"
                )
            summary_lines.append("</tbody></table>")

    summary_lines.append(
        "<p>Note: This custom scraper summarizes CSV rows into compact HTML so the existing "
        "HTML extraction pipeline can process it deterministically.</p>"
    )
    summary_lines.append("</body></html>")

    rendered = "\n".join(summary_lines).encode("utf-8")
    return FetchResult(
        url=csv_url,
        final_url=final_url,
        status_code=status_code,
        content_type="text/html",
        content_bytes=rendered,
        fetched_at=fetched_at,
        error=None,
    )