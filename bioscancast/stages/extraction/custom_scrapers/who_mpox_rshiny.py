from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from typing import Any, Optional

from bs4 import BeautifulSoup
from curl_cffi import requests as curl_requests

from bioscancast.stages.extraction.config import ExtractionConfig
from bioscancast.stages.extraction.fetcher import FetchResult

_WHO_MPX_API_URL = "https://xmart-api-public.who.int/MPX/V_MPX_VALIDATED_DAILY"
_MAIN_URL_TEXT_MAX_BYTES = 2_500_000
_TREND_FIELDS = [
    "TOTAL_CONF_CASES",
    "TOTAL_CONF_DEATHS",
    "NEW_CONF_CASES",
    "NEW_CONF_DEATHS",
]


def _normalize_content_type(header_value: Optional[str]) -> Optional[str]:
    if not header_value:
        return None
    return header_value.split(";", 1)[0].strip().lower()


def _normalize_key(text: str) -> str:
    # Keep alphanumerics only so matching is robust across punctuation/spaces.
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def _fetch_json(
    *,
    url: str,
    cfg: ExtractionConfig,
) -> tuple[Optional[dict[str, Any]], str, Optional[int], Optional[str], Optional[str]]:
    try:
        response = curl_requests.get(
            url,
            stream=True,
            timeout=cfg.fetch_timeout_seconds,
            impersonate=cfg.impersonate,
            allow_redirects=True,
        )
        try:
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
            payload = b"".join(chunks)
            try:
                data = json.loads(payload.decode("utf-8"))
            except Exception as exc:  # noqa: BLE001
                return (
                    None,
                    str(response.url),
                    response.status_code,
                    _normalize_content_type(response.headers.get("content-type")),
                    f"invalid_json: {exc}",
                )
            return (
                data,
                str(response.url),
                response.status_code,
                _normalize_content_type(response.headers.get("content-type")),
                None,
            )
        finally:
            response.close()
    except Exception as exc:  # noqa: BLE001
        return None, url, None, None, str(exc)


def _fetch_main_url_excerpt(
    *,
    url: str,
    cfg: ExtractionConfig,
) -> tuple[Optional[str], str, Optional[int], bool, Optional[str]]:
    """Fetch a bounded excerpt of the main URL and extract visible text.

    Returns: (extracted_text, final_url, status_code, truncated, error)
    """
    try:
        response = curl_requests.get(
            url,
            stream=True,
            timeout=cfg.fetch_timeout_seconds,
            impersonate=cfg.impersonate,
            allow_redirects=True,
        )
        try:
            chunks: list[bytes] = []
            total = 0
            truncated = False
            for chunk in response.iter_content():
                total += len(chunk)
                if total > _MAIN_URL_TEXT_MAX_BYTES:
                    truncated = True
                    break
                chunks.append(chunk)

            html_bytes = b"".join(chunks)
            if not html_bytes:
                return None, str(response.url), response.status_code, truncated, None

            soup = BeautifulSoup(
                html_bytes.decode("utf-8", errors="replace"),
                "html.parser",
            )
            for tag in soup(["script", "style", "noscript"]):
                tag.extract()

            parts: list[str] = []
            title = soup.title.get_text(" ", strip=True) if soup.title else ""
            if title:
                parts.append(f"Title: {title}")

            for el in soup.find_all(["h1", "h2", "h3", "p", "li"]):
                text = el.get_text(" ", strip=True)
                if len(text) < 30:
                    continue
                parts.append(text)
                if len("\n".join(parts)) >= 7000:
                    break

            if not parts:
                fallback = soup.get_text(" ", strip=True)
                if fallback:
                    parts = [fallback[:7000]]

            if not parts:
                return None, str(response.url), response.status_code, truncated, None
            return "\n".join(parts), str(response.url), response.status_code, truncated, None
        finally:
            response.close()
    except Exception as exc:  # noqa: BLE001
        return None, url, None, False, str(exc)


def _parse_date(date_text: Any) -> datetime | None:
    if not isinstance(date_text, str):
        return None
    date_text = date_text.strip()
    if not date_text:
        return None
    try:
        return datetime.strptime(date_text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        raw = value.strip().replace(",", "")
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None
    return None


def _resolve_country_targets(
    *,
    region: str | None,
    question_text: str | None,
    records: list[dict[str, Any]],
) -> list[str]:
    countries = sorted(
        {
            (rec.get("COUNTRY") or "").strip()
            for rec in records
            if (rec.get("COUNTRY") or "").strip()
        }
    )
    iso_to_country = {
        (rec.get("ISO3") or "").strip().upper(): (rec.get("COUNTRY") or "").strip()
        for rec in records
        if (rec.get("ISO3") or "").strip() and (rec.get("COUNTRY") or "").strip()
    }
    norm_to_country = {_normalize_key(country): country for country in countries}

    if region:
        region_str = region.strip()
        if region_str:
            iso_hit = iso_to_country.get(region_str.upper())
            if iso_hit:
                return [iso_hit]
            direct = norm_to_country.get(_normalize_key(region_str))
            if direct:
                return [direct]

    if question_text:
        qnorm = _normalize_key(question_text)
        matches = [
            country
            for country in countries
            if _normalize_key(country) and _normalize_key(country) in qnorm
        ]
        if matches:
            matches.sort(key=len, reverse=True)
            return [matches[0]]

    return []


def _series_delta(values: list[tuple[datetime, float]], n_back: int) -> float | None:
    if len(values) <= n_back:
        return None
    return values[-1][1] - values[-(n_back + 1)][1]


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
    cutoff = as_of_date or fetched_at

    data, final_url, status_code, content_type, err = _fetch_json(url=_WHO_MPX_API_URL, cfg=cfg)
    if err is not None or data is None:
        return None

    rows = data.get("value") if isinstance(data, dict) else None
    if not isinstance(rows, list) or not rows:
        return None

    targets = _resolve_country_targets(
        region=region,
        question_text=question_text,
        records=[r for r in rows if isinstance(r, dict)],
    )
    if not targets:
        # Keep deterministic and strict: no country hint -> no custom summary.
        return None

    by_country: dict[str, list[dict[str, Any]]] = {country: [] for country in targets}
    for rec in rows:
        if not isinstance(rec, dict):
            continue
        country = (rec.get("COUNTRY") or "").strip()
        if country not in by_country:
            continue
        dt = _parse_date(rec.get("DATE"))
        if dt is None or dt > cutoff:
            continue
        rec_copy = dict(rec)
        rec_copy["_dt"] = dt
        by_country[country].append(rec_copy)

    if not any(by_country.values()):
        return None

    main_text, main_final_url, main_status, main_truncated, main_err = _fetch_main_url_excerpt(
        url=url,
        cfg=cfg,
    )

    lines = [
        "<html><head><meta charset='utf-8'><title>WHO Mpox API Country Trends</title></head><body>",
        "<h1>WHO Mpox Hybrid Summary (API + Main URL)</h1>",
        f"<p>API source: {html.escape(final_url)}</p>",
        f"<p>Main URL source: {html.escape(main_final_url)}</p>",
        f"<p>Region hint: {html.escape(region or '(none)')}</p>",
        f"<p>Question hint: {html.escape(question_text or '(none)')}</p>",
    ]

    lines.append("<h2>Main URL text insights</h2>")
    if main_text:
        lines.append("<p>Extracted narrative text from the Shiny page (bounded excerpt).</p>")
        if main_truncated:
            lines.append(
                f"<p>Note: Main URL content was truncated at {_MAIN_URL_TEXT_MAX_BYTES:,} bytes for safety.</p>"
            )
        lines.append("<pre>")
        lines.append(html.escape(main_text))
        lines.append("</pre>")
    else:
        lines.append(
            f"<p>No usable narrative text extracted from main URL. error={html.escape(str(main_err))}, status={main_status}</p>"
        )

    lines.append("<h2>API trend extraction</h2>")

    for country in targets:
        country_rows = by_country.get(country, [])
        if not country_rows:
            continue
        country_rows.sort(key=lambda r: r["_dt"])
        lines.append(f"<h2>{html.escape(country)}</h2>")

        latest = country_rows[-1]
        latest_date = latest["_dt"].date().isoformat()
        lines.append(f"<p>Latest date: {latest_date}</p>")

        for metric in _TREND_FIELDS:
            metric_series: list[tuple[datetime, float]] = []
            for rec in country_rows:
                val = _to_float(rec.get(metric))
                if val is None:
                    continue
                metric_series.append((rec["_dt"], val))

            if not metric_series:
                lines.append(f"<h3>{metric}</h3><p>No usable values.</p>")
                continue

            d7 = _series_delta(metric_series, 7)
            d14 = _series_delta(metric_series, 14)
            lines.append(f"<h3>{metric}</h3>")
            lines.append(
                f"<p>Latest: {metric_series[-1][1]:,.2f}; "
                f"delta_7_points: {d7 if d7 is not None else 'n/a'}; "
                f"delta_14_points: {d14 if d14 is not None else 'n/a'}</p>"
            )
            lines.append("<table><thead><tr><th>Date</th><th>Value</th></tr></thead><tbody>")
            for dt, val in metric_series[-30:]:
                lines.append(
                    "<tr>"
                    f"<td>{dt.date().isoformat()}</td>"
                    f"<td>{val:,.2f}</td>"
                    "</tr>"
                )
            lines.append("</tbody></table>")

        lines.append(
            "<p>Note: This custom scraper combines bounded main-URL text extraction "
            "with structured country trends from the WHO API dataset.</p>"
        )

    lines.append("</body></html>")
    rendered = "\n".join(lines).encode("utf-8")

    return FetchResult(
        url=_WHO_MPX_API_URL,
        final_url=final_url,
        status_code=status_code,
        content_type="text/html",
        content_bytes=rendered,
        fetched_at=fetched_at,
        error=None,
    )