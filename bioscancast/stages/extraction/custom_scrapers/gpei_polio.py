"""Custom scraper for the GPEI "Polio this week" page (``gpei_polio``).

The page carries the current wild-poliovirus (WPV1) case narrative and a
year-to-date country table, but the generic HTML parser drops the main content
container (extracting only ~1.4k chars of chrome), so polio questions got 0
records. This fetches the page, strips nav/script/style, and re-renders the
visible text as minimal HTML the existing parser consumes cleanly — the same
"render to clean HTML" approach the OWID scrapers use.
"""

from __future__ import annotations

import html as html_lib
from datetime import datetime, timezone

from bs4 import BeautifulSoup
from curl_cffi import requests as curl_requests

from bioscancast.stages.extraction.config import ExtractionConfig
from bioscancast.stages.extraction.fetcher import FetchResult

_STRIP_TAGS = ("script", "style", "noscript", "form")


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
    try:
        resp = curl_requests.get(
            url, timeout=max(cfg.fetch_timeout_seconds, 30.0),
            impersonate=cfg.impersonate, allow_redirects=True,
        )
    except Exception:  # noqa: BLE001 - network best-effort
        return None
    if resp.status_code != 200 or not resp.text:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(list(_STRIP_TAGS)):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    # Keep substantive lines (the WPV1/cVDPV narrative + country table rows);
    # drop one/two-word nav fragments.
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 3]
    if sum(len(ln) for ln in lines) < 500:
        return None  # nothing useful recovered; fall back to generic fetch

    paragraphs = "".join(f"<p>{html_lib.escape(ln)}</p>" for ln in lines)
    rendered = (
        "<html><head><meta charset='utf-8'><title>GPEI Polio this week</title></head><body>"
        f"<h1>GPEI Polio this week</h1><p>Source: {html_lib.escape(url)}</p>"
        f"{paragraphs}</body></html>"
    ).encode("utf-8")

    return FetchResult(
        url=url, final_url=str(resp.url), status_code=200,
        content_type="text/html", content_bytes=rendered,
        fetched_at=fetched_at, error=None,
    )
