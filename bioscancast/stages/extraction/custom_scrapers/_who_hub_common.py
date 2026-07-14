"""Shared WHO situation-hub scraper core.

Several WHO emergency hubs (cholera upsurge, mpox outbreak, ...) render as a
landing page that lists dated *epidemiological update* / *situation report*
item pages under ``who.int/publications/m/item/...``. The landing page itself
carries only narrative context, not the current case/death totals (issue: those
questions extracted 0 records because the injected hub is context-only), while
each dated item page links a ``cdn.who.int/.../situation-reports/*.pdf`` whose
tables hold the cumulative figures.

This module resolves a hub URL to the latest dated item at-or-before the cutoff
and returns that item's PDF bytes, so the existing PDF parser + Docling table
refiner (the ``situation-reports`` path is on the Docling allowlist) extract the
numbers unchanged. Each WHO source gets a thin ``custom_scrapers/<id>.py`` that
delegates here with its hub keyword.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from curl_cffi import requests as curl_requests

from bioscancast.stages.extraction.config import ExtractionConfig
from bioscancast.stages.extraction.fetcher import FetchResult

logger = logging.getLogger(__name__)

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

# "30 June 2026" / "5 Sept 2026"
_DATE_RE = re.compile(
    r"\b(\d{1,2})\s+"
    r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+(\d{4})\b",
    flags=re.IGNORECASE,
)


def _parse_date(text: str) -> Optional[datetime]:
    m = _DATE_RE.search(text or "")
    if not m:
        return None
    month = _MONTHS.get(m.group(2).lower())
    if month is None:
        return None
    try:
        return datetime(int(m.group(3)), month, int(m.group(1)), tzinfo=timezone.utc)
    except ValueError:
        return None


def _select_item_pdf(isoup: BeautifulSoup, item_url: str, kw: str) -> Optional[str]:
    """Pick the epidemiological-update PDF from a WHO item page.

    An item page can link several PDFs (annexes, translations, briefs), so
    taking the first ``.pdf`` blindly can extract the wrong document (issue
    #59). Score candidates by how strongly they look like the situation
    report/epidemiological update and prefer the highest scorer, falling back
    to the first PDF in document order so behaviour is unchanged when nothing
    is distinguishable.
    """
    best_url: Optional[str] = None
    best_score = -1
    for idx, a in enumerate(isoup.find_all("a", href=True)):
        cand = urljoin(item_url, a["href"])
        low = cand.lower()
        if ".pdf" not in low:
            continue
        text = a.get_text(" ", strip=True).lower()

        score = 0
        # The Docling allowlist and the cumulative tables live on the
        # ``situation-reports`` path — strongest signal.
        if "situation-report" in low:
            score += 4
        if "epidemiological-update" in low or "epidemiological update" in text:
            score += 3
        if kw in low or kw in text:
            score += 2
        # WHO serves the real report PDFs from the CDN host.
        if urlparse(low).netloc.endswith("cdn.who.int"):
            score += 1

        # Strictly-greater keeps the first candidate on ties, preserving the
        # previous "first PDF" fallback for indistinguishable pages.
        if score > best_score:
            best_score = score
            best_url = cand

    return best_url


def _get(url: str, cfg: ExtractionConfig) -> Optional[curl_requests.Response]:
    try:
        return curl_requests.get(
            url,
            timeout=max(cfg.fetch_timeout_seconds, 30.0),
            impersonate=cfg.impersonate,
            allow_redirects=True,
        )
    except Exception as exc:  # noqa: BLE001 - network best-effort
        logger.info("WHO hub fetch failed for %s: %s", url, exc)
        return None


def fetch_who_hub_latest_pdf(
    url: str,
    keyword: str,
    *,
    config: ExtractionConfig | None = None,
    as_of_date: datetime | None = None,
    **_ignored,
) -> FetchResult | None:
    """Resolve a WHO hub to its latest dated item PDF at-or-before the cutoff.

    Returns a ``FetchResult`` holding PDF bytes, or ``None`` to fall back to the
    generic fetch of the hub itself.
    """
    cfg = config or ExtractionConfig()
    fetched_at = datetime.now(timezone.utc)
    cutoff = as_of_date or fetched_at
    kw = keyword.lower()

    hub = _get(url, cfg)
    if hub is None or hub.status_code != 200:
        return None

    soup = BeautifulSoup(hub.text, "html.parser")

    # Collect dated item pages whose link text mentions the keyword.
    candidates: list[tuple[datetime, str]] = []
    for a in soup.find_all("a", href=True):
        item_url = urljoin(url, a["href"])
        if "/publications/" not in item_url:
            continue
        text = a.get_text(" ", strip=True)
        if kw not in text.lower() and kw not in item_url.lower():
            continue
        dt = _parse_date(text) or _parse_date(item_url.replace("-", " "))
        if dt is None or dt > cutoff:
            continue
        candidates.append((dt, item_url))

    if not candidates:
        return None

    # Newest item at-or-before the cutoff.
    candidates.sort(key=lambda c: c[0], reverse=True)
    _, item_url = candidates[0]

    item = _get(item_url, cfg)
    if item is None or item.status_code != 200:
        return None
    isoup = BeautifulSoup(item.text, "html.parser")

    pdf_url = _select_item_pdf(isoup, item_url, kw)
    if not pdf_url:
        return None

    pdf = _get(pdf_url, cfg)
    if pdf is None or pdf.status_code != 200 or not pdf.content:
        return None
    if len(pdf.content) > cfg.fetch_max_bytes:
        return None

    return FetchResult(
        url=pdf_url,
        final_url=str(pdf.url),
        status_code=pdf.status_code,
        content_type="application/pdf",
        content_bytes=pdf.content,
        fetched_at=fetched_at,
        error=None,
    )
