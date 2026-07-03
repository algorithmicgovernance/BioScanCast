from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import parse_qs, urljoin, urlparse

from bs4 import BeautifulSoup
from curl_cffi import requests as curl_requests

from bioscancast.stages.extraction.config import ExtractionConfig
from bioscancast.stages.extraction.fetcher import FetchResult

# English month names used on PAHO pages, both abbreviated and full.
_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

_DATE_RE = re.compile(
    r"\b(\d{1,2})\s+"
    r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+(\d{4})\b",
    flags=re.IGNORECASE,
)


def _normalize_content_type(header_value: Optional[str]) -> Optional[str]:
    if not header_value:
        return None
    return header_value.split(";")[0].strip().lower()


def _fetch_bytes(url: str, cfg: ExtractionConfig) -> tuple[Optional[bytes], str, Optional[int], Optional[str], Optional[str]]:
    """Fetch bytes with the same curl impersonation strategy as core fetcher.

    Returns: (content_bytes, final_url, status_code, content_type, error)
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


def _parse_date(text: str) -> Optional[datetime]:
    match = _DATE_RE.search(text)
    if not match:
        return None
    day = int(match.group(1))
    month_raw = match.group(2).lower()
    year = int(match.group(3))
    month = _MONTHS.get(month_raw)
    if month is None:
        return None
    try:
        return datetime(year, month, day, tzinfo=timezone.utc)
    except ValueError:
        return None


def _collect_report_candidates(listing_html: str, listing_url: str) -> list[str]:
    """Extract report page URLs from listing HTML."""
    soup = BeautifulSoup(listing_html, "html.parser")
    candidates: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        abs_url = urljoin(listing_url, href)
        if "/en/documents/" not in abs_url:
            continue

        text_blob = " ".join(
            [
                a.get_text(" ", strip=True),
                a.parent.get_text(" ", strip=True) if a.parent else "",
            ]
        )
        text_lower = text_blob.lower()
        if "situation report" not in text_lower:
            continue
        if "mpox" not in text_lower and "monkeypox" not in text_lower:
            continue

        candidates.add(abs_url)

    return sorted(candidates)


def _listing_pages_seed(base_url: str) -> list[str]:
    """Collect listing pages to scan.

    We always include the first page and then any explicit pagination links
    found on that first page (typically page=1, ...).
    """
    return [base_url]


def _extract_pdf_url(report_html: str, report_url: str) -> Optional[str]:
    soup = BeautifulSoup(report_html, "html.parser")

    # Prefer explicit DOWNLOAD button text.
    for a in soup.find_all("a", href=True):
        label = a.get_text(" ", strip=True).lower()
        href = a.get("href", "")
        abs_url = urljoin(report_url, href)
        if "download" in label and abs_url.lower().endswith(".pdf"):
            return abs_url

    # Fallback: any site-hosted PDF on the document page.
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        abs_url = urljoin(report_url, href)
        if abs_url.lower().endswith(".pdf") and "paho.org" in urlparse(abs_url).netloc:
            return abs_url

    return None


def _extract_report_date(report_html: str) -> Optional[datetime]:
    """Extract the report publication date from a PAHO report page."""
    soup = BeautifulSoup(report_html, "html.parser")
    text = soup.get_text(" ", strip=True)
    return _parse_date(text)


def _add_pagination_pages(
    *,
    seed_html: str,
    seed_url: str,
    pages: list[str],
) -> list[str]:
    soup = BeautifulSoup(seed_html, "html.parser")
    seen = set(pages)
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        abs_url = urljoin(seed_url, href)
        parsed = urlparse(abs_url)
        if parsed.path != urlparse(seed_url).path:
            continue
        qs = parse_qs(parsed.query)
        if "page" not in qs:
            continue
        if abs_url not in seen:
            seen.add(abs_url)
            pages.append(abs_url)
    return pages


def fetch(
    url: str,
    *,
    config: ExtractionConfig | None = None,
    as_of_date: datetime | None = None,
) -> FetchResult | None:
    """Resolve PAHO mpox listing URL to latest report PDF and fetch that PDF.

    Returns a FetchResult containing PDF bytes so the existing extraction stage
    can reuse its standard PDF parser/chunking pipeline unchanged.
    """
    cfg = config or ExtractionConfig()
    fetched_at = datetime.now(timezone.utc)
    cutoff = as_of_date or fetched_at

    listing_pages = _listing_pages_seed(url)
    report_urls: set[str] = set()

    # Fetch first page, then discover pagination links from it.
    page0_bytes, page0_final, page0_status, page0_ct, page0_err = _fetch_bytes(
        listing_pages[0], cfg
    )
    if page0_err is not None or page0_bytes is None:
        return None

    page0_html = page0_bytes.decode("utf-8", errors="replace")
    listing_pages = _add_pagination_pages(
        seed_html=page0_html,
        seed_url=page0_final,
        pages=listing_pages,
    )
    report_urls.update(_collect_report_candidates(page0_html, page0_final))

    for listing_page in listing_pages[1:]:
        html_bytes, final_url, _status, _ct, err = _fetch_bytes(listing_page, cfg)
        if err is not None or html_bytes is None:
            continue
        html_text = html_bytes.decode("utf-8", errors="replace")
        report_urls.update(_collect_report_candidates(html_text, final_url))

    if not report_urls:
        return None

    # Resolve each report page to (date, pdf_url), then pick latest <= cutoff.
    resolved_candidates: list[tuple[datetime, str]] = []
    for report_url in sorted(report_urls):
        report_bytes, report_final, _report_status, _report_ct, report_err = _fetch_bytes(
            report_url, cfg
        )
        if report_err is not None or report_bytes is None:
            continue
        report_html = report_bytes.decode("utf-8", errors="replace")
        report_dt = _extract_report_date(report_html)
        pdf_url = _extract_pdf_url(report_html, report_final)
        if report_dt is None or not pdf_url:
            continue
        if report_dt <= cutoff:
            resolved_candidates.append((report_dt, pdf_url))

    if not resolved_candidates:
        return None

    _, selected_pdf_url = max(resolved_candidates, key=lambda item: item[0])

    pdf_bytes, pdf_final, pdf_status, pdf_ct, pdf_err = _fetch_bytes(selected_pdf_url, cfg)
    if pdf_err is not None or pdf_bytes is None:
        return None

    content_type = pdf_ct or "application/pdf"
    if not content_type.lower().endswith("pdf") and not pdf_final.lower().endswith(".pdf"):
        # Unexpected content type (e.g., HTML interstitial). Let default fetcher try.
        return None

    return FetchResult(
        url=selected_pdf_url,
        final_url=pdf_final,
        status_code=pdf_status,
        content_type=content_type,
        content_bytes=pdf_bytes,
        fetched_at=fetched_at,
        error=None,
    )