"""Custom scraper for PAHO dengue (``paho_dengue``).

The PAHO dengue landing/arbo-portal pages are dashboard gateways that carry only
programme prose, so dengue questions extracted 0 records. PAHO publishes dated
"Epidemiological Update - Dengue in the Americas Region" documents on its
epidemiological-alerts-and-updates listing; each links a PDF whose tables hold
the Americas cumulative suspected/confirmed case totals. This resolves the
listing to the latest dengue update at-or-before the cutoff and returns its PDF.

Reuses the tested primitives from :mod:`paho_mpox` (curl fetch, date parsing,
PDF-link extraction) so only the listing-filter differs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from bioscancast.stages.extraction.config import ExtractionConfig
from bioscancast.stages.extraction.custom_scrapers.paho_mpox import (
    _extract_pdf_url,
    _fetch_bytes,
    _parse_date,
)
from bioscancast.stages.extraction.fetcher import FetchResult


def _collect_dengue_documents(listing_html: str, listing_url: str) -> list[tuple[datetime, str]]:
    """Return (date, document_url) for dated dengue updates on the listing."""
    soup = BeautifulSoup(listing_html, "html.parser")
    out: list[tuple[datetime, str]] = []
    for a in soup.find_all("a", href=True):
        abs_url = urljoin(listing_url, a["href"])
        if "/documents/" not in abs_url:
            continue
        text = a.get_text(" ", strip=True)
        if "dengue" not in (text + abs_url).lower():
            continue
        dt = _parse_date(text) or _parse_date(abs_url.replace("-", " "))
        if dt is not None:
            out.append((dt, abs_url))
    return out


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

    listing_bytes, listing_final, _s, _ct, err = _fetch_bytes(url, cfg)
    if err is not None or listing_bytes is None:
        return None
    docs = _collect_dengue_documents(
        listing_bytes.decode("utf-8", errors="replace"), listing_final
    )
    docs = [(d, u) for d, u in docs if d <= cutoff]
    if not docs:
        return None

    docs.sort(key=lambda item: item[0], reverse=True)
    _, doc_url = docs[0]

    doc_bytes, doc_final, _s2, _ct2, err2 = _fetch_bytes(doc_url, cfg)
    if err2 is not None or doc_bytes is None:
        return None
    pdf_url = _extract_pdf_url(doc_bytes.decode("utf-8", errors="replace"), doc_final)
    if not pdf_url:
        return None

    pdf_bytes, pdf_final, pdf_status, pdf_ct, pdf_err = _fetch_bytes(pdf_url, cfg)
    if pdf_err is not None or pdf_bytes is None:
        return None

    return FetchResult(
        url=pdf_url,
        final_url=pdf_final,
        status_code=pdf_status,
        content_type=pdf_ct or "application/pdf",
        content_bytes=pdf_bytes,
        fetched_at=fetched_at,
        error=None,
    )
