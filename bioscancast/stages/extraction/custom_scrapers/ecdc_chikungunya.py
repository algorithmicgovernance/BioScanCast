"""Custom scraper for ECDC chikungunya (``ecdc_chikungunya``).

q19 asks how many **EU/EEA countries** report locally-acquired (autochthonous)
chikungunya in the year. The registered source URL is the *worldwide* monthly
overview (``ecdc.europa.eu/en/chikungunya-monthly``), which discusses non-EU
countries (Saint Lucia, Peru, Suriname, ...) — so the generic extractor grabs an
off-scope "1" from, e.g., a Saint Lucia sentence. The EU/EEA figure lives instead
in ECDC's *seasonal surveillance* report, a client-side ``htmlwidgets``/DataTables
page at ``chik-weekly.ecdc.europa.eu`` whose static cells are otherwise empty.

That report carries one clean prose summary, e.g.:

    "... two countries in Europe have reported cases of chikungunya virus disease:
     France (788) and Italy (384)."

so the country **count** (the q19 anchor) is enumerable from ``Country (n)`` pairs.
Between seasons the report is intentionally empty — "this section will remain empty"
until the first cases are entered — in which case the correct current anchor is
**0 EU/EEA countries**. We render either case as compact HTML prose the existing
extractor consumes, pinning the scope to *locally-acquired EU/EEA* so it no longer
mis-fires on the worldwide page.

Returns ``None`` (fall back to the generic fetch) on any failure, and in
historical-replay mode — the report is a current-snapshot with no in-page date
history to cut on, so serving it under an ``as_of_date`` would leak.
"""

from __future__ import annotations

import html
import logging
import re
from datetime import datetime, timezone

from bioscancast.stages.extraction.config import ExtractionConfig
from bioscancast.stages.extraction.fetcher import FetchResult

logger = logging.getLogger(__name__)

# The EU/EEA seasonal-surveillance report (client-side rendered). Host-pinned:
# this scraper is specific to the ECDC chikungunya source; the ``url`` passed in
# is the human-facing worldwide-overview page.
_REPORT_URL = "https://chik-weekly.ecdc.europa.eu/"

# "this section will remain empty" between seasons; also the announce sentence.
_EMPTY_MARKERS = ("will remain empty", "will begin once the first")

# "<n> countr(y|ies) in Europe have reported cases of chikungunya"
_REPORT_SENTENCE = re.compile(
    r"(\w+)\s+countr(?:y|ies)\s+in\s+Europe\s+have\s+reported\s+cases\s+of\s+chikungunya",
    re.IGNORECASE,
)
# "France (788)" / "Czechia (12)" — a country name followed by a case count.
_COUNTRY_PAIR = re.compile(r"([A-Z][A-Za-zÀ-ſ'’.\- ]*?)\s*\(([\d, ]+)\)")

_NUM_WORDS = {
    "no": 0, "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12,
}


def _strip_tags(raw: str) -> str:
    t = re.sub(r"<script[\s\S]*?</script>", " ", raw, flags=re.IGNORECASE)
    t = re.sub(r"<style[\s\S]*?</style>", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _detect_year(text: str, fallback: int) -> int:
    m = re.search(r"updates for (20\d{2})", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"\bin (20\d{2})\b", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"(20\d{2})", text)
    if m:
        return int(m.group(1))
    return fallback


def _get_html(url: str, cfg: ExtractionConfig):
    try:
        from curl_cffi import requests as curl_requests

        resp = curl_requests.get(
            url,
            timeout=max(cfg.fetch_timeout_seconds, 30.0),
            impersonate=cfg.impersonate,
            allow_redirects=True,
        )
    except Exception as exc:  # noqa: BLE001 - network best-effort
        logger.info("ECDC chikungunya fetch failed for %s: %s", url, exc)
        return None
    if resp.status_code != 200 or not resp.text:
        return None
    return resp.text


def _parse(text: str):
    """Return (count, breakdown_pairs) or (0, []) pre-season or None if unparseable."""
    sentence = _REPORT_SENTENCE.search(text)
    if sentence is not None:
        # Bound pair extraction to the reporting clause (up to "This week"/period).
        tail = text[sentence.end(): sentence.end() + 400]
        tail = re.split(r"This week|\. [A-Z]", tail, maxsplit=1)[0]
        pairs = [(n.strip(), c.replace(" ", "").replace(",", ""))
                 for n, c in _COUNTRY_PAIR.findall(tail)]
        token = sentence.group(1).lower()
        stated = _NUM_WORDS.get(token)
        if stated is None and token.isdigit():
            stated = int(token)
        count = stated if stated is not None else len(pairs)
        # Prefer the enumerated list when it disagrees and is non-empty.
        if pairs and count != len(pairs):
            count = len(pairs)
        return count, pairs
    if any(mk in text.lower() for mk in _EMPTY_MARKERS):
        return 0, []
    return None


def fetch(
    url: str,
    *,
    config: ExtractionConfig | None = None,
    as_of_date: datetime | None = None,
    region: str | None = None,
    question_text: str | None = None,
    html_getter=None,
) -> FetchResult | None:
    # Live-only: the report is a current snapshot with no in-page date history,
    # so serving it in replay mode would leak post-cutoff values.
    if as_of_date is not None:
        return None

    cfg = config or ExtractionConfig()
    fetched_at = datetime.now(timezone.utc)

    getter = html_getter or _get_html
    raw = getter(_REPORT_URL, cfg)
    if not raw:
        return None

    text = _strip_tags(raw)
    parsed = _parse(text)
    if parsed is None:
        return None
    count, pairs = parsed
    year = _detect_year(text, fetched_at.year)

    if count == 0:
        body = (
            f"<p>As of {fetched_at.date().isoformat()}, a cumulative total of "
            f"0 (zero) EU/EEA countries have reported locally-acquired "
            f"(autochthonous) chikungunya virus disease in {year} (year-to-date). "
            f"The {year} seasonal surveillance cycle in the EU/EEA has not yet "
            f"begun (no cases entered into ECDC EpiPulse).</p>"
        )
    else:
        if pairs:
            breakdown = ", ".join(
                f"{html.escape(n)} ({html.escape(c)})" for n, c in pairs
            )
            detail = f" Countries reporting: {breakdown}."
        else:
            detail = ""
        body = (
            f"<p>As of {fetched_at.date().isoformat()}, a cumulative total of "
            f"{count} EU/EEA countr{'y' if count == 1 else 'ies'} have reported "
            f"locally-acquired (autochthonous) chikungunya virus disease in "
            f"{year} (year-to-date).{detail}</p>"
        )

    rendered = (
        "<html><head><meta charset='utf-8'>"
        "<title>ECDC — Seasonal surveillance of chikungunya virus disease in the "
        "EU/EEA</title></head><body>"
        "<h1>ECDC — Seasonal surveillance of chikungunya virus disease in the "
        "EU/EEA (weekly report)</h1>"
        f"<p>Source: {html.escape(url)} "
        f"(EU/EEA seasonal report {html.escape(_REPORT_URL)}), retrieved "
        f"{fetched_at.date().isoformat()}.</p>"
        f"{body}"
        "<p>Note: this custom scraper renders ECDC's client-side EU/EEA seasonal "
        "surveillance report into compact HTML because the count of EU/EEA "
        "countries with locally-acquired chikungunya is injected client-side and "
        "is absent from the statically served page. The count is a cumulative "
        "year-to-date number of EU/EEA countries reporting autochthonous "
        "transmission.</p>"
        "</body></html>"
    ).encode("utf-8")

    return FetchResult(
        url=_REPORT_URL,
        final_url=_REPORT_URL,
        status_code=200,
        content_type="text/html",
        content_bytes=rendered,
        fetched_at=fetched_at,
        error=None,
    )
