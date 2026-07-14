"""Custom scraper for the WHO cumulative A(H5N1) human-case table (``who_h5n1_cumulative``).

BFG q6 (how many human H5 cases are reported to WHO in the Jul-Dec 2026 window)
and q8 (predominant H5 subtype) need the count of confirmed human A(H5N1) cases
and its per-year trend. The ``who_h5_hai`` monthly assessment reports only the
*reporting period's* events (currently an A(H9N2) cluster) and merely **links** to
the cumulative figure; the registered ``avian-a-h5n1-virus`` landing page (~1.3 MB)
likewise only carries the *title* of that figure as a link. The number itself lives
in a dated *"Cumulative number of confirmed human cases for avian influenza A(H5N1)
reported to WHO, 2003-<year>"* item whose PDF holds a wide per-country x per-period
table (2003-2009, 2010-2014, ..., <prior year>, <current year>, Total).

That PDF path (``cdn.who.int/.../influenza/h5n1-human-case-cumulative-table/``) is
not on the Docling allowlist, and a plain text extraction of the table linearises
into a column-less stream of digits that the insight stage cannot map back to a
year. So - like ``cdc_measles`` / ``ecdc_chikungunya`` - we resolve the dated PDF
(reusing the ``who_h5_hai`` hub->item->PDF resolver), parse its table with the same
PyMuPDF ``find_tables`` the PDF parser uses, and render the scope-matched anchors as
compact prose the existing HTML extractor consumes:

  * the cumulative 2003-<year> total (context / base rate),
  * the **current-year** count (the year-to-date anchor for q6),
  * the prior-year count (a base rate for the window), and
  * the countries reporting in the current year,

all pinned to subtype **A(H5N1)** (the q8 anchor). This complements ``who_h5_hai``
(kept for recent-event context) rather than replacing it.

Unlike the CDC/ECDC current-snapshot feeds, the resolved WHO item is *dated*, so the
resolver's at-or-before-cutoff selection makes this leakage-safe under ``as_of_date``
(same guarantee as ``who_cholera`` / ``who_h5_hai``): it only ever surfaces a
snapshot published on or before the cutoff. Returns ``None`` (fall back to the
generic fetch) on any resolution/parse failure.
"""

from __future__ import annotations

import html
import logging
import re
from datetime import datetime, timezone

from bioscancast.stages.extraction.config import ExtractionConfig
from bioscancast.stages.extraction.custom_scrapers._who_hub_common import (
    fetch_who_hub_latest_pdf,
)
from bioscancast.stages.extraction.fetcher import FetchResult

logger = logging.getLogger(__name__)

# "human case" links on the landing page all carry "cumulative" in both their
# text and slug; this selects the dated cumulative-table item out of the mixed
# publication links (FAO/WOAH assessments, vaccine options, ...) on the page.
_HUB_KEYWORD = "cumulative"

# "Source: WHO/GIP, data in HQ as of 7 July 2026."
_AS_OF_RE = re.compile(r"as of\s+(\d{1,2}\s+[A-Za-z]+\s+20\d{2})", re.IGNORECASE)

# A pure four-digit year column label (e.g. "2026"), distinct from the multi-year
# aggregate labels ("2003-2009").
_YEAR_RE = re.compile(r"^(20\d{2})$")


def _int(value) -> int | None:
    """Parse a table cell to an int, tolerating thousands separators and junk."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    # WHO PDFs use regular / non-breaking / narrow spaces (and commas) as
    # thousands separators; strip any whitespace or comma between the digits.
    digits = re.sub(r"[\s,]", "", text)
    if not digits.isdigit():
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def _clean_label(cell) -> str:
    """Reduce a header cell ("2026\\ncases deaths", "2003-2009*\\n...") to its label."""
    if cell is None:
        return ""
    first = str(cell).splitlines()[0] if str(cell).strip() else ""
    return first.strip().rstrip("*").strip()


def _pair(row: list, cases_idx: int) -> tuple[int | None, int | None]:
    """Return (cases, deaths) for the column whose *cases* header sits at ``cases_idx``.

    Data rows keep cases and deaths in separate adjacent cells ("7", "2"); the
    ``Total`` row merges them into the cases cell ("7 2") with the deaths cell
    holding a replacement glyph. Handle both.
    """
    if cases_idx >= len(row):
        return None, None
    raw = str(row[cases_idx] or "").strip()
    merged = raw.split()
    if len(merged) >= 2 and _int(merged[0]) is not None and _int(merged[1]) is not None:
        return _int(merged[0]), _int(merged[1])
    deaths = row[cases_idx + 1] if cases_idx + 1 < len(row) else None
    return _int(raw), _int(deaths)


def _find_header(rows: list[list]) -> int | None:
    for i, row in enumerate(rows):
        if row and str(row[0] or "").strip().lower() == "country":
            return i
    return None


def _parse_table(rows: list[list] | None, full_text: str) -> dict | None:
    """Parse the cumulative-table rows into scope-matched anchors, or ``None``.

    Returns a dict: ``as_of`` (str|None), ``current_year`` (str), plus per-period
    (cases, deaths) tuples under ``cumulative`` / ``current`` / ``prior`` and a
    ``countries`` list of (name, cases, deaths) for the current year.
    """
    if not rows:
        return None
    hi = _find_header(rows)
    if hi is None:
        return None
    header = rows[hi]

    # Each truthy header cell (past col 0) labels the *cases* column at its index;
    # deaths sit at index+1. Ordered so we can zip labels to the Total row too.
    periods: list[tuple[str, int]] = []
    for j, cell in enumerate(header):
        if j == 0:
            continue
        label = _clean_label(cell)
        if label:
            periods.append((label, j))
    if not periods:
        return None
    idx_by_label = {label: j for label, j in periods}

    # Current year = newest pure-year column; prior year = the one before it.
    year_values = sorted(
        int(m.group(1)) for lbl, _ in periods if (m := _YEAR_RE.match(lbl))
    )
    if not year_values:
        return None
    current_year = str(year_values[-1])
    prior_year = str(year_values[-2]) if len(year_values) >= 2 else None

    total_row = next(
        (r for r in rows if r and str(r[0] or "").strip().lower() == "total"), None
    )

    def _column_total(label: str | None) -> tuple[int | None, int | None]:
        if label is None:
            return None, None
        idx = idx_by_label.get(label)
        if idx is None:
            return None, None
        if total_row is not None:
            cases, deaths = _pair(total_row, idx)
            if cases is not None:
                return cases, deaths
        # Fall back to summing the country rows for this column.
        c_sum = d_sum = 0
        seen = False
        for r in rows[hi + 1 :]:
            if not r or str(r[0] or "").strip().lower() in ("", "total", "country"):
                continue
            c, d = _pair(r, idx)
            if c is not None:
                c_sum += c
                seen = True
            if d is not None:
                d_sum += d
        return (c_sum, d_sum) if seen else (None, None)

    total_label = next(
        (lbl for lbl, _ in periods if lbl.lower() == "total"), "Total"
    )
    cumulative = _column_total(total_label)
    current = _column_total(current_year)
    prior = _column_total(prior_year) if prior_year else (None, None)

    if cumulative[0] is None or cumulative[0] <= 0:
        return None

    # Countries reporting in the current year (cases > 0).
    countries: list[tuple[str, int, int]] = []
    cur_idx = idx_by_label.get(current_year)
    if cur_idx is not None:
        for r in rows[hi + 1 :]:
            if not r or str(r[0] or "").strip().lower() in ("", "total", "country"):
                continue
            name = re.sub(r"\s+", " ", str(r[0] or "")).strip().rstrip("*").strip()
            cases, deaths = _pair(r, cur_idx)
            if cases and cases > 0:
                countries.append((name, cases, deaths or 0))

    as_of_match = _AS_OF_RE.search(full_text or "")
    return {
        "as_of": as_of_match.group(1) if as_of_match else None,
        "current_year": current_year,
        "prior_year": prior_year,
        "cumulative": cumulative,
        "current": current,
        "prior": prior,
        "countries": countries,
    }


def _fmt(pair: tuple[int | None, int | None]) -> tuple[str, str]:
    cases, deaths = pair
    return (
        str(cases) if cases is not None else "an unknown number of",
        str(deaths) if deaths is not None else "an unknown number of",
    )


def _render_html(
    parsed: dict, *, source_url: str, pdf_url: str, fetched_at: datetime
) -> bytes:
    year = parsed["current_year"]
    as_of = parsed["as_of"] or f"the latest WHO cumulative update ({year})"
    cum_c, cum_d = _fmt(parsed["cumulative"])
    cur_c, cur_d = _fmt(parsed["current"])

    blocks = [
        f"<h2>WHO cumulative confirmed human A(H5N1) cases, 2003-{year}</h2>",
        "<p>",
        f"As of {html.escape(as_of)}, the cumulative number of confirmed human "
        f"cases of avian influenza A(H5N1) reported to WHO since 2003 is "
        f"{html.escape(cum_c)} cases, including {html.escape(cum_d)} deaths. ",
        f"In {html.escape(year)} (year-to-date) {html.escape(cur_c)} confirmed "
        f"human A(H5N1) cases and {html.escape(cur_d)} deaths have been reported "
        f"to WHO globally. ",
    ]
    if parsed.get("prior_year") and parsed["prior"][0] is not None:
        pri_c, pri_d = _fmt(parsed["prior"])
        blocks.append(
            f"In {html.escape(parsed['prior_year'])} the total was "
            f"{html.escape(pri_c)} cases and {html.escape(pri_d)} deaths. "
        )
    blocks.append("</p>")

    if parsed["countries"]:
        listed = ", ".join(
            f"{html.escape(n)} ({c} case{'s' if c != 1 else ''})"
            for n, c, _ in parsed["countries"]
        )
        blocks.append(
            f"<p>Countries reporting confirmed human A(H5N1) cases in "
            f"{html.escape(year)}: {listed}.</p>"
        )

    rendered = (
        "<html><head><meta charset='utf-8'>"
        "<title>WHO - Cumulative confirmed human cases of avian influenza "
        "A(H5N1) reported to WHO</title></head><body>"
        "<h1>WHO - Cumulative number of confirmed human cases of avian influenza "
        "A(H5N1) reported to WHO</h1>"
        f"<p>Source: {html.escape(source_url)} (cumulative-table PDF "
        f"{html.escape(pdf_url)}), retrieved {fetched_at.date().isoformat()}.</p>"
        + "".join(blocks)
        + "<p>Note: this custom scraper renders the WHO cumulative A(H5N1) "
        "human-case table into compact prose because the count lives only in a "
        "linked PDF whose wide per-country table does not extract cleanly. Counts "
        "are laboratory-confirmed human cases of avian influenza A(H5N1) reported "
        "to WHO (dates refer to onset of illness; the case total includes deaths; "
        "for the United States, cases reported as A(H5) are included). A(H5N1) is "
        "the predominant subtype among reported human H5 infections; other H5 "
        "subtypes (e.g. A(H5N6)) are covered by the WHO influenza-at-the-human-"
        "animal-interface assessments.</p>"
        "</body></html>"
    ).encode("utf-8")
    return rendered


def _default_document_getter(
    url: str, cfg: ExtractionConfig, as_of_date: datetime | None
) -> tuple[list[list], str, str] | None:
    """Resolve the dated cumulative PDF and extract (table rows, text, pdf_url)."""
    pdf_result = fetch_who_hub_latest_pdf(
        url, _HUB_KEYWORD, config=cfg, as_of_date=as_of_date
    )
    if pdf_result is None or not pdf_result.content_bytes:
        return None
    try:
        import pymupdf

        doc = pymupdf.open(stream=pdf_result.content_bytes, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 - parser best-effort
        logger.info("WHO H5N1 cumulative PDF open failed for %s: %s", url, exc)
        return None

    rows: list[list] = []
    text_parts: list[str] = []
    try:
        for page in doc:
            text_parts.append(page.get_text())
            if not rows:
                found = page.find_tables()
                tables = getattr(found, "tables", [])
                if tables:
                    rows = tables[0].extract()
    except Exception as exc:  # noqa: BLE001 - parser best-effort
        logger.info("WHO H5N1 cumulative table extract failed for %s: %s", url, exc)
        return None
    finally:
        doc.close()

    if not rows:
        return None
    return rows, "\n".join(text_parts), pdf_result.url


def fetch(
    url: str,
    *,
    config: ExtractionConfig | None = None,
    as_of_date: datetime | None = None,
    region: str | None = None,
    question_text: str | None = None,
    document_getter=None,
) -> FetchResult | None:
    cfg = config or ExtractionConfig()
    fetched_at = datetime.now(timezone.utc)

    # ``document_getter`` is injectable for no-network tests; defaults to the live
    # resolve + PyMuPDF table extraction.
    getter = document_getter or _default_document_getter
    doc = getter(url, cfg, as_of_date)
    if not doc:
        return None
    rows, full_text, pdf_url = doc

    parsed = _parse_table(rows, full_text)
    if parsed is None:
        return None

    rendered = _render_html(
        parsed, source_url=url, pdf_url=pdf_url, fetched_at=fetched_at
    )
    return FetchResult(
        url=pdf_url,
        final_url=pdf_url,
        status_code=200,
        content_type="text/html",
        content_bytes=rendered,
        fetched_at=fetched_at,
        error=None,
    )
