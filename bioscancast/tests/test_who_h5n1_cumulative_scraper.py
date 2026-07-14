"""Tests for the WHO cumulative A(H5N1) custom scraper (``who_h5n1_cumulative``).

No network: the scraper accepts an injected ``document_getter`` returning
``(rows, full_text, pdf_url)`` shaped exactly like PyMuPDF ``find_tables().extract()``
output on the live cumulative-table PDF (captured 2026-07-07). Assertions target the
rendered prose (``content_bytes``) the HTML extraction pipeline then consumes: the
point of the scraper is to turn the column-less PDF table into unambiguous,
scope-matched anchors (2026 year-to-date count for q6, subtype A(H5N1) for q8).
"""

from __future__ import annotations

from datetime import datetime, timezone

from bioscancast.stages.extraction.custom_scrapers import who_h5n1_cumulative as mod

_PAGE = (
    "https://www.who.int/teams/global-influenza-programme/avian-influenza/"
    "avian-a-h5n1-virus"
)
_PDF = (
    "https://cdn.who.int/media/docs/default-source/influenza/"
    "h5n1-human-case-cumulative-table/cumulative.pdf"
)

# Header + a representative subset of country rows + the merged Total row, exactly
# as PyMuPDF ``find_tables().extract()`` returns them for the 7 July 2026 PDF. The
# 2026 (year-to-date) column is Bangladesh 2, Cambodia 4, India 1 -> 7 cases, 2
# deaths, matching the Total row's "7 2"; grand total "1000 479".
_HEADER = [
    "Country", "2003-2009*\ncases deaths", None, "2010-2014*\ncases deaths", None,
    "2015-2019*\ncases deaths", None, "2020-2024*\ncases deaths", None,
    "2025\ncases deaths", None, "2026\ncases deaths", None, "Total\ncases deaths",
    None,
]
_ROWS = [
    _HEADER,
    ["Australia", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "1", "0"],
    ["Bangladesh", "1", "0", "6", "1", "1", "0", "0", "0", "3", "0", "2", "1", "13", "2"],
    ["Cambodia", "9", "7", "47", "30", "0", "0", "16", "6", "18", "9", "4", "1", "94", "53"],
    ["China", "38", "25", "9", "5", "6", "1", "3", "1", "1", "0", "0", "0", "57", "32"],
    ["India", "0", "0", "0", "0", "0", "0", "1", "1", "2", "2", "1", "0", "4", "3"],
    ["United States of America**", "0", "0", "0", "0", "0", "0", "68", "1", "3", "0", "0", "0", "71", "1"],
    ["Total", "468 282", "�", "233 125", "�", "160 48", "�",
     "102 10", "�", "30 12", "�", "7 2", "�", "1000 479", "�"],
]
_TEXT = "Source: WHO/GIP, data in HQ as of 7 July 2026."


def _getter(rows=_ROWS, text=_TEXT, pdf=_PDF):
    return lambda url, cfg, as_of_date: (rows, text, pdf)


def _html(result) -> str:
    assert result is not None
    return result.content_bytes.decode("utf-8")


def test_renders_cumulative_and_current_year_anchors():
    result = mod.fetch(_PAGE, document_getter=_getter())
    html = _html(result)
    assert result.content_type == "text/html"
    # Cumulative total (context / base rate).
    assert "1000 cases, including 479 deaths" in html
    # 2026 year-to-date count is the q6 anchor.
    assert "In 2026 (year-to-date) 7 confirmed human A(H5N1) cases and 2 deaths" in html
    # Prior-year base rate.
    assert "In 2025 the total was 30 cases and 12 deaths" in html
    # Report date surfaced from the PDF prose.
    assert "As of 7 July 2026" in html


def test_scope_pins_subtype_for_q8():
    # q8 needs the predominant subtype; the prose must name A(H5N1) explicitly.
    html = _html(mod.fetch(_PAGE, document_getter=_getter()))
    assert "A(H5N1)" in html
    assert "predominant subtype" in html


def test_current_year_country_breakdown():
    html = _html(mod.fetch(_PAGE, document_getter=_getter()))
    # Only countries with 2026 cases > 0 appear (Bangladesh 2, Cambodia 4, India 1);
    # a zero-in-2026 country (China) does not.
    assert "Bangladesh (2 cases)" in html
    assert "Cambodia (4 cases)" in html
    assert "India (1 case)" in html
    assert "China" not in html.split("Countries reporting", 1)[1]


def test_current_year_column_is_dynamic_not_hardcoded():
    # When WHO adds a 2027 column, that becomes the current-year anchor without a
    # code change. Append a 2027 column to header + rows + total.
    header = _HEADER[:-2] + ["2027\ncases deaths", None] + _HEADER[-2:]
    rows = [header]
    for r in _ROWS[1:-1]:
        rows.append(r[:-2] + ["5", "1"] + r[-2:])
    total = _ROWS[-1][:-2] + ["5 1", "�"] + _ROWS[-1][-2:]
    rows.append(total)
    html = _html(mod.fetch(_PAGE, document_getter=_getter(rows=rows)))
    assert "In 2027 (year-to-date) 5 confirmed human A(H5N1) cases" in html
    assert "In 2026 the total was" in html  # prior year rolls to 2026


def test_historical_snapshot_is_served_not_blocked():
    # Unlike cdc_measles/ecdc, the resolved WHO item is dated, so replay mode is
    # leakage-safe (the getter/resolver picks an at-or-before-cutoff snapshot).
    result = mod.fetch(
        _PAGE,
        as_of_date=datetime(2026, 8, 1, tzinfo=timezone.utc),
        document_getter=_getter(),
    )
    assert result is not None
    assert "1000 cases" in _html(result)


def test_getter_returns_none_falls_back():
    assert mod.fetch(_PAGE, document_getter=lambda url, cfg, aod: None) is None


def test_unparseable_table_falls_back():
    # No 'Country' header row -> cannot map columns -> fall back to generic fetch.
    junk = [["foo", "bar"], ["1", "2"]]
    assert mod.fetch(_PAGE, document_getter=_getter(rows=junk)) is None


def test_missing_total_row_sums_country_rows():
    # Drop the Total row: cumulative + current-year totals must still be computed
    # by summing the country rows.
    rows = _ROWS[:-1]  # header + country rows, no Total
    html = _html(mod.fetch(_PAGE, document_getter=_getter(rows=rows)))
    # Country-row sum of the Total column: 1+13+94+57+4+71 = 240.
    assert "240 cases" in html
    # 2026 column sum: Bangladesh 2 + Cambodia 4 + India 1 = 7.
    assert "In 2026 (year-to-date) 7 confirmed human A(H5N1) cases" in html
