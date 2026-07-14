"""Tests for the CDC measles custom scraper (``cdc_measles``).

No network: the scraper accepts an injected ``json_getter`` returning a fixture
dict shaped like the live ``measles_hosp.json`` feed. Assertions target the
rendered HTML summary (``content_bytes``) that the HTML extraction pipeline then
consumes — the point of the scraper is to surface the JS-injected **deaths**
figure (absent from the statically served page) as extractable prose.
"""

from __future__ import annotations

from datetime import datetime, timezone

from bioscancast.stages.extraction.custom_scrapers import cdc_measles

_PAGE = "https://www.cdc.gov/measles/data-research/index.html"

# Shape mirrors the live feed: each field is a single-element list.
_FEED = {
    "2026": {
        "total_cases": ["2,170"],
        "total_deaths": ["0"],
        "deaths_sentence": ["There have been 0 confirmed deaths from measles in 2026."],
    },
    "2025": {
        "total_cases": ["2,289"],
        "total_deaths": ["3"],
        "deaths_sentence": ["There were 3 confirmed deaths from measles in 2025."],
    },
}


def _getter(feed):
    return lambda url, config: feed


def _html(result) -> str:
    assert result is not None
    return result.content_bytes.decode("utf-8")


def test_renders_current_year_death_count_as_prose():
    result = cdc_measles.fetch(_PAGE, json_getter=_getter(_FEED))
    html = _html(result)
    assert result.content_type == "text/html"
    # The death figure that is JS-injected (and thus missing from a static
    # scrape) is now present as an unambiguous sentence.
    assert "0 confirmed deaths from measles in 2026" in html
    # Case count is surfaced too, corroborating the prose figure q14 uses.
    assert "2,170 confirmed measles cases" in html


def test_historical_mode_returns_none_to_avoid_leakage():
    # The feed carries only current per-year aggregates with no date history,
    # so replay mode must fall back rather than serve post-cutoff values.
    result = cdc_measles.fetch(
        _PAGE,
        as_of_date=datetime(2026, 3, 1, tzinfo=timezone.utc),
        json_getter=_getter(_FEED),
    )
    assert result is None


def test_missing_feed_falls_back():
    assert cdc_measles.fetch(_PAGE, json_getter=lambda url, config: None) is None
    assert cdc_measles.fetch(_PAGE, json_getter=lambda url, config: {}) is None
