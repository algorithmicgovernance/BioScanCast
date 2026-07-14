"""Tests for the ECDC chikungunya custom scraper (``ecdc_chikungunya``).

No network: an injected ``html_getter`` returns fixture HTML shaped like the live
``chik-weekly.ecdc.europa.eu`` EU/EEA seasonal report. Assertions target the
rendered prose (``content_bytes``) — the point of the scraper is to surface the
EU/EEA **country count** (q19's anchor) at the correct scope, both when the season
is populated and when it is pre-season empty (correct answer: 0).
"""

from __future__ import annotations

from datetime import datetime, timezone

from bioscancast.stages.extraction.custom_scrapers import ecdc_chikungunya

_PAGE = "https://www.ecdc.europa.eu/en/chikungunya-monthly"

# Populated shape (mirrors the 2025 archive summary sentence).
_POPULATED = """
<html><body><p>Seasonal surveillance of chikungunya virus disease in the EU/EEA.
In 2026, two countries in Europe have reported cases of chikungunya virus disease:
France (788) and Italy (384). This week, no new cases were reported.</p></body></html>
"""

# Pre-season empty shape (mirrors the current live report).
_EMPTY = """
<html><body><p>Seasonal surveillance of chikungunya virus disease in the EU/EEA,
weekly report. The next cycle of chikungunya virus disease updates for 2026 will
begin once the first cases are entered into the ECDC EpiPulse Cases platform by
EU/EEA countries. Until then, this section will remain empty.</p></body></html>
"""

# Singular shape: ECDC uses "has reported" with grammatically singular subjects.
_SINGULAR = """
<html><body><p>Seasonal surveillance of chikungunya virus disease in the EU/EEA.
In 2026, one country in Europe has reported cases of chikungunya virus disease:
France (1).</p></body></html>
"""


def _getter(page):
    return lambda url, config: page


def _html(result) -> str:
    assert result is not None
    return result.content_bytes.decode("utf-8")


def test_renders_eu_country_count_at_correct_scope():
    result = ecdc_chikungunya.fetch(_PAGE, html_getter=_getter(_POPULATED))
    html = _html(result)
    assert result.content_type == "text/html"
    # The q19 anchor is the count of EU/EEA countries (2), enumerated from the
    # France/Italy pairs, scoped to locally-acquired (autochthonous) transmission.
    assert "2 EU/EEA countries have reported locally-acquired" in html
    assert "France (788)" in html and "Italy (384)" in html
    assert "2026" in html


def test_preseason_renders_zero():
    result = ecdc_chikungunya.fetch(_PAGE, html_getter=_getter(_EMPTY))
    html = _html(result)
    # Between seasons the correct anchor is 0 EU/EEA countries, not the off-scope
    # non-EU "1" the worldwide overview would otherwise yield.
    assert "0 (zero) EU/EEA countries have reported" in html
    assert "2026" in html


def test_historical_mode_returns_none_to_avoid_leakage():
    result = ecdc_chikungunya.fetch(
        _PAGE,
        as_of_date=datetime(2026, 3, 1, tzinfo=timezone.utc),
        html_getter=_getter(_POPULATED),
    )
    assert result is None


def test_missing_or_unparseable_falls_back():
    assert ecdc_chikungunya.fetch(_PAGE, html_getter=lambda url, config: None) is None
    assert ecdc_chikungunya.fetch(
        _PAGE, html_getter=lambda url, config: "<html><body>unrelated</body></html>"
    ) is None


def test_handles_singular_has_reported():
    """ECDC uses singular 'has reported' when exactly one country is affected."""
    result = ecdc_chikungunya.fetch(_PAGE, html_getter=_getter(_SINGULAR))
    html = _html(result)
    assert "1 EU/EEA country have reported locally-acquired" in html
    assert "France (1)" in html
    assert "2026" in html
