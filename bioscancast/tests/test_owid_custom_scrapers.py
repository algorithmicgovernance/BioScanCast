"""Tests for the OWID custom scrapers (issues #34 / #38).

No network: the scrapers accept an injected ``csv_fetcher`` returning fixture
text. Adapted from the feature-branch ``test_extraction_dashboards.py`` to the
``source_id``-dispatched ``custom_scrapers.<id>.fetch() -> FetchResult`` contract.
Assertions target the rendered HTML summary (``content_bytes``) rather than a
``ParsedContent``, and entity isolation is expressed as "the World cumulative
figure is not contaminated" (the scraper also renders a top-locations table,
so other entities' values legitimately appear elsewhere).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from bioscancast.stages.extraction.config import ExtractionConfig
from bioscancast.stages.extraction.custom_scrapers import (
    ourworldindata_covid,
    ourworldindata_mpox,
)

# A tiny OWID-shaped mpox CSV: World + another entity, a few dates.
_MPOX_CSV = (
    "location,date,iso_code,total_cases,total_deaths,new_cases\n"
    "World,2025-02-20,OWID_WRL,129500.0,250.0,30.0\n"
    "World,2025-02-24,OWID_WRL,129602.0,254.0,20.0\n"
    "World,2025-02-27,OWID_WRL,129603.0,254.0,1.0\n"
    "World,2025-03-05,OWID_WRL,129999.0,260.0,80.0\n"
    "Africa,2025-02-24,OWID_AFR,40000.0,200.0,10.0\n"
    "Africa,2025-03-05,OWID_AFR,41000.0,205.0,20.0\n"
)

_COVID_CSV = (
    "iso_code,continent,location,date,total_cases,total_deaths,new_cases\n"
    "OWID_WRL,,World,2022-12-31,732000000.0,6800000.0,400000.0\n"
    "OWID_WRL,,World,2023-01-01,732363539.0,6810000.0,363539.0\n"
    "OWID_WRL,,World,2023-01-05,733000000.0,6820000.0,200000.0\n"
)


def _mpox_fetch(csv_text: str = _MPOX_CSV):
    return lambda url, config: csv_text


def _asof(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _html(result) -> str:
    assert result is not None
    return result.content_bytes.decode("utf-8")


class TestMpoxCumulativeMetric:
    def test_cutoff_selects_cumulative_value_as_of_date(self):
        result = ourworldindata_mpox.fetch(
            "https://ourworldindata.org/mpox",
            as_of_date=_asof("2025-02-24"),
            csv_fetcher=_mpox_fetch(),
        )
        body = _html(result)
        # The cumulative figure as of the cutoff, not the later 129,999.
        assert "cumulative confirmed cases (World): 129,602" in body
        assert "cumulative deaths (World): 254" in body
        assert "129,999" not in body
        assert result.content_type == "text/html"

    def test_later_cutoff_advances_value(self):
        body = _html(
            ourworldindata_mpox.fetch(
                "https://ourworldindata.org/mpox",
                as_of_date=_asof("2025-02-27"),
                csv_fetcher=_mpox_fetch(),
            )
        )
        assert "cumulative confirmed cases (World): 129,603" in body

    def test_live_mode_uses_latest_row(self):
        body = _html(
            ourworldindata_mpox.fetch(
                "https://ourworldindata.org/mpox",
                as_of_date=None,
                csv_fetcher=_mpox_fetch(),
            )
        )
        assert "cumulative confirmed cases (World): 129,999" in body

    def test_world_figure_not_contaminated_by_other_entity(self):
        # Africa's 40,000 may appear in the top-locations table, but must never
        # be surfaced as the World cumulative figure.
        body = _html(
            ourworldindata_mpox.fetch(
                "https://ourworldindata.org/mpox",
                as_of_date=_asof("2025-02-24"),
                csv_fetcher=_mpox_fetch(),
            )
        )
        assert "cumulative confirmed cases (World): 40,000" not in body
        assert "cumulative confirmed cases (World): 129,602" in body

    def test_emits_prose_and_trend_table(self):
        body = _html(
            ourworldindata_mpox.fetch(
                "https://ourworldindata.org/mpox",
                as_of_date=_asof("2025-03-05"),
                csv_fetcher=_mpox_fetch(),
            )
        )
        assert "<table" in body
        # Trend columns present, and the last World trend row is the cutoff date.
        assert "<th>total_cases</th>" in body
        assert "2025-03-05" in body
        assert "Trend summary (total_cases, World):" in body
        assert "Linear fit on recent cumulative trend (total_cases" in body
        assert "Incident trend stats (new_cases, World):" in body

    def test_returns_none_when_no_rows_before_cutoff(self):
        assert (
            ourworldindata_mpox.fetch(
                "https://ourworldindata.org/mpox",
                as_of_date=_asof("2020-01-01"),
                csv_fetcher=_mpox_fetch(),
            )
            is None
        )

    def test_returns_none_on_fetch_failure(self):
        assert (
            ourworldindata_mpox.fetch(
                "https://ourworldindata.org/mpox",
                as_of_date=None,
                csv_fetcher=lambda url, config: None,
            )
            is None
        )


class TestLocationTargeting:
    def test_region_surfaces_target_entity(self):
        body = _html(
            ourworldindata_mpox.fetch(
                "https://ourworldindata.org/mpox",
                as_of_date=_asof("2025-03-05"),
                region="Africa",
                csv_fetcher=_mpox_fetch(),
            )
        )
        assert "<h2>Africa</h2>" in body
        assert "cumulative confirmed cases (Africa): 41,000" in body
        assert "Region/question-target focus: Africa trend statistics" in body
        assert "Trend summary (total_cases, Africa):" in body

    def test_question_text_infers_target_entity(self):
        body = _html(
            ourworldindata_mpox.fetch(
                "https://ourworldindata.org/mpox",
                as_of_date=_asof("2025-02-24"),
                question_text="Will mpox cases increase in Africa in 2025?",
                csv_fetcher=_mpox_fetch(),
            )
        )
        assert "<h2>Africa</h2>" in body
        # World is still led with as the headline entity.
        assert "cumulative confirmed cases (World): 129,602" in body


class TestCovidDataset:
    def test_covid_cumulative_regression(self):
        # Feature-branch regression criterion: 732,363,539 @2023-01-01.
        body = _html(
            ourworldindata_covid.fetch(
                "https://ourworldindata.org/coronavirus",
                as_of_date=_asof("2023-01-01"),
                csv_fetcher=lambda url, config: _COVID_CSV,
            )
        )
        assert "cumulative confirmed cases (World): 732,363,539" in body
        assert "733,000,000" not in body  # later row excluded by cutoff


class TestDispatchRouting:
    def test_dispatcher_routes_source_id_through_custom_scraper(self, monkeypatch):
        """fetcher.fetch(source_id=...) must use the custom scraper and tag the
        result fetch_strategy=custom:<source_id>, with no network."""
        from bioscancast.stages.extraction import fetcher as fetcher_mod
        from bioscancast.stages.extraction.custom_scrapers import _owid_common

        monkeypatch.setattr(
            _owid_common, "_fetch_csv_text", lambda url, cfg: _MPOX_CSV
        )

        result = fetcher_mod.fetch(
            "https://ourworldindata.org/mpox",
            config=ExtractionConfig(),
            as_of_date=_asof("2025-02-24"),
            source_id="ourworldindata_mpox",
        )
        assert result is not None
        assert result.fetch_strategy == "custom:ourworldindata_mpox"
        assert result.content_type == "text/html"
        assert "cumulative confirmed cases (World): 129,602" in _html(result)


# ---------------------------------------------------------------------------
# Numeric correctness of the trend-fit helpers
# ---------------------------------------------------------------------------
#
# The rendering tests above assert the fit sections are *present*; these lock
# in the *values* the least-squares helpers produce on known series.

import math

from bioscancast.stages.extraction.custom_scrapers._owid_common import (
    _fit_linear,
    _fit_exponential,
    _r2,
)


class TestTrendFitHelpers:
    def test_fit_linear_recovers_known_slope(self):
        fit = _fit_linear([1.0, 2.0, 3.0, 4.0, 5.0])
        assert fit is not None
        assert fit["slope"] == pytest.approx(1.0)
        assert fit["intercept"] == pytest.approx(1.0)
        assert fit["r2"] == pytest.approx(1.0)

    def test_fit_linear_needs_two_points(self):
        assert _fit_linear([]) is None
        assert _fit_linear([5.0]) is None

    def test_fit_exponential_recovers_doubling(self):
        fit = _fit_exponential([1.0, 2.0, 4.0, 8.0, 16.0])
        assert fit is not None
        # y = a * exp(b*x) with a ~ 1 and b ~ ln(2)
        assert fit["a"] == pytest.approx(1.0, abs=1e-6)
        assert fit["b"] == pytest.approx(math.log(2), rel=1e-6)
        assert fit["r2"] == pytest.approx(1.0, abs=1e-9)

    def test_fit_exponential_needs_two_positive_points(self):
        # Fewer than two strictly-positive observations -> None.
        assert _fit_exponential([0.0, 0.0, 5.0]) is None

    def test_r2_perfect_and_mean_baseline(self):
        assert _r2([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
        # Predicting the mean everywhere yields R^2 == 0.
        assert _r2([1.0, 2.0, 3.0], [2.0, 2.0, 2.0]) == pytest.approx(0.0)
