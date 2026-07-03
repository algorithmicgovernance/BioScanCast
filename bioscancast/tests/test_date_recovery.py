from datetime import datetime, timezone
from unittest.mock import patch

from bioscancast.stages.searching.date_recovery import (
    date_from_last_modified,
    date_from_url_slug,
    recover_published_date,
)


class TestDateFromUrlSlug:
    def test_year_month_day_path(self):
        assert date_from_url_slug(
            "https://example.com/2024/03/15/some-article"
        ) == datetime(2024, 3, 15, tzinfo=timezone.utc)

    def test_year_month_only(self):
        assert date_from_url_slug(
            "https://example.com/news/2023/06/topic"
        ) == datetime(2023, 6, 1, tzinfo=timezone.utc)

    def test_iso_dashed(self):
        assert date_from_url_slug(
            "https://example.com/p/2025-01-20/title"
        ) == datetime(2025, 1, 20, tzinfo=timezone.utc)

    def test_no_match(self):
        assert date_from_url_slug("https://example.com/about/contact") is None

    def test_implausible_year_rejected(self):
        # 1872 looks like a year but is too old to be a sensible publication
        assert date_from_url_slug("https://example.com/1872/03/15") is None


class TestDateFromLastModified:
    def test_no_fetcher_returns_none(self):
        # Off by default: requires explicit injection
        assert date_from_last_modified("https://example.com/a") is None

    def test_rfc7231_format(self):
        header = "Wed, 21 Oct 2015 07:28:00 GMT"
        result = date_from_last_modified(
            "https://example.com/a", head_fetcher=lambda _: header
        )
        assert result == datetime(2015, 10, 21, 7, 28, 0, tzinfo=timezone.utc)

    def test_fetcher_returning_none(self):
        assert date_from_last_modified(
            "https://example.com/a", head_fetcher=lambda _: None
        ) is None

    def test_fetcher_raises_returns_none(self):
        def boom(_):
            raise RuntimeError("network down")

        assert date_from_last_modified("https://example.com/a", head_fetcher=boom) is None

    def test_unparseable_header(self):
        assert date_from_last_modified(
            "https://example.com/a", head_fetcher=lambda _: "not a date"
        ) is None


class TestRecoverPublishedDate:
    def test_url_slug_wins(self):
        dt, source = recover_published_date(
            "https://example.com/2024/03/15/x", use_wayback=False
        )
        assert source == "url_slug"
        assert dt == datetime(2024, 3, 15, tzinfo=timezone.utc)

    def test_wayback_used_when_no_slug(self):
        with patch(
            "bioscancast.stages.searching.date_recovery._wayback_first_seen"
        ) as mock_wb:
            mock_wb.return_value = datetime(2020, 1, 1, tzinfo=timezone.utc)
            dt, source = recover_published_date("https://example.com/about")
            assert source == "wayback_first_seen"
            assert dt == datetime(2020, 1, 1, tzinfo=timezone.utc)

    def test_all_strategies_fail(self):
        with patch(
            "bioscancast.stages.searching.date_recovery._wayback_first_seen"
        ) as mock_wb:
            mock_wb.return_value = None
            dt, source = recover_published_date("https://example.com/about")
            assert dt is None
            assert source is None

    def test_wayback_disabled(self):
        dt, source = recover_published_date(
            "https://example.com/about", use_wayback=False
        )
        assert dt is None
        assert source is None
