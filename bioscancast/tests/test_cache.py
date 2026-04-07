import os
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from bioscancast.stages.search_stage.backends.base import RawSearchResult
from bioscancast.stages.search_stage.cache import SearchCache


def _make_results():
    return [
        RawSearchResult(
            url="https://example.com/a",
            title="Result A",
            snippet="Snippet A",
            rank=1,
        ),
        RawSearchResult(
            url="https://example.com/b",
            title="Result B",
            snippet="Snippet B",
            rank=2,
            published_date="2025-01-10",
        ),
    ]


class TestSearchCache:
    def test_put_then_get_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_cache.sqlite")
            cache = SearchCache(db_path=db_path)
            try:
                results = _make_results()
                cache.put("tavily", "h5n1 cases", results)

                cached = cache.get("tavily", "h5n1 cases")
                assert cached is not None
                assert len(cached) == 2
                assert cached[0].url == "https://example.com/a"
                assert cached[1].title == "Result B"
                assert cached[1].published_date == "2025-01-10"
            finally:
                cache.close()

    def test_cache_miss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_cache.sqlite")
            cache = SearchCache(db_path=db_path)
            try:
                result = cache.get("tavily", "nonexistent query")
                assert result is None
            finally:
                cache.close()

    def test_different_backends_isolated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_cache.sqlite")
            cache = SearchCache(db_path=db_path)
            try:
                cache.put("tavily", "query", _make_results())
                assert cache.get("tavily", "query") is not None
                assert cache.get("google", "query") is None
            finally:
                cache.close()

    def test_expired_entry_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_cache.sqlite")
            cache = SearchCache(db_path=db_path)
            try:
                cache.put("tavily", "old query", _make_results())

                # Mock datetime to simulate 25 hours passing
                future = datetime.now(timezone.utc) + timedelta(hours=25)
                with patch("bioscancast.stages.search_stage.cache.datetime") as mock_dt:
                    mock_dt.now.return_value = future
                    mock_dt.fromisoformat = datetime.fromisoformat
                    mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

                    result = cache.get("tavily", "old query")
                    assert result is None
            finally:
                cache.close()

    def test_query_normalization(self):
        """Cache keys are case-insensitive and whitespace-stripped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_cache.sqlite")
            cache = SearchCache(db_path=db_path)
            try:
                cache.put("tavily", "  H5N1 Cases  ", _make_results())
                result = cache.get("tavily", "h5n1 cases")
                assert result is not None
            finally:
                cache.close()
