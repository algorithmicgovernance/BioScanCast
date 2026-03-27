from datetime import datetime

from bioscancast.filtering.heuristics import is_low_value_page
from bioscancast.filtering.models import SearchResult


def test_low_value_page_detected():
    result = SearchResult(
        id="1",
        question_id="q1",
        query_id="s1",
        engine="google",
        url="https://example.com/login",
        canonical_url=None,
        domain="example.com",
        title="Login",
        snippet="Sign in here",
        rank=1,
        retrieved_at=datetime.utcnow(),
    )
    assert is_low_value_page(result) is True