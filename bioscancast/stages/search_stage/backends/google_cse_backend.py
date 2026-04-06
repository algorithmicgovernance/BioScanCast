"""Google Custom Search Engine backend — stub.

This exists to show the SearchBackend interface so swapping in a real
Google CSE implementation is trivial later.  See tavily_backend.py docstring
for why Tavily is the default.
"""

from __future__ import annotations

from typing import List

from .base import RawSearchResult


class GoogleCSEBackend:
    """Stub backend — raises NotImplementedError on use."""

    def search(self, query: str, max_results: int = 10) -> List[RawSearchResult]:
        raise NotImplementedError(
            "GoogleCSEBackend is a stub. Implement using the Google Custom Search "
            "JSON API ($5/1k queries after 100/day free tier). See base.py for the "
            "SearchBackend protocol."
        )
