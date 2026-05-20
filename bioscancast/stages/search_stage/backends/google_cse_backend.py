"""Google Custom Search Engine backend — stub.

This exists to show the SearchBackend interface so swapping in a real
Google CSE implementation is trivial later.  See tavily_backend.py docstring
for why Tavily is the default.
"""

from __future__ import annotations

from typing import List, Optional

from .base import RawSearchResult


class GoogleCSEBackend:
    """Stub backend — raises NotImplementedError on use."""

    def search(
        self,
        query: str,
        max_results: int = 10,
        end_date: Optional[str] = None,
    ) -> List[RawSearchResult]:
        raise NotImplementedError(
            "GoogleCSEBackend is a stub. Implement using the Google Custom Search "
            "JSON API ($5/1k queries after 100/day free tier). When implementing, "
            "the YYYY-MM-DD `end_date` argument should be honoured via the CSE "
            "`sort=date:r:YYYYMMDD:YYYYMMDD` parameter for historical-replay mode. "
            "See base.py for the SearchBackend protocol."
        )
