from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol


@dataclass
class RawSearchResult:
    """Intermediate result returned by any search backend."""

    url: str
    title: str
    snippet: str
    rank: int
    published_date: Optional[str] = None
    score: Optional[float] = None


class SearchBackend(Protocol):
    """Interface that all search backends must satisfy.

    ``start_date`` and ``end_date`` are optional YYYY-MM-DD bounds used by
    historical-replay mode. Tavily's news endpoint requires the **pair** to be
    set together (see ``tavily_backend.py``); passing ``end_date`` alone is
    silently ignored. Backends that don't support either should accept and
    ignore them — the post-retrieval cutoff filter in the pipeline will still
    apply.
    """

    def search(
        self,
        query: str,
        max_results: int = 10,
        end_date: Optional[str] = None,
        start_date: Optional[str] = None,
    ) -> List[RawSearchResult]: ...
