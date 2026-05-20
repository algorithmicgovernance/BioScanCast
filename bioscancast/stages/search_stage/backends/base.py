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

    ``end_date`` is an optional YYYY-MM-DD upper bound used by historical-
    replay mode. Backends that don't support it should accept and ignore it
    (the post-retrieval cutoff filter in the pipeline will still apply).
    """

    def search(
        self,
        query: str,
        max_results: int = 10,
        end_date: Optional[str] = None,
    ) -> List[RawSearchResult]: ...
