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
    """Interface that all search backends must satisfy."""

    def search(self, query: str, max_results: int = 10) -> List[RawSearchResult]: ...
