"""Tavily search backend — the default for BioScanCast.

Tavily returns published_date, cleaner snippets, and supports topic="news"
and date filtering.  Free tier: 1,000 requests/month, sufficient for dev.

Why not Google CSE directly:
  There is no free, ToS-compliant programmatic Google search.  Google Custom
  Search JSON API is paid ($5/1k after 100/day free) and returns thinner
  metadata.  If a Google backend is needed later, implement GoogleCSEBackend
  using the SearchBackend protocol in base.py.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from .base import RawSearchResult, SearchBackend

logger = logging.getLogger(__name__)


class TavilyBackend:
    """Concrete SearchBackend using the Tavily Search API."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or os.environ.get("TAVILY_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "TAVILY_API_KEY is required. Set it in your environment or pass api_key."
            )

    def search(self, query: str, max_results: int = 10) -> List[RawSearchResult]:
        from tavily import TavilyClient  # lazy import to avoid hard dep at import time

        client = TavilyClient(api_key=self._api_key)
        try:
            response = client.search(
                query=query,
                max_results=max_results,
                topic="news",
                include_answer=False,
            )
        except Exception:
            logger.exception("Tavily search failed for query: %s", query)
            return []

        results: list[RawSearchResult] = []
        for i, item in enumerate(response.get("results", [])):
            try:
                results.append(
                    RawSearchResult(
                        url=item["url"],
                        title=item.get("title", ""),
                        snippet=item.get("content", ""),
                        rank=i + 1,
                        published_date=item.get("published_date"),
                        score=item.get("score"),
                    )
                )
            except (KeyError, TypeError):
                logger.warning("Skipping malformed Tavily result at index %d", i)
                continue

        return results
