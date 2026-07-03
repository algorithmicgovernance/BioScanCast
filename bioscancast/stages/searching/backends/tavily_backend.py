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

    def search(
        self,
        query: str,
        max_results: int = 10,
        end_date: Optional[str] = None,
        start_date: Optional[str] = None,
    ) -> List[RawSearchResult]:
        # Date-window behavior (verified 2026-05-20, see
        # ``specs/tavily-investigation-findings.md``): Tavily's news endpoint
        # honors ``start_date`` + ``end_date`` only when **both** are passed
        # together. Passing ``end_date`` alone is silently ignored and the
        # results come back unfiltered. The pipeline is responsible for
        # supplying a sensible ``start_date`` alongside any ``end_date``; if
        # only ``end_date`` is passed here we drop it rather than send a
        # request we know Tavily will misinterpret. The post-retrieval cutoff
        # filter in ``SearchStagePipeline`` remains the authoritative defense.
        from tavily import TavilyClient  # lazy import to avoid hard dep at import time

        client = TavilyClient(api_key=self._api_key)
        kwargs: dict = {
            "query": query,
            "max_results": max_results,
            "topic": "news",
            "include_answer": False,
        }
        if start_date and end_date:
            kwargs["start_date"] = start_date
            kwargs["end_date"] = end_date
        elif end_date and not start_date:
            logger.warning(
                "TavilyBackend received end_date=%s without start_date; "
                "dropping (Tavily ignores end_date alone). Cutoff filter "
                "will still apply post-retrieval.",
                end_date,
            )
        try:
            response = client.search(**kwargs)
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
