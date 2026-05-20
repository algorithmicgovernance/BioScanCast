"""SQLite-backed search result cache.

Keyed on (backend_name, normalized_query, date_bucket) where date_bucket is
the current date (YYYY-MM-DD).  Entries expire after 24 hours.

This saves real money during iterative development by avoiding redundant
API calls to Tavily / other backends.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from .backends.base import RawSearchResult

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS search_cache (
    cache_key TEXT PRIMARY KEY,
    results_json TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""


class SearchCache:
    def __init__(self, db_path: str = "data/cache/search_cache.sqlite") -> None:
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    @staticmethod
    def _make_key(
        backend_name: str,
        query: str,
        as_of_date: Optional[datetime] = None,
    ) -> str:
        # In historical-replay mode the bucket is the cutoff date, so that two
        # benchmark runs against different cutoffs never share cache entries.
        if as_of_date is not None:
            date_bucket = as_of_date.strftime("%Y-%m-%d")
        else:
            date_bucket = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        raw = f"{backend_name}|{query.strip().lower()}|{date_bucket}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(
        self,
        backend_name: str,
        query: str,
        max_age_hours: int = 24,
        as_of_date: Optional[datetime] = None,
    ) -> Optional[List[RawSearchResult]]:
        key = self._make_key(backend_name, query, as_of_date)
        row = self._conn.execute(
            "SELECT results_json, created_at FROM search_cache WHERE cache_key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None

        created = datetime.fromisoformat(row[1])
        if datetime.now(timezone.utc) - created > timedelta(hours=max_age_hours):
            self._conn.execute("DELETE FROM search_cache WHERE cache_key = ?", (key,))
            self._conn.commit()
            return None

        items = json.loads(row[0])
        return [RawSearchResult(**item) for item in items]

    def put(
        self,
        backend_name: str,
        query: str,
        results: List[RawSearchResult],
        as_of_date: Optional[datetime] = None,
    ) -> None:
        key = self._make_key(backend_name, query, as_of_date)
        payload = json.dumps(
            [
                {
                    "url": r.url,
                    "title": r.title,
                    "snippet": r.snippet,
                    "rank": r.rank,
                    "published_date": r.published_date,
                    "score": r.score,
                }
                for r in results
            ]
        )
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO search_cache (cache_key, results_json, created_at) "
            "VALUES (?, ?, ?)",
            (key, payload, now),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
