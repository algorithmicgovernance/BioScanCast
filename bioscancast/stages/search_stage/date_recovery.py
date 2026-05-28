"""Recover a plausible publication date for a SearchResult whose backend
didn't supply one.

Why this exists: in historical-replay mode the pipeline must drop any source
it cannot date, because undated pages are exactly where post-cutoff content
can hide (a page first published before the cutoff but rewritten afterwards
will still report no ``published_date`` from Tavily). Soft-allowing undated
results would silently defeat the benchmark; this module instead tries cheap
external signals before giving up.

Recovery strategies, cheapest first:

1. URL slug regex (``/2024/03/15/...`` and ``/2024-03-15/...``) — free, no
   network call. Catches most news organisations.
2. ``Last-Modified`` header via HEAD request — off by default. Requires
   passing a fetcher callable explicitly; the search stage does not normally
   carry an HTTP client. Opt in only when you need the recall.
3. Wayback Machine first-seen — one CDX call per URL. Conservative: "first
   archived" is an upper bound on first published, so a pre-cutoff first-
   seen is sound evidence the page existed before the cutoff.

Each function returns ``Optional[datetime]`` and never raises on network or
parse errors (it logs and returns ``None``).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Callable, Optional

from .wayback import first_seen as _wayback_first_seen

logger = logging.getLogger(__name__)

# Matches /YYYY/MM/DD/ and /YYYY/MM/ and /YYYY-MM-DD/ within a URL path.
_URL_DATE_PATTERNS = [
    re.compile(r"/(\d{4})/(\d{1,2})/(\d{1,2})(?:/|$|[?#])"),
    re.compile(r"/(\d{4})-(\d{1,2})-(\d{1,2})(?:/|$|[?#])"),
    re.compile(r"/(\d{4})/(\d{1,2})(?:/|$|[?#])"),
]


def date_from_url_slug(url: str) -> Optional[datetime]:
    """Extract a date from common URL slug patterns. Returns midnight UTC of
    the matched date, or None if no pattern matches or the date is invalid."""
    for pattern in _URL_DATE_PATTERNS:
        m = pattern.search(url)
        if not m:
            continue
        groups = m.groups()
        try:
            year = int(groups[0])
            month = int(groups[1])
            day = int(groups[2]) if len(groups) >= 3 else 1
            if year < 1990 or year > 2100:
                continue  # almost certainly not a date
            return datetime(year, month, day, tzinfo=timezone.utc)
        except (ValueError, IndexError):
            continue
    return None


def date_from_last_modified(
    url: str, head_fetcher: Optional[Callable[[str], Optional[str]]] = None
) -> Optional[datetime]:
    """Issue a HEAD request and parse the Last-Modified header.

    The caller must pass a ``head_fetcher`` callable that returns the
    Last-Modified header string (or None). The search stage does not have a
    built-in HTTP client, so this is dependency-injected to avoid an awkward
    import of ``curl_cffi`` into the search-stage package. Off by default.
    """
    if head_fetcher is None:
        return None
    try:
        header = head_fetcher(url)
    except Exception as exc:
        logger.warning("HEAD request failed for %s: %s", url, exc)
        return None
    if not header:
        return None
    # RFC 7231 format: "Wed, 21 Oct 2015 07:28:00 GMT"
    for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S GMT"):
        try:
            dt = datetime.strptime(header, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def date_from_wayback_first_seen(url: str) -> Optional[datetime]:
    """Earliest Wayback capture timestamp for ``url`` as an upper bound on
    first publication. Returns None on lookup failure."""
    return _wayback_first_seen(url)


def recover_published_date(
    url: str,
    head_fetcher: Optional[Callable[[str], Optional[str]]] = None,
    use_wayback: bool = True,
) -> tuple[Optional[datetime], Optional[str]]:
    """Try each strategy in order. Returns (date, source_label) where the
    label is one of ``"url_slug" | "last_modified" | "wayback_first_seen"``
    on success, or (None, None) when no strategy yielded a date.
    """
    dt = date_from_url_slug(url)
    if dt is not None:
        return dt, "url_slug"

    dt = date_from_last_modified(url, head_fetcher=head_fetcher)
    if dt is not None:
        return dt, "last_modified"

    if use_wayback:
        dt = date_from_wayback_first_seen(url)
        if dt is not None:
            return dt, "wayback_first_seen"

    return None, None
