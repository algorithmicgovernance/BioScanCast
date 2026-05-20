"""Tiny Wayback Machine CDX client used by historical-replay mode.

Two callers need this:

* the search-stage dashboard rewrite (closest snapshot at-or-before cutoff),
* the extraction-stage fetcher (same lookup, then fetch the snapshot bytes),
* and the date-recovery chain in the search stage (first-seen capture).

The implementation deliberately uses stdlib ``urllib`` rather than ``curl_cffi``:
the Wayback CDX endpoint returns a small JSON document, is not protected by
Cloudflare/JA3 filters, and adding ``curl_cffi`` as a dependency of the
search stage would broaden the dependency surface unnecessarily.

All functions return ``None`` on any network/parse error and log at WARNING.
Callers must tolerate ``None`` — Wayback is best-effort.
"""

from __future__ import annotations

import json
import logging
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

CDX_ENDPOINT = "https://web.archive.org/cdx/search/cdx"
SNAPSHOT_TEMPLATE = "https://web.archive.org/web/{timestamp}/{url}"
# The ``id_`` modifier on the timestamp returns the raw original bytes
# (no Wayback toolbar / no rewriting). Use this when we want to feed the
# response straight into our HTML/PDF parsers.
RAW_SNAPSHOT_TEMPLATE = "https://web.archive.org/web/{timestamp}id_/{url}"

_REQUEST_TIMEOUT_SECONDS = 30

# Retry schedule for the Wayback CDX endpoint. The endpoint frequently
# returns HTTP 503 or times out under load — empirically a single
# benchmark run can trigger dozens of consecutive failures. We trade
# wall-clock time for completeness because historical-replay benchmarks
# are not latency-sensitive. Pre-attempt delays in seconds; the i-th
# entry is the wait BEFORE attempt i (so the first attempt fires
# immediately). Override at module level in tests to keep them fast.
RETRY_BACKOFF_SECONDS: tuple[float, ...] = (0, 10, 30, 90, 240)

# Recoverable HTTP status codes that warrant a retry.
_RECOVERABLE_STATUSES = {429, 500, 502, 503, 504}


def _sleep(seconds: float) -> None:
    """Indirection so tests can monkeypatch a no-op sleep."""
    if seconds > 0:
        time.sleep(seconds)


def _cdx_query(params: dict) -> Optional[list]:
    """POST-free GET against the CDX endpoint. Returns the parsed JSON list,
    or None on any failure. Retries on HTTP 503/429/5xx and read timeouts
    according to ``RETRY_BACKOFF_SECONDS``."""
    query = urllib.parse.urlencode(params)
    full_url = f"{CDX_ENDPOINT}?{query}"

    body: Optional[str] = None
    for attempt, pre_delay in enumerate(RETRY_BACKOFF_SECONDS, start=1):
        if pre_delay:
            logger.info(
                "Wayback CDX backoff %.0fs before attempt %d/%d",
                pre_delay, attempt, len(RETRY_BACKOFF_SECONDS),
            )
            _sleep(pre_delay)
        try:
            req = urllib.request.Request(
                full_url, headers={"User-Agent": "BioScanCast/replay (+wayback-cdx)"}
            )
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            break  # success
        except urllib.error.HTTPError as exc:
            if exc.code in _RECOVERABLE_STATUSES and attempt < len(RETRY_BACKOFF_SECONDS):
                logger.info(
                    "Wayback CDX HTTP %d on attempt %d; retrying",
                    exc.code, attempt,
                )
                continue
            logger.warning(
                "Wayback CDX gave up after %d attempt(s): HTTP %d for %s",
                attempt, exc.code, full_url,
            )
            return None
        except (socket.timeout, TimeoutError, urllib.error.URLError) as exc:
            # urllib.error.URLError wraps socket.timeout on read timeouts in
            # some Python builds; check both.
            is_timeout = isinstance(exc, (socket.timeout, TimeoutError)) or (
                isinstance(exc, urllib.error.URLError)
                and isinstance(getattr(exc, "reason", None), (socket.timeout, TimeoutError))
            )
            if is_timeout and attempt < len(RETRY_BACKOFF_SECONDS):
                logger.info(
                    "Wayback CDX timeout on attempt %d; retrying", attempt,
                )
                continue
            logger.warning(
                "Wayback CDX gave up after %d attempt(s): %s for %s",
                attempt, exc, full_url,
            )
            return None
        except Exception as exc:
            logger.warning(
                "Wayback CDX non-recoverable error: %s for %s", exc, full_url
            )
            return None

    if body is None:
        return None

    if not body.strip():
        return []
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        logger.warning("Wayback CDX returned non-JSON body for %s", full_url)
        return None

    # First row is the header. Drop it.
    if isinstance(data, list) and data and isinstance(data[0], list):
        return data[1:]
    return []


def _parse_cdx_timestamp(ts: str) -> Optional[datetime]:
    """CDX timestamps are YYYYMMDDhhmmss in UTC."""
    try:
        return datetime.strptime(ts, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def closest_snapshot_before(
    url: str, as_of: datetime
) -> Optional[Tuple[datetime, str]]:
    """Return (snapshot_datetime, raw_snapshot_url) for the latest Wayback
    capture of ``url`` whose timestamp is ``<= as_of``. Returns None when no
    suitable snapshot exists or the lookup fails.

    The returned URL uses the ``id_`` modifier so callers get unwrapped
    original content (no Wayback chrome).
    """
    to_param = as_of.astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")
    rows = _cdx_query(
        {
            "url": url,
            "to": to_param,
            "limit": "-1",  # most recent matching row
            "output": "json",
            "filter": "statuscode:200",
        }
    )
    if not rows:
        return None
    timestamp = rows[0][1]  # column 1 is the capture timestamp
    parsed = _parse_cdx_timestamp(timestamp)
    if parsed is None:
        return None
    snapshot_url = RAW_SNAPSHOT_TEMPLATE.format(timestamp=timestamp, url=url)
    return parsed, snapshot_url


def first_seen(url: str) -> Optional[datetime]:
    """Return the earliest Wayback capture timestamp for ``url``, or None."""
    rows = _cdx_query(
        {
            "url": url,
            "limit": "1",
            "output": "json",
            "filter": "statuscode:200",
            "sort": "ascending",
        }
    )
    if not rows:
        return None
    return _parse_cdx_timestamp(rows[0][1])
