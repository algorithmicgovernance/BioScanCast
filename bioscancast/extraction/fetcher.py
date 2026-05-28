from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from curl_cffi import requests as curl_requests

from bioscancast.stages.search_stage.wayback import closest_snapshot_before

from .config import ExtractionConfig

logger = logging.getLogger(__name__)

# TODO: Add retry logic with exponential backoff.
# TODO: Add rate limiting per domain.
# TODO: Add robots.txt compliance checking.


@dataclass
class FetchResult:
    url: str
    final_url: str
    status_code: Optional[int]
    content_type: Optional[str]
    content_bytes: Optional[bytes]
    fetched_at: datetime
    error: Optional[str]
    fetch_strategy: str = "live"
    snapshot_timestamp: Optional[datetime] = None


def _sniff_content_type(content: bytes) -> Optional[str]:
    """Detect content type from magic bytes when headers are missing/generic."""
    if not content:
        return None
    head = content[:64]
    if head.startswith(b"%PDF-"):
        return "application/pdf"
    head_str = head.lstrip().lower()
    if head_str.startswith((b"<!doctype", b"<html")):
        return "text/html"
    return None


def _normalize_content_type(header_value: Optional[str]) -> Optional[str]:
    """Extract the MIME type portion, stripping charset and other params."""
    if not header_value:
        return None
    return header_value.split(";")[0].strip().lower()


def fetch(
    url: str,
    *,
    config: ExtractionConfig | None = None,
    as_of_date: Optional[datetime] = None,
) -> FetchResult:
    """Fetch a URL and return the result. Never raises on network errors.

    Uses curl_cffi with a browser TLS fingerprint (configurable via
    ExtractionConfig.impersonate) to avoid Cloudflare/JA3-based blocks that
    reject httpx and requests. The impersonation profile sets a matching
    User-Agent automatically.

    Historical-replay mode: when ``as_of_date`` is set the function first
    asks Wayback for the closest capture at-or-before that date and fetches
    the raw snapshot bytes via the ``id_`` modifier. The returned FetchResult
    carries ``fetch_strategy="wayback"`` and ``snapshot_timestamp`` set to
    the capture time. If no snapshot exists, or the Wayback fetch errors,
    we fall back to a live fetch and tag the result
    ``fetch_strategy="wayback_fallback_to_live"`` so audit reports can see
    the leak. The fallback is logged at INFO — never silent.
    """
    if as_of_date is not None:
        snapshot = closest_snapshot_before(url, as_of_date)
        if snapshot is not None:
            snapshot_dt, snapshot_url = snapshot
            wb_result = _fetch_via_curl(
                target_url=snapshot_url,
                reported_url=url,
                config=config,
            )
            if wb_result.error is None and wb_result.content_bytes is not None:
                wb_result.fetch_strategy = "wayback"
                wb_result.snapshot_timestamp = snapshot_dt
                return wb_result
            logger.info(
                "Wayback fetch failed for %s (snapshot %s, error=%s); "
                "falling back to live",
                url, snapshot_dt.isoformat(), wb_result.error,
            )
        else:
            logger.info(
                "No Wayback snapshot for %s at-or-before %s; falling back to live",
                url, as_of_date.isoformat(),
            )
        live_result = _fetch_via_curl(target_url=url, reported_url=url, config=config)
        live_result.fetch_strategy = "wayback_fallback_to_live"
        return live_result

    return _fetch_via_curl(target_url=url, reported_url=url, config=config)


def _fetch_via_curl(
    *,
    target_url: str,
    reported_url: str,
    config: ExtractionConfig | None,
) -> FetchResult:
    """Issue the actual HTTP GET. ``target_url`` is what we hit (may be a
    Wayback ``id_`` URL); ``reported_url`` is what we record in
    ``FetchResult.url`` so downstream consumers see the original publisher
    URL, not archive.org."""
    cfg = config or ExtractionConfig()
    fetched_at = datetime.now(timezone.utc)

    try:
        # curl_cffi's streaming Response is not a context manager in the
        # installed version, so we close it explicitly in a finally block.
        response = curl_requests.get(
            target_url,
            stream=True,
            timeout=cfg.fetch_timeout_seconds,
            impersonate=cfg.impersonate,
            allow_redirects=True,
        )
        try:
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > cfg.fetch_max_bytes:
                return FetchResult(
                    url=reported_url,
                    final_url=str(response.url),
                    status_code=response.status_code,
                    content_type=_normalize_content_type(
                        response.headers.get("content-type")
                    ),
                    content_bytes=None,
                    fetched_at=fetched_at,
                    error=f"Content-Length {content_length} exceeds max {cfg.fetch_max_bytes} bytes",
                )

            chunks = []
            total = 0
            # chunk_size is not honoured by curl_cffi (warns); curl picks
            # the buffer size internally.
            for chunk in response.iter_content():
                total += len(chunk)
                if total > cfg.fetch_max_bytes:
                    return FetchResult(
                        url=reported_url,
                        final_url=str(response.url),
                        status_code=response.status_code,
                        content_type=_normalize_content_type(
                            response.headers.get("content-type")
                        ),
                        content_bytes=None,
                        fetched_at=fetched_at,
                        error=f"Response exceeded max {cfg.fetch_max_bytes} bytes during streaming",
                    )
                chunks.append(chunk)

            content_bytes = b"".join(chunks)
            raw_ct = _normalize_content_type(
                response.headers.get("content-type")
            )

            # Fall back to magic-byte sniffing if header is
            # missing or generic (e.g. application/octet-stream)
            if not raw_ct or raw_ct == "application/octet-stream":
                raw_ct = _sniff_content_type(content_bytes) or raw_ct

            return FetchResult(
                url=reported_url,
                final_url=str(response.url),
                status_code=response.status_code,
                content_type=raw_ct,
                content_bytes=content_bytes,
                fetched_at=fetched_at,
                error=None,
            )
        finally:
            response.close()

    except Exception as exc:
        logger.warning("Fetch failed for %s: %s", target_url, exc)
        return FetchResult(
            url=reported_url,
            final_url=reported_url,
            status_code=None,
            content_type=None,
            content_bytes=None,
            fetched_at=fetched_at,
            error=str(exc),
        )
