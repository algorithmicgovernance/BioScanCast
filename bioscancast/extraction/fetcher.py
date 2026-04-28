from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx

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
) -> FetchResult:
    """Fetch a URL and return the result. Never raises on network errors."""
    cfg = config or ExtractionConfig()
    fetched_at = datetime.now(timezone.utc)

    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(cfg.fetch_timeout_seconds),
            headers={"User-Agent": cfg.user_agent},
        ) as client:
            with client.stream("GET", url) as response:
                # Check Content-Length header first
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > cfg.fetch_max_bytes:
                    return FetchResult(
                        url=url,
                        final_url=str(response.url),
                        status_code=response.status_code,
                        content_type=_normalize_content_type(
                            response.headers.get("content-type")
                        ),
                        content_bytes=None,
                        fetched_at=fetched_at,
                        error=f"Content-Length {content_length} exceeds max {cfg.fetch_max_bytes} bytes",
                    )

                # Stream and accumulate bytes, checking size limit
                chunks = []
                total = 0
                for chunk in response.iter_bytes(chunk_size=65536):
                    total += len(chunk)
                    if total > cfg.fetch_max_bytes:
                        return FetchResult(
                            url=url,
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
                    url=url,
                    final_url=str(response.url),
                    status_code=response.status_code,
                    content_type=raw_ct,
                    content_bytes=content_bytes,
                    fetched_at=fetched_at,
                    error=None,
                )

    except Exception as exc:
        logger.warning("Fetch failed for %s: %s", url, exc)
        return FetchResult(
            url=url,
            final_url=url,
            status_code=None,
            content_type=None,
            content_bytes=None,
            fetched_at=fetched_at,
            error=str(exc),
        )
