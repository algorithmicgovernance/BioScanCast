"""URL normalization for deduplication.

normalize_url() produces a canonical form so the same page reached via
different tracking-parameter variants collapses to a single entry.
"""

from __future__ import annotations

from urllib.parse import ParseResult, parse_qs, urlencode, urlparse

_TRACKING_PARAMS: set[str] = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_id",
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
}


def normalize_url(url: str) -> str:
    """Return a canonical URL suitable for deduplication.

    Steps:
      1. Lowercase scheme and host.
      2. Strip ``www.`` prefix from host.
      3. Strip trailing slash from path.
      4. Strip URL fragment (``#...``).
      5. Strip tracking query parameters (utm_*, fbclid, etc.).
      6. Preserve remaining query parameters (sorted for stability).
    """
    parsed: ParseResult = urlparse(url)

    scheme = parsed.scheme.lower() or "https"
    host = parsed.hostname or ""
    if host.startswith("www."):
        host = host[4:]

    # Reconstruct netloc (host + optional port)
    port = parsed.port
    netloc = host
    if port and not (scheme == "http" and port == 80) and not (scheme == "https" and port == 443):
        netloc = f"{host}:{port}"

    path = parsed.path.rstrip("/") or ""

    # Filter tracking query params, keep the rest sorted
    query_params = parse_qs(parsed.query, keep_blank_values=True)
    filtered = {
        k: v for k, v in query_params.items() if k.lower() not in _TRACKING_PARAMS
    }
    query_string = urlencode(sorted(filtered.items()), doseq=True) if filtered else ""

    # Reassemble without fragment
    result = f"{scheme}://{netloc}{path}"
    if query_string:
        result = f"{result}?{query_string}"
    return result


def extract_domain(url: str) -> str:
    """Extract the registrable domain from a URL, stripping ``www.`` prefix."""
    host = urlparse(url).hostname or ""
    if host.startswith("www."):
        host = host[4:]
    return host.lower()
