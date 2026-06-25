from datetime import datetime
from bioscancast.stages.extraction.fetcher import FetchResult

def fetch(url: str, *, config=None, as_of_date: datetime | None = None) -> FetchResult | None:
    content = b"..."
    return FetchResult(
        url=url,
        final_url=url,
        status_code=200,
        content_type="text/html",
        content_bytes=content,
        fetched_at=datetime.now(timezone.utc),
        error=None,
        fetch_strategy="custom:who_don",
    )