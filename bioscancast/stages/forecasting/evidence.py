"""Turn insight records into a compact, forecast-ready evidence digest.

This is pure Python — no LLM calls. The digest is the only view of the
retrieved evidence the reasoning model sees, so it must be compact (to
bound input-token cost), ordered (recency first, because forecasting
rewards the freshest signal), and auditable (each line carries a verbatim
quote and a source URL back to the InsightRecord's provenance).
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Tuple

from bioscancast.schemas import InsightRecord


def _format_date(dt: Optional[datetime], precision: Optional[str]) -> str:
    """Render an event date at its known precision (year/month/day)."""
    if dt is None:
        return "undated"
    if precision == "year":
        return dt.strftime("%Y")
    if precision == "month":
        return dt.strftime("%Y-%m")
    return dt.strftime("%Y-%m-%d")


def _has_content(record: InsightRecord) -> bool:
    """A record is useful to a forecaster if it carries a metric or a
    free-text summary. Records with neither are noise."""
    return record.metric_value is not None or bool(
        record.summary and record.summary.strip()
    )


def _sort_key(record: InsightRecord) -> Tuple[int, float, float]:
    """Sort dated records before undated ones, newest first, then by
    extraction confidence. Returns a tuple compared in descending order.
    """
    if record.event_date is not None:
        return (1, record.event_date.timestamp(), record.confidence)
    return (0, 0.0, record.confidence)


def _format_record(record: InsightRecord) -> str:
    """Render one InsightRecord as a single digest line."""
    date_str = _format_date(record.event_date, record.event_date_precision)
    parts: List[str] = [f"[{date_str}]"]

    if record.pathogen:
        parts.append(record.pathogen)

    location = record.location or record.iso_country_code
    if location:
        parts.append(f"in {location}")

    if record.metric_value is not None:
        metric = record.metric_name or "value"
        value = record.metric_value
        # Render whole numbers without a trailing ".0".
        value_str = str(int(value)) if float(value).is_integer() else str(value)
        unit = f" {record.metric_unit}" if record.metric_unit else ""
        # Surface the epidemiological basis so the forecaster does not read
        # cumulative totals, weekly increments, and active-case counts as one
        # comparable series. "unknown"/None add no tag (keeps the line clean).
        qual = ""
        cb = record.count_basis
        if cb and cb != "unknown":
            tw = record.time_window
            if cb == "incident" and tw and tw != "unknown":
                qual = f" [{cb}/{tw}]"
            else:
                qual = f" [{cb}]"
        parts.append(f"{metric}={value_str}{unit}{qual}")
    elif record.summary:
        parts.append(record.summary.strip())

    # Explicit data-quality caveats surface even on metric lines (whose
    # summary is otherwise dropped above), so the forecaster can weigh
    # under-reporting / reporting-lag / surveillance-change signals.
    if record.data_quality and record.data_quality.strip():
        parts.append(f"(caveat: {record.data_quality.strip()})")

    head = " ".join(parts)

    quote = ""
    source = ""
    if record.sources:
        ref = record.sources[0]
        if ref.quote:
            quote = f' "{ref.quote.strip()}"'
        if ref.source_url:
            source = f" ({ref.source_url})"

    return f"- {head}{quote}{source}"


def build_evidence_digest(
    records: List[InsightRecord],
    *,
    max_records: int = 40,
) -> Tuple[str, List[str]]:
    """Build the evidence digest text and the list of record IDs used.

    Args:
        records: InsightRecords from the insight stage.
        max_records: Cap on the number of records included, taken by
            recency then confidence.

    Returns:
        ``(digest_text, used_record_ids)``. ``digest_text`` is an empty
        string when no record carries usable content.
    """
    usable = [r for r in records if _has_content(r)]
    usable.sort(key=_sort_key, reverse=True)
    usable = usable[:max_records]

    if not usable:
        return "", []

    lines = [_format_record(r) for r in usable]
    return "\n".join(lines), [r.id for r in usable]
