from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from bioscancast.stages.filtering.models import ForecastQuestion


PathLike = Union[str, Path]

logger = logging.getLogger(__name__)


_MONTH_NUMBERS: dict[str, int] = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}

_MONTH_ALT = (
    "|".join(_MONTH_NUMBERS.keys())
    + "|"
    + "|".join(m[:3] for m in _MONTH_NUMBERS.keys())
)

# A date like "February 28, 2025" / "Feb 28th 2025" / month-only "February
# 2025". No preposition is required here — the preposition is inspected
# separately so a deadline date ("by/before/to <date>") can be told apart
# from a window-start date ("from <date>").
_DATE_RE = re.compile(
    r"(?P<month>" + _MONTH_ALT + r")"
    r"(?:\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?)?"
    r"(?:,)?\s+(?P<year>\d{4})",
    re.IGNORECASE,
)

# Prepositions that introduce a resolution/deadline date — including "to"
# as the end of a "from X to Y" window. Deliberately excludes "from",
# "between", and "since", which introduce a window's START.
_RESOLUTION_PREP_RE = re.compile(r"\b(?:by|before|until|through|to)\s+$", re.IGNORECASE)


def _match_to_date(m: "re.Match[str]") -> Optional[datetime]:
    """Build a UTC datetime from a ``_DATE_RE`` match.

    Day-precise when a day is present; otherwise the 1st of the following
    month (same cutoff semantics as an inclusive "by <month> <year>").
    """
    month_name = m.group("month").lower()
    if len(month_name) == 3:
        month = next(
            (num for full, num in _MONTH_NUMBERS.items() if full.startswith(month_name)),
            None,
        )
        if month is None:
            return None
    else:
        month = _MONTH_NUMBERS[month_name]
    year = int(m.group("year"))
    day_str = m.group("day")
    if day_str:
        try:
            return datetime(year, month, int(day_str), tzinfo=timezone.utc)
        except ValueError:
            return None
    next_month, next_year = (1, year + 1) if month == 12 else (month + 1, year)
    return datetime(next_year, next_month, 1, tzinfo=timezone.utc)


def _parse_target_date(text: Optional[str]) -> Optional[datetime]:
    """Extract a resolution/target date from natural-language text.

    Handles:
    - "by February 28, 2025", "by Feb 28 2025", month-only "by February 2025"
    - "before May 1st, 2025"
    - ranges "from January 1st, 2025, to May 1st, 2025" → the END date
    - the date living in the resolution_criteria rather than the question
      text (the caller passes that as a fallback)

    Strategy: find every "<Month> [day] <year>" in the text and prefer the
    last one introduced by a resolution preposition (by/before/until/
    through/to) — this picks a deadline and the end of a "from X to Y"
    window over its start — falling back to the last date overall. Returns
    None if nothing parses.
    """
    if not text:
        return None
    matches = list(_DATE_RE.finditer(text))
    if not matches:
        return None
    prep = [m for m in matches if _RESOLUTION_PREP_RE.search(text[: m.start()])]
    for m in reversed(prep or matches):
        dt = _match_to_date(m)
        if dt is not None:
            return dt
    return None


def _split_topic(topic: str) -> tuple[Optional[str], Optional[str]]:
    """Split a topic like "Mpox (World)" → ("mpox", "world").

    Returns (pathogen, region). If no parenthetical exists, the whole topic
    becomes the pathogen and region is None.
    """
    if not topic or pd.isna(topic):
        return None, None
    topic = topic.strip()
    if "(" in topic and topic.endswith(")"):
        head, _, tail = topic.rpartition(" (")
        return head.strip().lower() or None, tail.rstrip(")").strip().lower() or None
    return topic.lower() or None, None


def _infer_event_type(question_type: str, question_text: str) -> Optional[str]:
    """Map the CSV's question_type plus keyword hints to an event_type."""
    text = (question_text or "").lower()
    if "deaths" in text or "death" in text:
        return "death_count"
    if "cases" in text or "case " in text:
        return "case_count"
    if "outbreak" in text:
        return "outbreak_declared"
    return None


def _read_csv(path: PathLike) -> pd.DataFrame:
    """
    Read one of the BioScanCast CSV files with the correct separator,
    encoding, and decimal format.
    """
    return pd.read_csv(
        path,
        sep=";",
        encoding="cp1252",
        decimal=",",
    )


def _clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize common text fields so matching is stable across files.

    This mainly helps with:
    - spacing
    - dash variants
    - accidental surrounding whitespace
    """
    text_columns = df.select_dtypes(include="object").columns

    for col in text_columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("\u2013", "-", regex=False)  # en dash -> hyphen
            .str.replace("\u2014", "-", regex=False)  # em dash -> hyphen
            .str.strip()
        )

    return df


def load_questions(path: PathLike) -> pd.DataFrame:
    """
    Load the question metadata CSV.

    Expected columns:
    - question_id
    - topic
    - question_text
    - question_type
    - resolution_criteria
    - created_date
    - question_status
    - resolved_option
    - comparison_to_outcome
    - takeaways
    - relevant_links
    """
    df = _read_csv(path)
    df = _clean_text_columns(df)

    if "created_date" in df.columns:
        # The CSV stores created_date as an Excel serial day (e.g. 45712 →
        # 2025-02-19). Without unit="D" + origin="1899-12-30", pandas treats
        # the integer as nanoseconds past 1970 and produces garbage dates
        # like 1970-01-01 00:00:00.000045712.
        df["created_date"] = pd.to_datetime(
            df["created_date"], unit="D", origin="1899-12-30", errors="coerce",
        )

    if "question_status" in df.columns:
        df["question_status"] = df["question_status"].str.lower()

    return df


def load_forecasts(path: PathLike) -> pd.DataFrame:
    """
    Load the forecasts CSV.

    Expected columns:
    - question_id
    - forecast_version
    - option
    - probability
    """
    df = _read_csv(path)
    df = _clean_text_columns(df)

    if "probability" not in df.columns:
        raise ValueError("Forecast file must contain a 'probability' column.")

    df["probability"] = pd.to_numeric(df["probability"], errors="coerce")

    if df["probability"].isna().any():
        bad_rows = df[df["probability"].isna()]
        raise ValueError(
            "Some forecast probabilities could not be parsed as numbers. "
            f"Problematic rows: {bad_rows.index.tolist()}"
        )

    if "forecast_version" in df.columns:
        df["forecast_version"] = df["forecast_version"].astype(str).str.strip()

    if "question_id" in df.columns:
        df["question_id"] = df["question_id"].str.strip()

    return df


def build_forecast_question(
    row: pd.Series,
    *,
    as_of_date: Optional[datetime] = None,
) -> ForecastQuestion:
    """Convert one row of the question CSV into a ForecastQuestion.

    Used by the orchestrator (`bioscancast.main`) to turn a CSV row into the
    typed object the search/filter/insight stages expect. ``as_of_date`` is
    passed through verbatim and is the historical-replay cutoff; ``None``
    means live mode.
    """
    qid = str(row["question_id"]).strip()
    text = str(row["question_text"]).strip()

    created_value = row.get("created_date")
    if isinstance(created_value, pd.Timestamp) and not pd.isna(created_value):
        created_at = created_value.to_pydatetime()
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
    elif isinstance(created_value, datetime):
        created_at = (
            created_value if created_value.tzinfo
            else created_value.replace(tzinfo=timezone.utc)
        )
    else:
        logger.warning(
            "Question %s has unparseable created_date %r; defaulting to now()",
            qid, created_value,
        )
        created_at = datetime.now(timezone.utc)

    topic = row.get("topic", "")
    pathogen, region = _split_topic(str(topic) if not pd.isna(topic) else "")

    resolution_criteria_val = row.get("resolution_criteria")
    resolution_criteria = (
        str(resolution_criteria_val).strip()
        if resolution_criteria_val is not None and not pd.isna(resolution_criteria_val)
        else None
    )

    # Prefer a date in the question text; fall back to the resolution
    # criteria, which is the only place some questions (e.g. q11) state the
    # cutoff.
    target_date = _parse_target_date(text) or _parse_target_date(resolution_criteria)
    event_type = _infer_event_type(str(row.get("question_type", "")), text)

    return ForecastQuestion(
        id=qid,
        text=text,
        created_at=created_at,
        target_date=target_date,
        region=region,
        pathogen=pathogen,
        event_type=event_type,
        resolution_criteria=resolution_criteria,
        as_of_date=as_of_date,
    )


def load_question_by_id(
    path: PathLike,
    question_id: str,
    *,
    as_of_date: Optional[datetime] = None,
) -> ForecastQuestion:
    """Load a single question from the CSV by its question_id."""
    df = load_questions(path)
    matches = df[df["question_id"].astype(str).str.strip() == question_id.strip()]
    if matches.empty:
        available = sorted(df["question_id"].astype(str).str.strip().tolist())
        raise KeyError(
            f"question_id {question_id!r} not found in {path}. "
            f"Available: {available}"
        )
    return build_forecast_question(matches.iloc[0], as_of_date=as_of_date)


def load_options_for_question(path: PathLike, question_id: str) -> list[str]:
    """Return the answer options for a question from the forecasts CSV.

    The questions CSV doesn't carry the option set; it's defined implicitly
    by the (human) forecast rows in the forecasts CSV — one row per option.
    The orchestrator uses this to learn the option set a benchmark question
    must be forecast over. Options are returned in first-seen order with
    duplicates removed. Returns ``[]`` when the question_id has no rows.
    """
    df = load_forecasts(path)
    rows = df[df["question_id"].astype(str).str.strip() == question_id.strip()]
    options: list[str] = []
    for opt in rows["option"].astype(str).str.strip().tolist():
        if opt and opt not in options:
            options.append(opt)
    return options
