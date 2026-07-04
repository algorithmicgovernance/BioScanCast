"""Tests for resolution/target-date parsing in the eval-stage loaders.

Covers the phrasings the BFG question set uses — "by <date>", "before
<date>", "from X to Y" ranges, month-only, and the resolution_criteria
fallback — plus an end-to-end check that every BFG question now yields a
target_date (previously q6/q9/q11 returned None and needed a manual
--target-date override).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from bioscancast.stages.evaluation.loaders import (
    _parse_target_date,
    load_question_by_id,
)

QUESTIONS_CSV = "bioscancast/stages/evaluation/bioscancast_questions.csv"


def _utc(y, m, d):
    return datetime(y, m, d, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("...reported in the US by February 28, 2025, according...", _utc(2025, 2, 28)),
        ("...by Feb 28 2025", _utc(2025, 2, 28)),
        ("...by January 1st, 2026?", _utc(2026, 1, 1)),
        ("...a new outbreak before May 1st, 2025?", _utc(2025, 5, 1)),
        # Range: take the END date (introduced by "to"), not the "from" start.
        (
            "...reported globally from January 1st, 2025, to May 1st, 2025?",
            _utc(2025, 5, 1),
        ),
        # Month-only → 1st of the following month.
        ("...by February 2025", _utc(2025, 3, 1)),
        ("...by December 2025", _utc(2026, 1, 1)),
        # No date present.
        ("What is the most likely cause of the current Congo outbreak?", None),
        ("", None),
    ],
)
def test_parse_target_date_phrasings(text, expected):
    assert _parse_target_date(text) == expected


# Expected resolution dates for every BFG question (q6/q9 use prepositions
# other than "by"; q11's date is only in the resolution_criteria).
_EXPECTED = {
    "q1": _utc(2025, 2, 28),
    "q2": _utc(2026, 1, 1),
    "q3": _utc(2025, 2, 28),
    "q4": _utc(2025, 2, 28),
    "q5": _utc(2025, 5, 1),
    "q6": _utc(2025, 5, 1),
    "q7": _utc(2025, 2, 28),
    "q8": _utc(2025, 5, 1),
    "q9": _utc(2025, 5, 1),
    "q10": _utc(2025, 6, 30),
    "q11": _utc(2025, 3, 31),
}


@pytest.mark.parametrize("qid,expected", sorted(_EXPECTED.items()))
def test_every_bfg_question_has_target_date(qid, expected):
    q = load_question_by_id(QUESTIONS_CSV, qid)
    assert q.target_date == expected, f"{qid}: got {q.target_date}, want {expected}"
