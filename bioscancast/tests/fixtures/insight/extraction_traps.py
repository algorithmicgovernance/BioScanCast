"""Labeled trap-set for diagnosing the insight stage's "lurch" failure.

The historical-replay benchmark traced its recurring single-cutoff
trajectory "lurches" to the insight stage extracting a *bad number* and
filing it as an observed count. Two concrete examples surfaced in the
historical-replay benchmark:

* a **projection** ("one in 20 simulations projected an outbreak
  exceeding 10,000 cases") filed as ``confirmed_cases=10000``;
* a **wrong-scope** real number (``confirmed_cases=903`` at confidence
  1.0) that is not a plausible US human-case count for the window.

These are *different* mechanisms — one needs a representational slot for
non-observed values, the other needs scope-aware reasoning and honest
confidence — so "add a projected bucket" fixes only the first. This
module is the instrument that tells them apart: a small set of
``(chunk, question, expectation)`` cases plus a pure-Python scorer that
runs over the records a real extractor produces and reports how each
trap was handled.

The scorer is **forward-compatible** with the planned ``value_basis``
field: it reads ``getattr(record, "value_basis", None)`` so the same
harness measures the pre-fix failure rate today and the post-fix
improvement once the schema/prompt change lands.

No network or LLM calls happen here — this file is data + pure functions.
``scripts/eval_extraction_traps.py`` and
``bioscancast/tests/test_insight_extraction_traps.py`` drive it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

from bioscancast.schemas import Document, DocumentChunk, InsightRecord
from bioscancast.stages.filtering.models import ForecastQuestion


# Event types that represent an *observed* count in the current schema.
# Filing a projected or wrong-scope number under one of these (with an
# observed ``value_basis``) is the core failure the trap-set detects.
OBSERVED_COUNT_EVENT_TYPES = frozenset({"case_count", "death_count"})


# ---------------------------------------------------------------------------
# Trap / expectation data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrapValue:
    """A number planted in a chunk that a correct extractor must NOT treat
    as an observed, question-scoped count.

    Each flag encodes one way the value can be mishandled. A correct
    extraction may legitimately extract the value *with the right framing*
    (e.g. a projection tagged as projected, a global figure tagged
    global) or omit it entirely — both count as "handled". Only the
    specific mishandling described by the set flags is a failure.
    """

    value: float
    issue: str
    """Short mechanism label: 'projection' | 'offscope_global' |
    'weekly' | 'subnational'."""

    must_not_be_observed_count: bool = False
    """The value must not appear as an observed count event
    (``case_count``/``death_count`` with an observed ``value_basis``).
    Set for projections/modeled figures."""

    must_not_be_scoped_to_region: bool = False
    """The value must not be extracted as an observed count whose
    ``iso_country_code`` equals the question's region. Set for figures
    that belong to a different geographic scope (e.g. a global cumulative
    total against a US question)."""

    must_retain_qualifier: Tuple[str, ...] = ()
    """If the value is extracted, at least one of these tokens must appear
    in ``location``/``summary``/``metric_name``/``metric_unit`` so the
    figure cannot be misread as the question-relevant total. Set for
    weekly-vs-cumulative and sub-national-vs-national traps."""

    expected_basis: Optional[str] = None
    """The ``value_basis`` a correct (post-fix) extractor should assign,
    e.g. ``"projected"``. Used only for the post-fix basis-accuracy
    metric; ignored when the field is absent."""


@dataclass(frozen=True)
class ExtractionTrap:
    """One labeled extraction case: a chunk + question + expectations."""

    label: str
    document: Document
    chunk: DocumentChunk
    question: ForecastQuestion
    trap_values: Tuple[TrapValue, ...] = ()
    control_values: Tuple[float, ...] = ()
    """Numbers that a correct extractor SHOULD surface as observed counts
    (recall guard — stops a fix that just suppresses everything)."""
    region_iso: Optional[str] = None
    """ISO alpha-2 of the question's region, for the scope checks."""
    notes: str = ""


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass
class MishandledValue:
    value: float
    issue: str
    reasons: List[str]


@dataclass
class TrapResult:
    label: str
    trap_values_total: int
    mishandled: List[MishandledValue] = field(default_factory=list)
    trap_confidences: List[float] = field(default_factory=list)
    controls_total: int = 0
    controls_recalled: int = 0
    basis_present: int = 0
    basis_correct: int = 0

    @property
    def ok(self) -> bool:
        """True when no trap value was mishandled and every control was
        recalled."""
        return not self.mishandled and self.controls_recalled == self.controls_total


def _matches_value(record: InsightRecord, value: float, tol: float = 0.5) -> bool:
    mv = record.metric_value
    return mv is not None and abs(float(mv) - value) <= tol


def _is_observed_count(record: InsightRecord) -> bool:
    """Whether the record presents a number as an observed count.

    Forward-compatible with ``value_basis``: an explicit non-observed
    basis (e.g. ``"projected"``) means the record is *not* an observed
    count, so a correctly-tagged projection no longer trips the guard.
    Pre-fix (no such field) every count-typed record is treated as
    observed.
    """
    basis = getattr(record, "value_basis", None)
    if basis is not None and basis != "observed":
        return False
    return (
        record.event_type in OBSERVED_COUNT_EVENT_TYPES
        and record.metric_value is not None
    )


def _scoped_to_region(record: InsightRecord, region_iso: Optional[str]) -> bool:
    return region_iso is not None and record.iso_country_code == region_iso


def _qualifier_present(record: InsightRecord, tokens: Sequence[str]) -> bool:
    haystack = " ".join(
        part
        for part in (
            record.location,
            record.summary,
            record.metric_name,
            record.metric_unit,
        )
        if part
    ).lower()
    return any(token.lower() in haystack for token in tokens)


def score_extraction(
    records: Sequence[InsightRecord], trap: ExtractionTrap
) -> TrapResult:
    """Score one trap's extraction output against its expectations."""
    result = TrapResult(
        label=trap.label,
        trap_values_total=len(trap.trap_values),
        controls_total=len(trap.control_values),
    )

    for tv in trap.trap_values:
        matches = [r for r in records if _matches_value(r, tv.value)]
        for r in matches:
            result.trap_confidences.append(r.confidence)
            basis = getattr(r, "value_basis", None)
            if basis is not None:
                result.basis_present += 1
                if tv.expected_basis and basis == tv.expected_basis:
                    result.basis_correct += 1

        reasons: List[str] = []
        if tv.must_not_be_observed_count and any(
            _is_observed_count(r) for r in matches
        ):
            reasons.append("filed-as-observed-count")
        if tv.must_not_be_scoped_to_region and any(
            _is_observed_count(r) and _scoped_to_region(r, trap.region_iso)
            for r in matches
        ):
            reasons.append("mis-scoped-to-region")
        if (
            tv.must_retain_qualifier
            and matches
            and not any(_qualifier_present(r, tv.must_retain_qualifier) for r in matches)
        ):
            reasons.append("qualifier-dropped")
        if reasons:
            result.mishandled.append(MishandledValue(tv.value, tv.issue, reasons))

    result.controls_recalled = sum(
        1
        for cv in trap.control_values
        if any(_is_observed_count(r) and _matches_value(r, cv) for r in records)
    )
    return result


def aggregate(results: Sequence[TrapResult]) -> dict:
    """Roll trap results up into headline metrics for a diagnosis report."""
    trap_values_total = sum(r.trap_values_total for r in results)
    mishandled = [m for r in results for m in r.mishandled]
    reason_counts = {
        "filed-as-observed-count": 0,
        "mis-scoped-to-region": 0,
        "qualifier-dropped": 0,
    }
    for m in mishandled:
        for reason in m.reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    controls_total = sum(r.controls_total for r in results)
    controls_recalled = sum(r.controls_recalled for r in results)
    trap_confidences = [c for r in results for c in r.trap_confidences]
    basis_present = sum(r.basis_present for r in results)
    basis_correct = sum(r.basis_correct for r in results)

    return {
        "traps": len(results),
        "traps_clean": sum(1 for r in results if r.ok),
        "trap_values_total": trap_values_total,
        "trap_values_mishandled": len(mishandled),
        "false_observed_count": reason_counts["filed-as-observed-count"],
        "mis_scoped_to_region": reason_counts["mis-scoped-to-region"],
        "qualifier_dropped": reason_counts["qualifier-dropped"],
        "control_recall": (
            controls_recalled / controls_total if controls_total else 1.0
        ),
        "controls_recalled": controls_recalled,
        "controls_total": controls_total,
        "mean_trap_confidence": (
            sum(trap_confidences) / len(trap_confidences) if trap_confidences else 0.0
        ),
        # Populated only once records carry a ``value_basis`` field.
        "basis_present": basis_present,
        "basis_correct": basis_correct,
    }


# ---------------------------------------------------------------------------
# Trap fixtures
# ---------------------------------------------------------------------------


def _doc(doc_id: str, url: str, domain: str, chunk: DocumentChunk) -> Document:
    """Minimal single-chunk Document wrapper for a trap chunk."""
    return Document(
        id=doc_id,
        result_id=f"r-{doc_id}",
        source_url=url,
        domain=domain,
        fetched_at=datetime(2026, 1, 31, 12, 0, 0),
        document_type="html",
        status="success",
        title=doc_id,
        chunks=[chunk],
    )


# --- Question reused by the H5N1 traps (matches the BFG q1/q3 framing) ---

_Q_H5N1_US_CASES = ForecastQuestion(
    id="trap-h5n1-us-cases",
    text=(
        "How many confirmed human cases of H5N1 will be reported in the "
        "United States by December 2026?"
    ),
    created_at=datetime(2026, 1, 31),
    target_date=datetime(2026, 12, 31),
    region="United States",
    pathogen="H5N1",
    event_type="case_count",
)


# Trap A — projection-as-observed (the live-demo "10,000 cases" case).
_CHUNK_PROJECTION = DocumentChunk(
    chunk_id="trap-projection-c0",
    chunk_index=0,
    text=(
        "As of 30 January 2026, 47 confirmed human cases of H5N1 have been "
        "reported in the United States since 2024. A modeling study published "
        "this week found that, under a worst-case transmission scenario, one "
        "in 20 simulations projected an outbreak exceeding 10,000 human cases "
        "by the end of 2026; the authors stressed these figures are "
        "hypothetical and not observed counts."
    ),
    chunk_type="prose",
    heading="Situation summary",
)

TRAP_PROJECTION = ExtractionTrap(
    label="projection_as_observed",
    document=_doc(
        "doc-trap-projection",
        "https://cdc.gov/bird-flu/projection.html",
        "cdc.gov",
        _CHUNK_PROJECTION,
    ),
    chunk=_CHUNK_PROJECTION,
    question=_Q_H5N1_US_CASES,
    trap_values=(
        TrapValue(
            value=10000,
            issue="projection",
            must_not_be_observed_count=True,
            expected_basis="projected",
        ),
    ),
    control_values=(47,),
    region_iso="US",
    notes="Projection must not be filed as an observed US human-case count.",
)


# Trap A2 — projection without a "hypothetical" disclaimer (harder).
# The same projection failure but phrased assertively, the way a careless
# extractor is most likely to grab it as an observed count. Probes the
# representational gap directly (Trap A's explicit "not observed counts"
# cue made it too easy).
_CHUNK_PROJECTION_HARD = DocumentChunk(
    chunk_id="trap-projection-hard-c0",
    chunk_index=0,
    text=(
        "CDC modeling released today estimates that, absent additional "
        "interventions, the United States could see as many as 10,000 human "
        "H5N1 cases by the end of 2026. As of 30 January 2026, 47 human cases "
        "have been confirmed nationwide."
    ),
    chunk_type="prose",
    heading="Modeling outlook",
)

TRAP_PROJECTION_HARD = ExtractionTrap(
    label="projection_no_disclaimer",
    document=_doc(
        "doc-trap-projection-hard",
        "https://cdc.gov/bird-flu/modeling.html",
        "cdc.gov",
        _CHUNK_PROJECTION_HARD,
    ),
    chunk=_CHUNK_PROJECTION_HARD,
    question=_Q_H5N1_US_CASES,
    trap_values=(
        TrapValue(
            value=10000,
            issue="projection",
            must_not_be_observed_count=True,
            expected_basis="projected",
        ),
    ),
    control_values=(47,),
    region_iso="US",
    notes="Assertively-phrased projection; no 'hypothetical' cue.",
)


# Trap B — wrong-scope global cumulative (the "903" case).
_CHUNK_OFFSCOPE_GLOBAL = DocumentChunk(
    chunk_id="trap-offscope-c0",
    chunk_index=0,
    text=(
        "Globally, a cumulative total of 903 human cases of avian influenza "
        "A(H5N1), including 464 deaths, have been reported to WHO from 2003 "
        "to 2024 across 24 countries. In the United States, human infections "
        "remain rare and are mostly linked to occupational exposure on dairy "
        "and poultry farms."
    ),
    chunk_type="prose",
    heading="Global epidemiology",
)

TRAP_OFFSCOPE_GLOBAL = ExtractionTrap(
    label="offscope_global_cumulative",
    document=_doc(
        "doc-trap-offscope",
        "https://who.int/h5n1/global-summary.html",
        "who.int",
        _CHUNK_OFFSCOPE_GLOBAL,
    ),
    chunk=_CHUNK_OFFSCOPE_GLOBAL,
    question=_Q_H5N1_US_CASES,
    trap_values=(
        TrapValue(
            value=903,
            issue="offscope_global",
            must_not_be_scoped_to_region=True,
            must_retain_qualifier=("global", "globally", "world", "2003", "who"),
        ),
    ),
    control_values=(),
    region_iso="US",
    notes="903 is a global cumulative-since-2003 figure; must not be tagged US.",
)


# Trap C — weekly-vs-cumulative temporal scope.
_CHUNK_WEEKLY = DocumentChunk(
    chunk_id="trap-weekly-c0",
    chunk_index=0,
    text=(
        "As of 12 April 2026, a cumulative total of 1,847 measles cases have "
        "been reported in Romania this year. During the most recent "
        "epidemiological week (week 15), 56 new cases were recorded, a "
        "decline from the previous week."
    ),
    chunk_type="prose",
    heading="Measles surveillance",
)

_Q_MEASLES_RO = ForecastQuestion(
    id="trap-measles-ro-cumulative",
    text=(
        "Will the cumulative number of measles cases reported in Romania "
        "exceed 2,000 by 30 April 2026?"
    ),
    created_at=datetime(2026, 4, 12),
    target_date=datetime(2026, 4, 30),
    region="Romania",
    pathogen="measles",
    event_type="case_count",
)

TRAP_WEEKLY = ExtractionTrap(
    label="weekly_vs_cumulative",
    document=_doc(
        "doc-trap-weekly",
        "https://ecdc.europa.eu/measles/romania.html",
        "ecdc.europa.eu",
        _CHUNK_WEEKLY,
    ),
    chunk=_CHUNK_WEEKLY,
    question=_Q_MEASLES_RO,
    trap_values=(
        TrapValue(
            value=56,
            issue="weekly",
            must_retain_qualifier=("week", "weekly", "new"),
        ),
    ),
    control_values=(1847,),
    region_iso="RO",
    notes="56 is a weekly increment; must carry the weekly/new qualifier.",
)


# Trap D — sub-national-vs-national.
_CHUNK_SUBNATIONAL = DocumentChunk(
    chunk_id="trap-subnational-c0",
    chunk_index=0,
    text=(
        "Texas alone has reported 312 affected herds, the highest of any "
        "state. Nationwide, USDA has now confirmed H5N1 in a cumulative "
        "1,043 livestock herds across 16 states as of 30 January 2026."
    ),
    chunk_type="prose",
    heading="Livestock detections",
)

_Q_H5N1_US_HERDS = ForecastQuestion(
    id="trap-h5n1-us-herds",
    text=(
        "Will H5N1 be confirmed in more than 1,500 US livestock herds by "
        "June 2026?"
    ),
    created_at=datetime(2026, 1, 31),
    target_date=datetime(2026, 6, 30),
    region="United States",
    pathogen="H5N1",
    event_type="case_count",
)

TRAP_SUBNATIONAL = ExtractionTrap(
    label="subnational_vs_national",
    document=_doc(
        "doc-trap-subnational",
        "https://usda.gov/h5n1/herds.html",
        "usda.gov",
        _CHUNK_SUBNATIONAL,
    ),
    chunk=_CHUNK_SUBNATIONAL,
    question=_Q_H5N1_US_HERDS,
    trap_values=(
        TrapValue(
            value=312,
            issue="subnational",
            must_retain_qualifier=("texas", "state"),
        ),
    ),
    control_values=(1043,),
    region_iso="US",
    notes="312 is a Texas subtotal; must not be read as the national figure.",
)


# Trap E — clean observed control (no trap; recall guard).
_CHUNK_CLEAN = DocumentChunk(
    chunk_id="trap-clean-c0",
    chunk_index=0,
    text=(
        "On 12 February 2025, the Ministry of Health of Uganda notified WHO of "
        "an outbreak of Sudan virus disease in Mubende district. As of 15 "
        "February 2025, a total of 9 confirmed cases including 3 deaths have "
        "been reported."
    ),
    chunk_type="prose",
    heading="Epidemiological summary",
)

_Q_SUDAN = ForecastQuestion(
    id="trap-sudan-outbreak",
    text=(
        "Will the Sudan virus outbreak in Uganda exceed 50 confirmed cases "
        "by March 2025?"
    ),
    created_at=datetime(2025, 2, 16),
    target_date=datetime(2025, 3, 31),
    region="Uganda",
    pathogen="Sudan virus",
    event_type="case_count",
)

TRAP_CLEAN_CONTROL = ExtractionTrap(
    label="clean_observed_control",
    document=_doc(
        "doc-trap-clean",
        "https://who.int/don/svd-uganda.html",
        "who.int",
        _CHUNK_CLEAN,
    ),
    chunk=_CHUNK_CLEAN,
    question=_Q_SUDAN,
    trap_values=(),
    control_values=(9,),
    region_iso="UG",
    notes="Genuine observed count; a good extractor must surface it confidently.",
)


ALL_TRAPS: List[ExtractionTrap] = [
    TRAP_PROJECTION,
    TRAP_PROJECTION_HARD,
    TRAP_OFFSCOPE_GLOBAL,
    TRAP_WEEKLY,
    TRAP_SUBNATIONAL,
    TRAP_CLEAN_CONTROL,
]
