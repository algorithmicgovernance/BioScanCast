"""Evidence-coverage analyzer for the BFG summer-2026 forecast-quality pass.

Reads the per-stage artifacts an evidence-only pipeline run writes to
``data/runs/{qid}/{run_id}/`` (search.json, filtered.json, documents.json,
insight.json, manifest.json, optionally forecast.json) and produces, per
question, the diagnosis the spec asks for:

  * evidence sufficiency  — is there a usable current-value record, its
    count_basis, confidence, source URL + quote (surfaced for spot-check);
  * source attribution    — organic vs dashboard behind each insight record,
    and the number of independent sources behind the top record;
  * quantity              — organic vs dashboard results returned / surviving;
  * quality               — deterministic keyword_overlap of the organic pool
    vs its survivors, plus (optional) a capped gpt-4o-mini on-topic judge;
  * classification+cause  — well_supported / dashboard_only / under_supported,
    and for weak questions an attributed cause (search-recall / filter-recall /
    extraction).

Provenance chain used for organic-vs-dashboard attribution:

    InsightRecord.sources[].document_id
        -> Document.id            (documents.json)
        -> Document.result_id     (FK to FilteredDocument.result_id)
        -> SearchResult.id        (== result_id; search.json)
        -> retrieval_reason == "dashboard_lookup"  => dashboard, else organic

A FilteredDocument is also flagged dashboard when its selection_reasons
contains "dashboard_lookup_bypass".

Deterministic-only by default (free). Pass --llm-judge to add the on-topic
judge (costs a few gpt-4o-mini calls per question).

Usage:
    python scripts/analyze_evidence_coverage.py --all \
        --out data/investigations/bfg_evidence_audit
    python scripts/analyze_evidence_coverage.py bfg_q17          # one question
    python scripts/analyze_evidence_coverage.py --run-dir data/runs/bfg_q17/2026...
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Reuse the pipeline's own relevance signal so pool/survivor "quality" matches
# what the search and filter stages actually score on.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bioscancast.stages.filtering.utils import keyword_overlap_score, tokenize  # noqa: E402


DASHBOARD_REASON = "dashboard_lookup"
DASHBOARD_BYPASS = "dashboard_lookup_bypass"

# Confidence at/above which an extracted record is treated as "high
# confidence" for the evidence-sufficiency bar.
HIGH_CONFIDENCE = 0.6

# Below this max keyword-overlap in the organic *pool*, we treat the pool as
# essentially devoid of on-topic organic content (a search-recall failure).
SEARCH_RECALL_POOL_OVERLAP = 0.18

# An organic pool result at/above this overlap is "clearly on-topic"; if such a
# result is in the pool but not among survivors, that points at filter-recall.
ON_TOPIC_OVERLAP = 0.30

# Question types that resolve to a numeric count/range and therefore need a
# metric-bearing anchor record for evidence sufficiency.
NUMERIC_TYPES = {"range"}

# Low floor below which a metric record is treated as too weak to be an anchor
# candidate at all. We deliberately keep this well under HIGH_CONFIDENCE so a
# *present-but-under-confident* correct anchor (e.g. q17's Global cumulative
# extracted at 0.5 while off-scope regional rows got 0.85) is still detected —
# the gap between "anchor present" and "anchor present AND high-confidence" is
# itself a finding (an extraction/confidence-calibration cause, not search/filter).
ANCHOR_FLOOR = 0.30

# Geography synonym sets for scope-matching a record's ``location`` against the
# geography a range question actually asks about. Substring match on the
# lowercased location, plus token overlap.
GEO_SYNONYMS: dict[str, set[str]] = {
    "global": {"global", "world", "worldwide", "globally", "multi-country",
               "multicountry", "international", "total", "overall"},
    "us": {"united states", "u.s.", "usa", "us"},
    "americas": {"americas", "region of the americas", "latin america",
                 "caribbean", "paho", "south america", "central america"},
    "eu_eea": {"eu", "eea", "european union", "europe", "european"},
    "drc_uganda": {"democratic republic", "drc", "congo", "uganda", "combined",
                   "total"},
}

# Authoritative resolution-source domain per question (from the CSV
# resolution_criteria). Used to measure route/coverage: did the pipeline
# actually inject a dashboard on the domain that resolves the question?
# Feeds issue #41 (route_sources single-bucket misroute rate).
EXPECT_RESOLUTION_DOMAIN: dict[str, str] = {
    "bfg_q1": "who.int", "bfg_q2": "who.int", "bfg_q3": "who.int",
    "bfg_q4": "who.int", "bfg_q5": "who.int", "bfg_q6": "who.int",
    "bfg_q7": "who.int", "bfg_q8": "who.int", "bfg_q9": "aphis.usda.gov",
    "bfg_q10": "who.int", "bfg_q11": "who.int", "bfg_q12": "who.int",
    "bfg_q13": "who.int", "bfg_q14": "cdc.gov", "bfg_q15": "paho.org",
    "bfg_q16": "cdc.gov", "bfg_q17": "who.int", "bfg_q18": "paho.org",
    "bfg_q19": "ecdc.europa.eu", "bfg_q20": "paho.org", "bfg_q21": "who.int",
    "bfg_q22": "polioeradication.org", "bfg_q23": "who.int",
    "bfg_q24": "who.int", "bfg_q25": "who.int",
}

# Expected geography per range question (which scope the anchor must carry).
GEO_EXPECT: dict[str, str] = {
    "bfg_q1": "drc_uganda", "bfg_q2": "drc_uganda", "bfg_q6": "global",
    "bfg_q9": "us", "bfg_q13": "global", "bfg_q14": "us", "bfg_q16": "us",
    "bfg_q17": "global", "bfg_q18": "americas", "bfg_q19": "eu_eea",
    "bfg_q20": "americas", "bfg_q22": "global", "bfg_q25": "global",
}


def _scope_match(location: Optional[str], geo_key: Optional[str]) -> bool:
    """Does this record's location match the question's expected geography?"""
    if not geo_key:
        return True  # no expectation encoded -> don't penalise
    loc = (location or "").strip().lower()
    if not loc:
        return False
    syns = GEO_SYNONYMS.get(geo_key, set())
    if any(s in loc for s in syns):
        return True
    loc_tokens = set(tokenize(loc))
    return bool(loc_tokens & {t for s in syns for t in s.split()})

# Per-question hint for the count_basis we'd expect the anchor to carry.
# "cumulative" = running total to date; "incident" = new within a stated
# window. Advisory only — surfaced next to the record's actual basis so a
# mismatch is visible; never used to hard-fail a record.
EXPECTED_BASIS: dict[str, str] = {
    "bfg_q1": "cumulative", "bfg_q2": "cumulative", "bfg_q6": "incident",
    "bfg_q9": "cumulative", "bfg_q13": "incident", "bfg_q14": "cumulative",
    "bfg_q16": "cumulative", "bfg_q17": "cumulative", "bfg_q18": "cumulative",
    "bfg_q19": "cumulative", "bfg_q20": "cumulative", "bfg_q22": "cumulative",
    "bfg_q25": "incident",
}


# --------------------------------------------------------------------------
# artifact loading
# --------------------------------------------------------------------------


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class RunArtifacts:
    qid: str
    run_dir: Path
    question: dict
    search: list[dict]
    filtered: list[dict]
    documents: list[dict]
    insight: dict
    manifest: dict
    forecast: Optional[dict]

    @property
    def records(self) -> list[dict]:
        return (self.insight or {}).get("records", []) or []


def load_run(run_dir: Path) -> RunArtifacts:
    return RunArtifacts(
        qid=run_dir.parent.name,
        run_dir=run_dir,
        question=_load_json(run_dir / "question.json") or {},
        search=_load_json(run_dir / "search.json") or [],
        filtered=_load_json(run_dir / "filtered.json") or [],
        documents=_load_json(run_dir / "documents.json") or [],
        insight=_load_json(run_dir / "insight.json") or {},
        manifest=_load_json(run_dir / "manifest.json") or {},
        forecast=_load_json(run_dir / "forecast.json"),
    )


def latest_run_dir(runs_root: Path, qid: str) -> Optional[Path]:
    qdir = runs_root / qid
    if not qdir.is_dir():
        return None
    runs = [d for d in qdir.iterdir() if d.is_dir()]
    if not runs:
        return None
    # Run dirs are UTC timestamps (YYYYmmdd_HHMMSS) so lexical == chronological.
    return sorted(runs, key=lambda d: d.name)[-1]


# --------------------------------------------------------------------------
# query terms (rebuilt from question.json, mirroring
# heuristics.build_query_terms without needing a live ForecastQuestion)
# --------------------------------------------------------------------------


def query_terms(question: dict) -> list[str]:
    terms = [question.get("text") or ""]
    for key in ("pathogen", "region", "event_type", "resolution_criteria"):
        val = question.get(key)
        if val:
            terms.append(str(val))
    return [t for t in terms if t]


# --------------------------------------------------------------------------
# organic vs dashboard classification + provenance
# --------------------------------------------------------------------------


def _is_dashboard_search(sr: dict) -> bool:
    return (sr.get("retrieval_reason") or "") .strip().lower() == DASHBOARD_REASON \
        or (sr.get("engine") or "").lower() == "dashboard"


def _is_dashboard_filtered(fd: dict, dashboard_result_ids: set[str]) -> bool:
    if DASHBOARD_BYPASS in (fd.get("selection_reasons") or []):
        return True
    return fd.get("result_id") in dashboard_result_ids


def _rel(text: str, terms: list[str]) -> float:
    return keyword_overlap_score(text, terms)


@dataclass
class Provenance:
    """Maps insight records back to organic/dashboard origins."""

    dashboard_result_ids: set[str]
    organic_result_ids: set[str]
    # document_id -> ("dashboard"|"organic"|"unknown", source_url, domain)
    doc_origin: dict[str, tuple[str, str, str]]


def build_provenance(art: RunArtifacts) -> Provenance:
    dash_ids = {sr["id"] for sr in art.search if _is_dashboard_search(sr)}
    org_ids = {sr["id"] for sr in art.search if not _is_dashboard_search(sr)}

    # Some filtered docs are dashboards only detectable via the bypass reason
    # (belt and suspenders with the search-side reason).
    for fd in art.filtered:
        if DASHBOARD_BYPASS in (fd.get("selection_reasons") or []):
            rid = fd.get("result_id")
            if rid:
                dash_ids.add(rid)
                org_ids.discard(rid)

    doc_origin: dict[str, tuple[str, str, str]] = {}
    for doc in art.documents:
        rid = doc.get("result_id")
        if rid in dash_ids:
            origin = "dashboard"
        elif rid in org_ids:
            origin = "organic"
        else:
            origin = "unknown"
        doc_origin[doc.get("id")] = (
            origin, doc.get("source_url") or "", (doc.get("domain") or "").lower()
        )

    return Provenance(dash_ids, org_ids, doc_origin)


@dataclass
class RecordProvenance:
    record: dict
    origins: set[str] = field(default_factory=set)      # {"organic","dashboard"}
    source_domains: set[str] = field(default_factory=set)
    source_urls: list[str] = field(default_factory=list)

    @property
    def from_organic(self) -> bool:
        return "organic" in self.origins

    @property
    def from_dashboard(self) -> bool:
        return "dashboard" in self.origins

    @property
    def n_independent_sources(self) -> int:
        # Independent = distinct source domain behind the record.
        return len(self.source_domains) or (1 if self.source_urls else 0)


def record_provenance(rec: dict, prov: Provenance) -> RecordProvenance:
    rp = RecordProvenance(record=rec)
    for src in rec.get("sources", []) or []:
        doc_id = src.get("document_id")
        url = src.get("source_url") or ""
        if url:
            rp.source_urls.append(url)
        origin, doc_url, domain = prov.doc_origin.get(doc_id, ("unknown", "", ""))
        if origin in ("organic", "dashboard"):
            rp.origins.add(origin)
        dom = domain or _domain_of(url)
        if dom:
            rp.source_domains.add(dom)
    return rp


def _domain_of(url: str) -> str:
    if not url:
        return ""
    u = url.split("//", 1)[-1]
    return u.split("/", 1)[0].lower().removeprefix("www.")


# --------------------------------------------------------------------------
# usability + classification
# --------------------------------------------------------------------------


def _basis_ok(rec: dict, qid: str) -> bool:
    """Record's count_basis matches the expected basis (advisory)."""
    exp = EXPECTED_BASIS.get(qid)
    if not exp:
        return True
    return (rec.get("count_basis") or "") == exp


def _is_usable(rec: dict, qtype: str, qid: str) -> bool:
    """A record is 'usable' as a current-value anchor for this question.

    Range questions require a metric value that clears the anchor floor AND
    is scoped to the question's geography (a Global cholera question is not
    answered by an Angola row). The strict HIGH_CONFIDENCE bar is applied
    separately as ``evidence_sufficient`` so a present-but-under-confident
    anchor is still counted as usable-but-weak rather than vanishing.
    """
    if qtype in NUMERIC_TYPES:
        if rec.get("metric_value") is None:
            return False
        if (rec.get("confidence") or 0.0) < ANCHOR_FLOOR:
            return False
        return _scope_match(rec.get("location"), GEO_EXPECT.get(qid))
    # binary / categorical: any high-confidence on-topic record with a metric
    # OR a substantive summary counts as usable evidence of current status.
    if (rec.get("confidence") or 0.0) < HIGH_CONFIDENCE:
        return False
    return rec.get("metric_value") is not None or bool(rec.get("summary"))


def pick_top_record(
    records: list[dict], qtype: str, qid: str, prov: Provenance
) -> Optional[RecordProvenance]:
    """Best current-value anchor. For range questions, rank by scope match →
    basis match → confidence → corroboration, so a correctly-scoped anchor
    beats a higher-confidence off-scope row."""
    if not records:
        return None
    rps = [record_provenance(r, prov) for r in records]
    geo = GEO_EXPECT.get(qid)

    def sort_key(rp: RecordProvenance):
        r = rp.record
        has_metric = 1 if r.get("metric_value") is not None else 0
        if qtype in NUMERIC_TYPES:
            return (
                1 if _scope_match(r.get("location"), geo) else 0,
                1 if _basis_ok(r, qid) else 0,
                has_metric,
                r.get("confidence") or 0.0,
                rp.n_independent_sources,
                1 if rp.from_organic else 0,
            )
        return (
            0, 0, 0,
            r.get("confidence") or 0.0,
            rp.n_independent_sources,
            1 if rp.from_organic else 0,
        )

    usable = [rp for rp in rps if _is_usable(rp.record, qtype, qid)]
    pool = usable or rps
    return sorted(pool, key=sort_key, reverse=True)[0]


@dataclass
class Diagnosis:
    qid: str
    run_id: str
    qtype: str
    topic: str
    # quantity
    organic_returned: int
    dashboard_returned: int
    organic_survivors: int
    dashboard_survivors: int
    # dashboard routing / coverage (issue #41)
    injected_dash_domains: str
    expected_resolution_domain: str
    resolution_source_injected: bool
    # quality
    pool_overlap_mean: float
    pool_overlap_max: float
    survivor_overlap_mean: float
    llm_pool_ontopic: Optional[float]
    llm_survivor_ontopic: Optional[float]
    # evidence / attribution
    n_records: int
    n_usable: int
    insight_from_organic: int
    insight_from_dashboard: int
    top_metric_value: Optional[float]
    top_metric_name: Optional[str]
    top_count_basis: Optional[str]
    expected_basis: Optional[str]
    top_confidence: Optional[float]
    top_n_sources: int
    top_origin: str
    top_source_url: Optional[str]
    top_quote: Optional[str]
    top_scope_ok: bool
    # verdicts
    anchor_present: bool          # a scope-matched metric record exists (any conf >= floor)
    evidence_sufficient: bool     # ...AND high-confidence with the right basis (spec headline)
    single_fragile_source: bool
    classification: str
    cause: Optional[str]
    cause_detail: str

    def as_row(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 4)
        return d


def diagnose(art: RunArtifacts, llm_judge=None) -> Diagnosis:
    q = art.question
    qtype = (q.get("question_type") or _infer_qtype(art)) or ""
    terms = query_terms(q)
    prov = build_provenance(art)

    # ---- quantity ----
    org_search = [s for s in art.search if not _is_dashboard_search(s)]
    dash_search = [s for s in art.search if _is_dashboard_search(s)]
    org_filt = [f for f in art.filtered if not _is_dashboard_filtered(f, prov.dashboard_result_ids)]
    dash_filt = [f for f in art.filtered if _is_dashboard_filtered(f, prov.dashboard_result_ids)]

    # ---- dashboard routing / coverage (issue #41) ----
    injected_domains = sorted({(s.get("domain") or "").lower() for s in dash_search})
    injected_domains = [d for d in injected_domains if d]
    exp_domain = EXPECT_RESOLUTION_DOMAIN.get(art.qid, "")
    resolution_injected = any(exp_domain and exp_domain in d for d in injected_domains)

    # ---- quality (deterministic keyword overlap) ----
    def _txt_search(s: dict) -> str:
        return f"{s.get('title','')} {s.get('snippet','')} {s.get('domain','')}"

    def _txt_filt(f: dict) -> str:
        return f"{f.get('title','')} {f.get('snippet','')} {f.get('domain','')}"

    pool_overlaps = [_rel(_txt_search(s), terms) for s in org_search]
    surv_overlaps = [_rel(_txt_filt(f), terms) for f in org_filt]
    pool_mean = sum(pool_overlaps) / len(pool_overlaps) if pool_overlaps else 0.0
    pool_max = max(pool_overlaps) if pool_overlaps else 0.0
    surv_mean = sum(surv_overlaps) / len(surv_overlaps) if surv_overlaps else 0.0

    # ---- optional LLM on-topic judge ----
    llm_pool = llm_surv = None
    if llm_judge is not None:
        llm_surv = llm_judge.judge_fraction(
            q, [(_txt_filt(f), f.get("url", "")) for f in org_filt], label="survivors"
        )
        sampled = sorted(
            org_search, key=lambda s: s.get("search_stage_score", 0.0), reverse=True
        )[:15]
        llm_pool = llm_judge.judge_fraction(
            q, [(_txt_search(s), s.get("url", "")) for s in sampled], label="pool_top15"
        )

    # ---- records + provenance ----
    records = art.records
    rps = [record_provenance(r, prov) for r in records]
    usable_rps = [rp for rp in rps if _is_usable(rp.record, qtype, art.qid)]
    n_usable = len(usable_rps)
    insight_from_organic = sum(1 for rp in rps if rp.from_organic)
    insight_from_dashboard = sum(1 for rp in rps if rp.from_dashboard)

    top = pick_top_record(records, qtype, art.qid, prov)
    top_scope_ok = bool(
        top is not None
        and _scope_match(top.record.get("location"), GEO_EXPECT.get(art.qid))
    )

    # ---- classification ----
    # anchor_present = a usable (scope-matched, floor-clearing) record exists.
    # evidence_sufficient (spec headline) is stricter: high-confidence AND the
    # expected count_basis. A present-but-under-confident anchor (q17) is
    # anchor_present=True, evidence_sufficient=False -> an extraction/insight
    # calibration finding rather than a search/filter one.
    anchor_present = bool(usable_rps)
    high_conf_rps = [
        rp for rp in usable_rps
        if (rp.record.get("confidence") or 0.0) >= HIGH_CONFIDENCE
        and _basis_ok(rp.record, art.qid)
    ]
    evidence_sufficient = bool(high_conf_rps)

    if not anchor_present:
        classification = "under_supported"
    elif any(rp.from_organic for rp in usable_rps):
        classification = "well_supported"
    else:
        classification = "dashboard_only"

    # single fragile source: usable anchors rest on <=1 independent domain
    single_fragile = False
    if anchor_present:
        corr_domains: set[str] = set()
        for rp in usable_rps:
            corr_domains |= rp.source_domains
        single_fragile = len(corr_domains) <= 1

    # ---- cause attribution for weak/fragile/under-confident questions ----
    cause = None
    cause_detail = ""
    weak = (
        classification in ("under_supported", "dashboard_only")
        or single_fragile
        or not evidence_sufficient
    )
    if weak:
        cause, cause_detail = attribute_cause(
            qtype, art.qid, org_search, org_filt, dash_filt, records, prov,
            terms, pool_max, anchor_present, evidence_sufficient, top,
            llm_pool_ontopic=llm_pool,
        )

    return Diagnosis(
        qid=art.qid,
        run_id=art.manifest.get("run_id", art.run_dir.name),
        qtype=qtype,
        topic=str(q.get("pathogen") or "") + (f" ({q.get('region')})" if q.get("region") else ""),
        organic_returned=len(org_search),
        dashboard_returned=len(dash_search),
        organic_survivors=len(org_filt),
        dashboard_survivors=len(dash_filt),
        injected_dash_domains=",".join(injected_domains),
        expected_resolution_domain=exp_domain,
        resolution_source_injected=resolution_injected,
        pool_overlap_mean=pool_mean,
        pool_overlap_max=pool_max,
        survivor_overlap_mean=surv_mean,
        llm_pool_ontopic=llm_pool,
        llm_survivor_ontopic=llm_surv,
        n_records=len(records),
        n_usable=n_usable,
        insight_from_organic=insight_from_organic,
        insight_from_dashboard=insight_from_dashboard,
        top_metric_value=(top.record.get("metric_value") if top else None),
        top_metric_name=(top.record.get("metric_name") if top else None),
        top_count_basis=(top.record.get("count_basis") if top else None),
        expected_basis=EXPECTED_BASIS.get(art.qid),
        top_confidence=(top.record.get("confidence") if top else None),
        top_n_sources=(top.n_independent_sources if top else 0),
        top_origin=(
            ("organic+dashboard" if top.from_organic and top.from_dashboard
             else "organic" if top.from_organic
             else "dashboard" if top.from_dashboard else "unknown")
            if top else "none"
        ),
        top_source_url=(top.source_urls[0] if top and top.source_urls else None),
        top_quote=_first_quote(top.record) if top else None,
        top_scope_ok=top_scope_ok,
        anchor_present=anchor_present,
        evidence_sufficient=evidence_sufficient,
        single_fragile_source=single_fragile,
        classification=classification,
        cause=cause,
        cause_detail=cause_detail,
    )


def _first_quote(rec: dict) -> Optional[str]:
    for src in rec.get("sources", []) or []:
        q = src.get("quote")
        if q:
            return q[:200]
    return rec.get("summary")


def _infer_qtype(art: RunArtifacts) -> str:
    # question.json doesn't carry question_type; leave blank and let the caller
    # override from the CSV. Kept for robustness.
    return ""


def attribute_cause(
    qtype, qid, org_search, org_filt, dash_filt, records, prov, terms, pool_max,
    anchor_present, evidence_sufficient, top, llm_pool_ontopic=None,
) -> tuple[str, str]:
    """Attribute the dominant cause of weak evidence.

    extraction    : a dashboard survived but yielded no usable record.
    insight       : the correct-scope anchor WAS extracted but is under-
                    confident / off-basis, so it isn't usable at the headline
                    bar (an insight-stage confidence/scope calibration gap).
    search-recall : the organic pool never contained on-topic content.
    filter-recall : on-topic organic was in the pool but didn't survive.
    robustness    : a good high-confidence anchor exists but on a single source.
    """
    def _txt_s(s):
        return f"{s.get('title','')} {s.get('snippet','')} {s.get('domain','')}"

    def _txt_f(f):
        return f"{f.get('title','')} {f.get('snippet','')} {f.get('domain','')}"

    on_topic_pool = [s for s in org_search if _rel(_txt_s(s), terms) >= ON_TOPIC_OVERLAP]
    on_topic_surv = [f for f in org_filt if _rel(_txt_f(f), terms) >= ON_TOPIC_OVERLAP]

    dash_doc_ids = {
        d for d, (origin, _u, _dm) in prov.doc_origin.items() if origin == "dashboard"
    }
    records_from_dash = sum(
        1 for rec in records
        if any(s.get("document_id") in dash_doc_ids for s in (rec.get("sources") or []))
    )
    dashboard_barren = bool(dash_filt) and records_from_dash == 0

    # 1. Dashboard survived but produced no records at all → extraction.
    if dashboard_barren:
        return (
            "extraction",
            f"{len(dash_filt)} dashboard(s) survived filtering but produced 0 "
            f"insight records (non-extractable tracker/index/JS page).",
        )

    # 2. Correct-scope anchor extracted but not usable at the headline bar →
    #    insight-stage calibration (present but under-confident / off-basis).
    if anchor_present and not evidence_sufficient and qtype in NUMERIC_TYPES:
        conf = (top.record.get("confidence") if top else None)
        basis = (top.record.get("count_basis") if top else None)
        exp = EXPECTED_BASIS.get(qid)
        return (
            "insight",
            f"scope-matched anchor present (value={top.record.get('metric_value') if top else None}, "
            f"basis={basis} vs expected {exp}, conf={conf}) but below the "
            f"high-confidence/expected-basis bar; off-scope rows outrank it.",
        )

    # 3. Organic pool had no on-topic content → search-recall.
    if pool_max < SEARCH_RECALL_POOL_OVERLAP and not on_topic_pool:
        return (
            "search-recall",
            f"organic pool max_overlap={pool_max:.2f} (<{SEARCH_RECALL_POOL_OVERLAP}); "
            f"~0 clearly-on-topic organic results returned by Tavily.",
        )

    # 4. On-topic organic (by keyword overlap) was in the pool but none
    #    survived. This is only a genuine filter-recall bug if the dropped
    #    organic is actually on-topic/authoritative. The LLM judge is the
    #    arbiter: if it says ~none of the pool is on-topic, keyword overlap
    #    was fooled by generic news and the true cause is an organic-authority
    #    gap (search-side), not the filter.
    if on_topic_pool and not on_topic_surv:
        exemplar = max(on_topic_pool, key=lambda s: _rel(_txt_s(s), terms))
        if llm_pool_ontopic is not None and llm_pool_ontopic < 0.2:
            return (
                "search-recall",
                f"{len(on_topic_pool)} organic looked on-topic by keyword overlap "
                f"(e.g. {exemplar.get('domain')}) but the LLM judge rates the pool "
                f"{llm_pool_ontopic:.0%} on-topic — generic news, not authoritative "
                f"coverage. Organic-authority gap; dashboard rightly carries it.",
            )
        judged = (
            f" LLM judge: pool {llm_pool_ontopic:.0%} on-topic."
            if llm_pool_ontopic is not None else
            " NOTE: keyword-overlap on-topic != authoritative; run --llm-judge to confirm."
        )
        return (
            "filter-recall",
            f"{len(on_topic_pool)} on-topic organic in pool (max_overlap="
            f"{_rel(_txt_s(exemplar), terms):.2f}, e.g. {exemplar.get('domain')}) "
            f"but 0 survived the filter.{judged}",
        )

    # 5. High-confidence anchor exists but single-sourced → robustness only.
    if evidence_sufficient:
        return (
            "robustness",
            "high-confidence anchor present but rests on a single source; "
            "add a corroborating source.",
        )

    return (
        "search-recall",
        f"organic pool weak (max_overlap={pool_max:.2f}); dashboards carry the question.",
    )


# --------------------------------------------------------------------------
# optional LLM on-topic judge
# --------------------------------------------------------------------------


class LLMJudge:
    """Capped gpt-4o-mini on-topic judge. One call per item, batched small."""

    def __init__(self, model: str = "gpt-4o-mini", max_items: int = 15):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        from bioscancast.llm.openai_client import OpenAILLMClient

        self._client = OpenAILLMClient()
        self._model = model
        self._max_items = max_items
        self.calls = 0

    def judge_fraction(self, question: dict, items: list[tuple[str, str]], label: str):
        items = items[: self._max_items]
        if not items:
            return None
        on_topic = 0
        qtext = question.get("text", "")
        for text, url in items:
            verdict = self._judge_one(qtext, text, url)
            self.calls += 1
            if verdict:
                on_topic += 1
        return on_topic / len(items)

    _SCHEMA = {
        "type": "object",
        "properties": {"on_topic": {"type": "boolean"}},
        "required": ["on_topic"],
        "additionalProperties": False,
    }

    def _judge_one(self, qtext: str, item_text: str, url: str) -> bool:
        system = (
            "You judge whether a search result is on-topic for a biosecurity "
            "forecasting question. on_topic=true only if the result plausibly "
            "carries evidence about the exact pathogen/place/metric the "
            "question asks about."
        )
        user = f"QUESTION: {qtext}\n\nRESULT (url={url}):\n{item_text[:600]}"
        try:
            resp = self._client.generate_json(
                system=system,
                user=user,
                schema=self._SCHEMA,
                model=self._model,
                max_tokens=20,
                temperature=0.0,
            )
            return bool((resp.content or {}).get("on_topic"))
        except Exception as exc:  # noqa: BLE001
            print(f"  ! llm judge error ({url}): {exc}", file=sys.stderr)
            return False


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


ALL_QIDS = [f"bfg_q{i}" for i in range(1, 26)]

# question_type per qid, from bfg_summer_2026_questions.csv (question.json
# does not persist it).
QTYPE: dict[str, str] = {
    "bfg_q1": "range", "bfg_q2": "range", "bfg_q3": "binary", "bfg_q4": "range",
    "bfg_q5": "categorical", "bfg_q6": "range", "bfg_q7": "binary",
    "bfg_q8": "categorical", "bfg_q9": "range", "bfg_q10": "categorical",
    "bfg_q11": "categorical", "bfg_q12": "binary", "bfg_q13": "range",
    "bfg_q14": "range", "bfg_q15": "binary", "bfg_q16": "range",
    "bfg_q17": "range", "bfg_q18": "range", "bfg_q19": "range",
    "bfg_q20": "range", "bfg_q21": "binary", "bfg_q22": "range",
    "bfg_q23": "binary", "bfg_q24": "binary", "bfg_q25": "range",
}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("qids", nargs="*", help="Question ids (default: all with runs).")
    ap.add_argument("--all", action="store_true", help="Analyze all bfg_q1..q25.")
    ap.add_argument("--runs-root", default="data/runs")
    ap.add_argument("--run-dir", default=None, help="Explicit run dir for one question.")
    ap.add_argument("--out", default=None, help="Output basename (writes .csv and .json).")
    ap.add_argument("--llm-judge", action="store_true", help="Add gpt-4o-mini on-topic judge.")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    runs_root = Path(args.runs_root)
    judge = LLMJudge() if args.llm_judge else None

    run_dirs: list[Path] = []
    if args.run_dir:
        run_dirs = [Path(args.run_dir)]
    else:
        qids = args.qids or (ALL_QIDS if args.all else None)
        if qids is None:
            # default: every qid that has at least one run
            qids = sorted(
                (d.name for d in runs_root.iterdir() if d.is_dir()),
                key=lambda n: (len(n), n),
            ) if runs_root.is_dir() else []
        for qid in qids:
            rd = latest_run_dir(runs_root, qid)
            if rd is None:
                print(f"  (no run found for {qid})", file=sys.stderr)
                continue
            run_dirs.append(rd)

    diags: list[Diagnosis] = []
    for rd in run_dirs:
        art = load_run(rd)
        # question.json lacks question_type; inject from the CSV-derived map.
        if not art.question.get("question_type"):
            art.question["question_type"] = QTYPE.get(art.qid, "")
        diag = diagnose(art, llm_judge=judge)
        diags.append(diag)
        _print_diag(diag)

    if args.out and diags:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        rows = [d.as_row() for d in diags]
        with (out.with_suffix(".csv")).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        with (out.with_suffix(".json")).open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"\nWrote {out.with_suffix('.csv')} and {out.with_suffix('.json')}")
        if judge is not None:
            print(f"LLM judge calls: {judge.calls}")

    return 0


def _print_diag(d: Diagnosis) -> None:
    flag = {
        "well_supported": "OK ", "dashboard_only": "DASH", "under_supported": "WEAK",
    }.get(d.classification, "?   ")
    frag = " [single-fragile]" if d.single_fragile_source else ""
    print(
        f"[{flag}] {d.qid:8s} {d.qtype:11s} "
        f"org {d.organic_survivors}/{d.organic_returned} "
        f"dash {d.dashboard_survivors}/{d.dashboard_returned} | "
        f"rec={d.n_records} usable={d.n_usable} "
        f"(org={d.insight_from_organic} dash={d.insight_from_dashboard}) | "
        f"top={d.top_metric_value} [{d.top_count_basis}] conf={d.top_confidence} "
        f"src={d.top_n_sources}/{d.top_origin}{frag}"
    )
    if d.cause:
        print(f"          cause={d.cause}: {d.cause_detail}")


if __name__ == "__main__":
    raise SystemExit(main())
