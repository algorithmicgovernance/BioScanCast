# BFG summer-2026 — evidence-coverage audit & remediation

**Date:** 2026-07-05 · **Branch:** `claude/brave-diffie-2895b6` (based on PR #52
`feat/bfg-summer-2026-readiness`) · **Mode:** live, evidence-only (`--no-forecast`)

Instrument: [`scripts/analyze_evidence_coverage.py`](../../scripts/analyze_evidence_coverage.py).
Diagnosis table: [`bfg_evidence_audit_2026-07-05.csv`](bfg_evidence_audit_2026-07-05.csv)
(+ `.json`). All 25 questions run once each in one ~19-min window; total pipeline
cost ≈ **$0.22**; the LLM on-topic judge added **403** gpt-4o-mini calls (≈ $0.03).

This is Phase 1 (diagnose) of `forecast-evidence-quality-spec.md`. The injected-vs-
organic split is used as the diagnostic instrument, not the deliverable.

---

## Headline

| Metric | Result |
|---|---|
| Resolution-source dashboard injected | **25/25** — routing is not dropping authoritative sources |
| Evidence-sufficient (high-confidence, correct-scope, correct-basis anchor) | **14/25** |
| Single fragile source (anchor rests on one source) | **7/25** |
| **0 usable records at all** (forecast runs on pure baseline) | **7/25** |

Classification: **well_supported 7 · dashboard_only 8 · under_supported 10.**

Two things the audit *rules out* as the dominant problem:

1. **Routing (#41) is fine in live mode.** Every question injected its resolution
   domain (who.int / cdc.gov / paho.org / ecdc.europa.eu / aphis.usda.gov /
   polioeradication.org). The single-bucket `route_sources` did not misroute any
   of the 25. The additive-injection change #41 proposes is not needed to fix a
   coverage gap here (still worth doing for robustness, but it's not load-bearing).
2. **The filter (#13/#44) is mostly fine.** Where the deterministic keyword-overlap
   heuristic flagged "on-topic organic dropped by the filter," the gpt-4o-mini judge
   rated those pools **0–13% on-topic** on 6 of 7 questions — i.e. generic news
   (nypost, abcnews, statnews, motor1), correctly rejected. Only **q15** shows a
   genuine authoritative-organic drop (judge: 53% on-topic).

The actual gaps are downstream: **extraction** and **insight-stage calibration**.

---

## Where forecasts are actually under-supported

### A. Extraction — dashboard survives but yields 0 records (7 questions) — **highest impact**

These forecasts have **no current-value evidence at all**; they run on the
retrieval-free baseline. The authoritative dashboard is injected and survives
filtering, but the insight stage extracts nothing usable from it.

| Q | Question | Injected source that went barren | Root cause |
|---|---|---|---|
| **q9** | US states w/ H5N1 dairy detections | APHIS livestock (Tableau) | **Issue #50** — Tableau behind Akamai; ~492-char stub. Curated-snapshot scraper already drafted in #50. |
| **q6** | H5 human cases (global, window) | WHO avian-influenza landing + monthly HAI risk-assessment | Landing/index pages; the human-case counts live in the HAI report/PDF, not the page body. |
| **q8** | Predominant H5 subtype | same WHO H5 pages | Same as q6; subtype breakdown is in the HAI report. |
| **q16** | US measles **deaths** 2026 | CDC measles data page | Same page yields **case** counts (q14 works) but not the **deaths** figure — extraction-scope gap. |
| **q11** | # PHEICs in effect | general_sources (15 WHO/CDC list/index pages) | "Count items on a list" — pipeline doesn't enumerate list entries. |
| **q24** | Novel-pathogen DON in window | general_sources (15) | Same enumeration gap over the WHO DON list. |
| **q25** | # new WHO DON items in window | general_sources (15) | Same — needs to count DON items in a date window. |

Two sub-families: **(a) fixable with a targeted scraper** (q9, q6, q8, and likely
q16) — resolve the landing page to the fact-bearing report/PDF, exactly as the
cholera/dengue/polio scrapers already do; **(b) list-enumeration questions**
(q11, q24, q25) — the pipeline has no "count entries on this authoritative list"
capability. That's a larger capability gap, better recommended than hacked in-pass.

### B. Insight-stage calibration — anchor extracted but not usable (q17) — **Issue #26**

q17 (global cholera 2026): the correct anchor **114,829 cumulative (Global)** *was*
extracted from the WHO cholera epi-update PDF — but at **confidence 0.5**, while
off-scope regional rows (African Region 61,888; Angola) were extracted at **0.85**.
The high-confidence rows outrank the row the question actually needs.

This is the exact "confidence miscalibration" failure mode **Issue #26** says to
watch for after the first benchmark. This audit is that benchmark; q17 is the
evidence #26's definition-of-done item 1 requires. Note the anchor *is* in the
evidence digest fed to the forecaster (records aren't confidence-gated before the
digest), so **Phase 3 must check whether the forecast still anchors on 114,829
despite the low confidence** before concluding a code change is required.

### C. Robustness — good anchor, single source (7 questions)

q3, q7, q12, q13, q14, q15, q18, q22 carry a **high-confidence, correct-scope
anchor** (e.g. mpox 180,545 cumulative; dengue-Americas 251,057; measles 2,170;
polio WPV1 7) — but from **one** source, with **no organic corroboration**. The
judge shows *why*: for off-peak counts the organic pool is generic news, so the
authoritative number exists only on the single injected dashboard. These forecasts
are supported but fragile to that one source failing. Corroboration must come from
a **second authoritative source** (another dashboard), not organic news.

### D. Thin / low-confidence anchor (q19, q20, q23)

q19 (chikungunya EU/EEA), q20 (Oropouche Americas), q23 (SARS-CoV-2 VOI/VOC): only
a weak dashboard record (conf 0.5–0.7, often a placeholder "1" or an off-scope
cumulative), below the usable bar. q23's "anchor" (775M cumulative COVID cases) is
irrelevant to a variant-designation binary — correctly classified under_supported.

---

## Well-supported (7) — do not regress

q1, q2, q4, q5 (Bundibugyo Ebola — peak pathogen, rich organic + dashboard),
q10 (PHEIC — organic PHEIC-mention records; *soft* — see caveat), q21 (filovirus
binary), q3 (Ebola PHEIC binary — flagged fragile). Peak pathogens get genuine
organic corroboration; off-peak ones do not.

**Caveat on categorical "count events" questions (q10, q11, q24, q25):** the
numeric-anchor concept fits case/range questions well but not "how many PHEICs /
DON items" — those need list enumeration. q10 scores well_supported only because it
picked up on-topic PHEIC-mention text, not because it counted PHEICs. Treat q10's
verdict as soft.

---

## Diagnosis → open-issue map

| Finding | Questions | Issue | How it feeds the issue |
|---|---|---|---|
| Dashboard barren at extraction | q9 | **#50** | Confirms the gap live (0 records); fix = land the drafted curated-snapshot scraper. |
| Anchor extracted but under-confident | q17 | **#26** | The "first benchmark" evidence that confidence miscalibration is real → justifies the refinement pass. |
| Routing never misrouted (25/25) | all | **#41** | Negative result: single-bucket routing isn't the coverage bottleneck in live mode. |
| Generic-news organic dropped correctly | q12,q13,q18,q20,q22,q23 | #13/#44 | Filter is working; judge confirms. No change warranted. |
| WHO H5 / list pages don't yield counts | q6,q8,q11,q24,q25 | (new) | Scraper for HAI report (q6/q8); list-enumeration capability (q11/q24/q25) is net-new. |

---

## Phase-2 remediation plan (prioritized, pending go-ahead)

Ordered by (impact × safety). Safe levers = **scrapers** (isolated, additive,
per-source — the cholera/dengue/polio pattern). Risky levers = search-weight /
filter-threshold / insight-prompt changes (broad blast radius; the spec and
`findings-issues-3-4-13.md` both warn on these).

1. **q9 → land the #50 APHIS curated-snapshot scraper.** Isolated, closes/advances
   #50. Turns 0 records → 1 clean "N states as of DATE" anchor. *Decision needed:*
   snapshot value source (verify current count via web) and refresh model.
2. **q6/q8 → add a WHO HAI scraper** (`who_h5_hai`) that resolves the
   avian-influenza monthly-risk-assessment landing to its latest report/PDF with
   human H5 case counts + subtype, mirroring `_who_hub_common`. Fixes 2 questions.
3. **q16 → measles-deaths extraction.** Check whether the CDC measles page carries
   the death figure and it's an insight-scope miss vs. genuinely absent; add a
   corroborating CDC source if needed.
4. **q17 → measure first (Phase 3), then #26.** Confirm whether the forecast
   anchors on 114,829 despite conf 0.5. Only if it doesn't, do a scoped
   confidence/scope fix under #26 (avoid a broad insight-prompt change mid-audit).
5. **Robustness (q7,q12,q13,q14,q18,q22) → add a second authoritative source** per
   pathogen family where one exists (e.g. ECDC/CDC alongside WHO/PAHO). Additive,
   low risk.
6. **q11/q24/q25 (list enumeration) → recommend, don't hack.** Net-new capability;
   file or extend an issue rather than force it in this pass.

**Explicitly not doing:** search-stage weight changes, filter-threshold changes,
or forecasting-model tuning (out of scope / broad blast radius).

## Phase-3 (after remediation)

Re-run each fixed question **with** the forecast; confirm the distribution anchors
on the correct current value and diverges sensibly from `bioscancast_baseline`.
Regression-guard: re-run a sample of the 7 well-supported questions + the historic
benchmark (`scripts/run_historical_trajectory.py`) to confirm no degradation.

---

# Phase 2 — remediation (outcomes)

## Environment finding (important): Docling was missing; a real refiner bug

The Phase-1 sweep ran in a worktree Python that was **missing `docling`** — a
*required* dep (`requirements.txt`: `docling[chunking]>=2.90`). The repo keeps a
separate `.venv-docling`; I completed it (added `tavily`, `pycountry`,
`matplotlib`, `trafilatura`, …) so the pipeline runs with Docling as intended.

While validating, I found a genuine defect in the extraction pipeline:

> **Docling table-refiner never fires on custom-scraper-resolved PDFs.**
> `pipeline.py` passed `source_url=filtered_doc.url` (the *hub* URL, e.g.
> `who.int/.../cholera-upsurge`) to `refiner.refine()`, but the Docling allowlist
> entries (`.../situation-reports/`, `.../_sage-`) target the *resolved* PDF URL
> (`cdn.who.int/...`). So the allowlist could never match for exactly the hub
> scrapers (cholera, mpox, HAI, …) it exists to serve.

**Fix (landed):** [`pipeline.py`](../../bioscancast/stages/extraction/pipeline.py)
now matches on `fetch_result.final_url or filtered_doc.url`. Verified: Docling now
fires on the cholera situation-report and PAHO mpox sitrep PDFs.

## Code changes landed

1. **`custom_scrapers/who_h5_hai.py`** (new) — resolves the WHO "human-animal
   interface" monthly-risk-assessment hub to its latest assessment PDF, mirroring
   `who_cholera`. q6 went 0 → records. **Caveat:** the HAI summary reports the
   reporting-period's *events* (this issue: an H9N2 China cluster) and *links to*
   the cumulative A(H5N1) human-case total on a **separate** page rather than
   stating it — so the scraper is a correct precondition but does not, by itself,
   deliver q6/q8's cumulative anchor.
2. **`pipeline.py` refiner-URL fix** (above).

## What Docling did (and didn't) change

Re-running the PDF questions with Docling + the URL fix **did not change any
classification or anchor**: q17 stays `114,829 cumulative @ conf 0.5`; q13/q18/q22
keep their high-confidence dashboard anchors. Conclusion: the Phase-1 verdicts are
**robust to the Docling gap** — the cholera/mpox/dengue/polio PDFs carry their key
numbers in prose or simple-enough tables that the base parser already handles.
Docling matters for genuinely hard tables, but it does **not** rescue the gaps below.

## Gaps that are NOT scraper/Docling-fixable (revised)

- **q16 (measles deaths):** the 2026 death count is in an **HTML** table on the CDC
  page; the Docling refiner is **PDF-only**, so it can't help. (Cases are in prose →
  q14 works.) Needs HTML-table extraction or a CDC data endpoint.
- **q9 (APHIS states):** Tableau-behind-Akamai — **deferred** per decision; needs a
  browser package (**issue #50**).
- **q11 / q24 / q25 (list enumeration):** "count PHEICs / DON items" — the pipeline
  has no capability to enumerate entries on an authoritative list. Net-new.
- **q6 / q8 (H5):** cumulative total lives on a separate page **and** the Jul–Dec
  window had barely opened at run time, so there is little in-window data to anchor.

# Phase 3 — forecast-quality delta (sample)

Ran 4 questions **with** the forecast (evidence `bioscancast` vs retrieval-free
`bioscancast_baseline`). The spec's north star — does evidence *change the forecast*?

| Q | Evidence forecast vs baseline | Read |
|---|---|---|
| **q1** (well-supported, anchor 1,460) | evidence: **0.64** on "1,000–2,499" + 0.36 on "2,500+"; baseline diffuse on low bins | Retrieval **decisively** anchors the forecast on the true current value. ✔ |
| **q17** (anchor 114,829 @ conf 0.5) | evidence shifts mass up to "300k–449k" 0.35 / "450k–599k" 0.27; baseline 0.25/0.35 on the low bins | Low-confidence anchor **is used** — the record reaches the digest regardless of confidence. **q17 needs no insight code change for the forecast to move.** ✔ |
| **q18** (dengue, anchor ~63k–251k) | evidence concentrates on "1–1.9M" 0.43 / "2–3.9M" 0.31 | Anchors sensibly toward the Americas full-year range. ✔ |
| **q6** (under-supported, ~0 records) | evidence: **0.96** on "0–4"; baseline diffuse | Sparse evidence → **over-confident** forecast. The real risk of under-support. ✘ |

**Takeaway:** the evidence→forecast path works well where evidence is sufficient
(q1/q17/q18 all diverge sensibly from baseline toward the anchor). The danger is not
that good anchors are ignored — it's that **thin evidence yields over-confident
forecasts** (q6). So the priority is the genuine extraction gaps, not the anchors
that already work.

# Revised recommendations (prioritized)

1. **HTML-table extraction** (q16 measles deaths; also helps any CDC/ECDC
   count-in-a-table). Highest ROI now that PDF tables are handled. — *new issue*
2. **List-enumeration capability** (q11/q24/q25) — count entries on a WHO DON/PHEIC
   list. — *new issue*
3. **q9 APHIS** — proceed on **#50** (curated snapshot or headless) when a browser
   dep is acceptable. — *deferred by decision*
4. **q17 / insight confidence calibration (#26)** — *optional*: the forecast already
   uses the anchor, so this is a nice-to-have (cleaner digests), not a blocker.
5. **Over-confidence guard for thin evidence** (q6-type) — a forecasting-stage
   concern (out of this spec's scope) worth a separate note.
6. **Ship the two landed fixes** (HAI scraper, refiner-URL bug) — the URL fix is a
   real correctness win for all hub-scraper PDFs regardless of this round.
