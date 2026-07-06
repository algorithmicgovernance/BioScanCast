# Validating the "client-side data feed" extraction tactic — BFG JS dashboards

**Date:** 2026-07-05
**Base:** PR #53 (`feat/bfg-evidence-quality-audit`)
**Audit input:** `data/investigations/bfg_evidence_audit_2026-07-05.csv`
**Scope:** probe the remaining JS-rendered BFG dashboards, land a scraper for every
dashboard with an **openly-served** JSON/CSV feed, and return a verdict on how far
the open-feed tactic generalizes. Excludes q9/APHIS (Akamai/Tableau → #50).

---

## Verdict (short version)

**The open-feed tactic does not generalize beyond the CDC source family in this
question set.** Of the four JS-rendered dashboards probed, **1 has an open feed**
(CDC measles — the positive control, which reproduces), and **0 new open feeds
were found.** The other three fail in three distinct ways:

| Failure mode | Target | What blocks the feed |
|---|---|---|
| **PDF-bound** | q6/q8 WHO A(H5N1) | The number lives only in `iris.who.int` PDFs (confirmed `application/pdf`), not a JSON/CSV endpoint. |
| **Seasonal-empty / enumeration** | q19 ECDC chikungunya | The EU/EEA dashboard's 2026 section is intentionally empty (season not started); correct anchor is **0**, stated in prose. Also an enumeration question (#54). |
| **Session-gated (Tableau)** | q20 PAHO Oropouche | The dashboard is a Tableau *trusted-ticket* embed (`phip.paho.org/trusted/<ticket>/...`, private CA cert) — same class as APHIS (#50). |

**Why it's CDC-specific:** CDC publishes its client-side numbers as *open JSON
module/vizdata feeds* (`/wcms/vizdata/...`, `/measles/*.json`, `/bird-flu/modules/*.json`)
that a plain `curl_cffi` GET returns in full. WHO, ECDC, and PAHO publish the same
class of numbers via **PDF** (WHO iris, PAHO epidemiological updates), **seasonal/empty
prose**, or **gated Tableau**. Those PDF cases are already served by the complementary
*PDF-hub* scrapers (`who_cholera`, `who_h5_hai`, `paho_dengue`) — a different tactic —
and the gated case belongs to #50. There was no new open-feed dashboard to remediate.

**Positive control reproduces.** q16 (CDC measles deaths) moved from **0 records
(under_supported)** in the pre-scraper audit to a **high-confidence, correct-scope
anchor** via the existing `cdc_measles` JSON-feed scraper (details below). The tactic
is validated end-to-end; it is simply narrow.

> **Follow-up (same day):** recommendations #1 and #2 were then investigated in place.
> **#2 (ECDC q19) was remediated** — not with a feed (there is none) but with a
> *routing* scraper (`ecdc_chikungunya`) that points the source at the EU/EEA seasonal
> report and renders a scope-correct anchor. q19 went from an off-scope non-EU "1" to
> **0 EU/EEA countries (cumulative, EU/EEA scope)**; forecast "0"-bin 0.40 → 0.914.
> **#1 (WHO q6/q8) was diagnosed, not a feed/scraper problem**: the `who_h5_hai` PDF
> pipeline works (resolves + parses the HAI PDF), but the current WHO reporting period
> has **zero new H5 human cases** (only H9N2), so there is no H5 anchor to extract — an
> insight-scope + question-window issue for #55. See "Follow-up investigation" below.

---

## Feed-probe table (one row per targeted dashboard)

| Q | source_id | Dashboard probed | Feed URL | Classification | Anchor field | Outcome |
|---|---|---|---|---|---|---|
| q16 | `cdc_measles` | cdc.gov/measles/data-research | `/wcms/vizdata/measles/measles_hosp.json` (open GET) | **OPEN** | `total_deaths` / `deaths_sentence` (+ `total_cases`) | **Positive control reproduces**: 0 records → deaths anchor `0.0`, basis=cumulative, conf=0.85; forecast 0-bin 0.70→0.976 |
| q19 | `ecdc_chikungunya` | ecdc.europa.eu/en/chikungunya-monthly → seasonal-surveillance → `chik-weekly.ecdc.europa.eu` | none (current-year data absent) | **no-feed / seasonal-empty** | count of EU/EEA countries (prose) | **Remediated (not via a feed)** — routing scraper renders the scope-correct anchor: off-scope "1" → **0 EU/EEA countries** (cumulative, EU/EEA); forecast "0"-bin 0.40→0.914 |
| q6/q8 | `who_h5n1` / `who_h5_hai` | who.int/…/avian-a-h5n1-virus | `iris.who.int/server/api/core/bitstreams/<uuid>/content` = **application/pdf** | **PDF-bound** | cumulative table (inside PDF) | Not this tactic — already the `who_h5_hai` PDF-hub target; PDF-extraction gap tracked in #55 |
| q20 | `paho_oropouche` | paho.org ARBO portal → `ais.paho.org/ha_viz/Oropouche/AME_OROV_Viz.asp` → `phip.paho.org/trusted/<ticket>/views/AME_OROV_Cases/Oropouche` | Tableau trusted-ticket (single-use, private cert) | **session-gated (Tableau)** | (Tableau VizQL) | Out of scope — same as APHIS #50; do not browser-automate |

The rest of the `under_supported` set was checked and is out-of-tactic: q11/q24/q25 are
WHO-list **enumerations** (PHEICs in effect / novel-pathogen judgment / DON-item count → #54),
q17 cholera is a **PDF-hub** case already served by `who_cholera` (low-confidence, not a
missing feed), and q23 SARS-CoV-2 is already extracted by the working OWID scraper
(a search-recall/robustness issue, not a feed gap).

---

## Positive control — end-to-end evidence (q16, CDC measles)

Live feed spot-check (`measles_hosp.json`, 2026-07-05):
`2026: total_cases=["2,170"], total_deaths=["0"], deaths_sentence=["There have been 0
confirmed deaths from measles in 2026."]` → ground-truth anchor = **0 deaths**.

Pipeline run `data/runs/bfg_q16/posctl_q16` (`--no-forecast`):
- `documents.json`: the `custom:cdc_measles` scraper fired (fetch_strategy `custom:cdc_measles`).
- `insight.json`: one record — `metric_name=deaths, metric_value=0.0, count_basis=cumulative,
  confidence=0.85, iso_country_code=US, pathogen=measles`, quote *"There have been 0
  confirmed deaths from measles in 2026."* → **matches the live feed**.
- Analyzer: `bfg_q16 range … rec=1 usable=1 | top=0.0 [cumulative] conf=0.85 src=1/dashboard`
  (was `rec=0`, `[NONE]`, under_supported in the audit).

Forecast run `data/runs/bfg_q16/posctl_q16_fc`:

| option | `bioscancast_baseline` | `bioscancast` (with anchor) |
|---|---|---|
| **0** | 0.70 | **0.976** |
| 1 | 0.10 | 0.020 |
| 2 to 3 | 0.10 | 0.003 |
| 4 to 6 | 0.10 | 0.0003 |
| 7 or more | 0.00 | 0.0002 |

The evidence-informed distribution sharpens decisively toward the anchor (0). ✔

---

## Failure taxonomy (for the three that didn't yield a feed)

1. **PDF-bound (WHO A(H5N1), q6/q8).** `who.int/.../avian-a-h5n1-virus` renders the
   cumulative counts only as titled links to `iris.who.int/server/api/core/bitstreams/<uuid>/content`,
   which return `application/pdf` (verified). The right scraper is the *PDF-hub* pattern
   (`who_h5_hai`, already present). The audit's 0 records for q6/q8 is therefore a
   **PDF-extraction** problem, not a feed problem — it belongs with #55, and the spec
   lists PDF-bound anchors as a non-goal here.

2. **Seasonal-empty + enumeration (ECDC chikungunya, q19).** The monthly *worldwide*
   overview page (which the pipeline retrieves) discusses non-EU countries (Saint Lucia,
   Peru, Suriname…), so the extractor grabs an off-scope "1". The EU/EEA number lives on
   the seasonal-surveillance page, which embeds the `chik-weekly.ecdc.europa.eu`
   `htmlwidgets`/DataTables report. That report's **2026 section is intentionally empty**:
   > "The next cycle of chikungunya virus disease updates for 2026 will begin once the
   > first cases are entered into the ECDC EpiPulse Cases platform… Until then, this
   > section will remain empty."

   So the correct current anchor is **0 EU/EEA countries** — there is no JS-injected
   number behind an open feed to grep; and the question is a country **enumeration**
   (#54). No feed exists to build against right now.

3. **Session-gated Tableau (PAHO Oropouche, q20).** The ARBO portal embeds
   `ais.paho.org/ha_viz/Oropouche/AME_OROV_Viz.asp`, a 407-byte ASP shell that iframes
   a **Tableau trusted-ticket** view:
   `phip.paho.org/trusted/<single-use-ticket>/views/AME_OROV_Cases/Oropouche`.
   The ticket is minted server-side per load and the Tableau host uses a private CA cert.
   Reaching the underlying VizQL/`bootstrapSession` data requires a live browser session —
   explicitly out of scope (same failure class as APHIS #50).

**Tally:** open 1/4 · PDF-bound 1/4 · seasonal-empty/enumeration 1/4 · session-gated 1/4.
New open feeds: **0**. New scrapers built: **0** (with the reasons above).

---

## Follow-up investigation (recommendations #1 and #2, same day)

### #1 — WHO q6/q8: investigated → **not a feed/scraper/Docling problem; it's #55 + no current H5 data**

Ran the resolver and the live question. Findings:
- **`who_h5_hai` resolves correctly.** `fetch_who_hub_latest_pdf(hub, "human-animal")` returns
  `…/influenza-at-the-human-animal-interface-summary-and-assessment--from-9-may-to-12-june-2026.pdf`
  (364 KB, `application/pdf`, 200). The hub → latest-item → PDF chain works.
- **The PDF text extracts fine without Docling.** A fresh `bfg_q6` run extracts the PDF via the
  base parser (pymupdf, ~15 k chars) and insight yields **1 record** (the audit's 0 was transient
  search/PDF drift). Docling is irrelevant here: the H5 counts are prose, not a sparse table, and
  the HAI PDF path (`_sage-2026/…`) isn't on the Docling allowlist (`…/situation-reports/`) anyway.
- **But there is no H5 anchor to extract right now.** The current assessment states:
  *"New human cases: From 9 May to 12 June 2026 … detections of influenza **A(H9N2)** in four
  humans were reported officially."* — i.e. **zero new H5 human cases** this period; `A(H5N1)`
  appears only in the references (a link to the cumulative-count table). The extractor therefore
  surfaced the off-scope **H9N2** record (conf 0.5) for an H5 question.
- **Conclusion:** the gap is (a) insight **scope-matching** (isolate H5 from the all-subtype HAI
  assessment) and (b) the question being a **forward window** (Jul–Dec 2026) with no single
  cumulative "N" — a rate/subtype forecasting problem. This is **#55 + insight-stage** work, not the
  feed tactic, not a scraper fix, and not Docling. Also note this **contradicts issue #50's claim**
  that "Q6/Q7/Q8 are unaffected."

### #2 — ECDC q19: **remediated** with a routing scraper (`ecdc_chikungunya`)

Built `bioscancast/stages/extraction/custom_scrapers/ecdc_chikungunya.py` (+ 4 no-network tests).
It ignores the worldwide-overview URL and hits the EU/EEA seasonal report
`chik-weekly.ecdc.europa.eu`, then:
- **pre-season (current live state):** detects "will remain empty" → renders *"a cumulative total of
  0 (zero) EU/EEA countries have reported locally-acquired (autochthonous) chikungunya … in 2026
  (year-to-date); the 2026 seasonal surveillance cycle has not yet begun."*
- **populated:** parses the summary sentence's `Country (n)` pairs to enumerate countries and render
  the count (verified against the 2025 archive → *"2 EU/EEA countries … France (788), Italy (384)"*).
- **leakage guard:** returns `None` under `as_of_date` (current-snapshot, no in-page history), like `cdc_measles`.

Before/after (`bfg_q19`, live evidence-only):

| | records | top anchor | basis | scope_ok | forecast "0"-bin |
|---|---|---|---|---|---|
| **before** (audit) | 1 (off-scope) | `1.0` (Saint Lucia, non-EU) | incident | ✗ | — |
| **after** (this scraper) | 1 | `0.0` (EU/EEA countries) | cumulative | ✓ (EU/EEA) | 0.40 → **0.914** |

Analyzer moved `bfg_q19` from `[NONE]`/under_supported to `[DASH]` with a scope-matched
`top=0.0 [cumulative]` anchor. Residual: the extractor's **confidence for this "absence" fact
varies run-to-run (0.5–0.9)**, occasionally dipping below the high-confidence bar — an insight
**calibration** matter (non-goal), not a scraper defect; the value/scope/basis are correct.
One honest caveat for the forecasting stage (also a non-goal): q19 resolves on **Dec 31**, and the
EU/EEA chikungunya season (Aug–Oct) typically adds a few autochthonous countries (2025 had 2), so a
Dec-31 forecast should temper a literal read of the pre-season 0 with a seasonal base rate.

---

## Prioritized recommendations (updated)

1. **q6/q8 (WHO A(H5N1)) → #55, insight-stage.** Not a feed/scraper/Docling fix. Give the insight
   stage an **H5-scoped** extraction for these questions (isolate A(H5Nx) human cases from the
   all-subtype HAI assessment; when the period has none, surface *"0 new H5 human cases in the latest
   WHO HAI reporting period"* as the scope-matched anchor), and treat q6/q8 as **rate/subtype
   forecasts** rather than anchor lookups. Correct the "Q6/Q7/Q8 unaffected" note in #50.

2. **q19 (ECDC chikungunya) — DONE this pass.** `ecdc_chikungunya` scraper + tests landed; q19 now
   has a scope-correct anchor. Future: when the 2026 season populates, the same scraper auto-enumerates
   countries (validated against the 2025 archive). Optional follow-up: an insight-calibration nudge so
   "absence" facts get steady high confidence, and a seasonal base rate in the forecast for the Dec-31
   horizon.

3. **q20 (PAHO Oropouche) — defer to #50** (gated Tableau). Adds a *second instance* of the gov-Tableau
   pattern; see the drafted #50 comment. If a non-gated path is wanted, check whether PAHO's ARBO
   **country-profile pages** / **arbo-bulletins** publish the confirmed-country list as HTML/CSV.

---

## Environment note (not a code issue)

In this worktree/session, `www.paho.org` resolved **IPv6-first** and the environment had
no working IPv6 route, so `curl_cffi` live fetches to PAHO failed (`Could not resolve host`)
even though the morning audit reached PAHO fine. Probing was done by forcing IPv4
(`CurlOpt.IPRESOLVE=1`, A-record `23.185.0.1`). The production fetcher needs no change —
this is a local routing quirk — but be aware that any live PAHO re-run **from this
environment** will fail until IPv6 routing is available.
