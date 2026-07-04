# Issue #34 — OWID dashboard data fetch (mpox + COVID)

Date: 2026-06-22 · Branch: `feat/pipeline-tuning` · Commit: `0fac291`

> **Update (issue #38, port onto the refactor).** The logic below was
> originally written as a `DashboardExtractor` under
> `bioscancast/extraction/dashboards/`. On `main` it now lives in the
> `source_id`-dispatched custom-scraper framework:
> `bioscancast/stages/extraction/custom_scrapers/_owid_common.py` (shared core)
> with thin `ourworldindata_mpox.py` / `ourworldindata_covid.py` modules. The
> data source (GitHub-raw cumulative CSV), the cumulative-metric fix, the
> as-of-date cutoff, and the revision caveat below are unchanged; the new
> version additionally layers PR #37's question/region location targeting on
> top. Tests: `bioscancast/tests/test_owid_custom_scrapers.py`.

## Problem

q7/q8 (mpox) extracted **zero insight records** → forecasts fell back to model
priors (the two worst questions in the postfix benchmark, Brier 0.818/0.733).
Root cause was extraction, not search/filter: OWID renders its charts
client-side (static scrape = prose + citations, no numbers) and the WHO
emergencies URL is an index landing page (~260 chars of links). Both reported
`status="success"` (the silent-truncation class the Track E guard now flags).

## Solution — extensible dashboard-fetcher framework

`bioscancast/extraction/dashboards/`:
- `base.DashboardExtractor` — strategy ABC (`matches(url)`, `extract(url, *,
  as_of_date, config) -> ParsedContent | None`).
- `__init__.DASHBOARD_EXTRACTORS` + `find_dashboard_extractor(url)` — registry;
  add a new tricky dashboard by appending an extractor.
- `owid.OWIDDashboardExtractor` — fetches OWID's published CSV time series
  (canonical GitHub raw — the grapher `.csv` slugs in #34 now 404), selects the
  entity (default `World`), applies the as-of cutoff as a row filter, emits a
  prose summary + recent-trend table. Datasets: mpox + COVID-19, same schema.

Wired into `ExtractionPipeline.extract_one` *before* the generic fetch path; a
match that yields nothing falls back to generic fetch (never drops the doc).
Shared `_finalize_document` builds the Document for both paths. New
`fetch_strategy="dashboard:owid"`, `document_type="dashboard_csv"`.

Because the CSV carries full history, replay needs **no Wayback snapshot** for
these sources — sidesteps the snapshot-selection fragility from Track E.
Caveat: replay uses *today's CSV's* estimate of the cutoff-date value (OWID
revises history), not the value as published then — far smaller error than the
zero-record fallback, and exactly right for live mode.

## Validation

Live OWID data (the extractor, real fetch):
- mpox World cumulative **129,602 @2025-02-24**, **129,603 @2025-02-27** (the
  previously zero-record cutoffs)
- COVID **732,363,539 @2023-01-01** (regression criterion — COVID path works)

End-to-end on the **saved** q7@2025-02-24 filtered docs (extract + insight, no
Tavily): OWID doc now `dashboard_csv`, prose+table chunks; insight yields
**1 record** `confirmed_cases=129,602` (date 2025-02-24, conf 0.9, valid
quote) vs **0 records** pre-fix. ✅ Acceptance criterion 1.

Tests: `test_extraction_dashboards.py` (matching incl. Wayback wrapper, cutoff
filtering, entity isolation, live-vs-replay, fetch-failure fallback, pipeline
routing). Full suite: 284 passed / 3 skipped.

## Full trajectory re-score (acceptance criterion 2) — MIXED, and instructive

Regenerated q7 (4 cutoffs) + q8 (5 cutoffs) full pipelines with #34 enabled
(`data/investigations/issue34/`, 2026-06-22). #34 fired in every run: each
cutoff now has 1 `dashboard_csv` doc and real insight records
(q7 `confirmed_cases=129,602 @02-24`, `129,603 @02-27`; was **0 records**).

| question | pre-#34 Brier | post-#34 Brier | verdict |
|---|---|---|---|
| q8 (resolve 131,001+) | 0.733 | **0.000** | clean win |
| q7 (resolve 126,001-128,500) | 0.818 | **~1.995** | regression — but a data-vintage artifact, not a pipeline defect |

**q8 is a clean win:** resolution `131,001+` is robust across OWID vintages
(contemporaneous and current both clear the floor), so feeding the model the
real cumulative figure makes it confidently correct at all 5 cutoffs.

**q7's "regression" is the OWID revision caveat made real.** q7's
contemporaneous ground truth is **126,441** (→ bucket `126,001-128,500`), but
OWID has since **retrospectively revised** the 2025-02-28 figure up to
**~130,142** (→ `128,501-131,000`). #34 feeds the model OWID's *current* series
(~129,602 climbing), so it confidently lands in the revised bucket and misses
the contemporaneous truth. Pre-#34 it had no data and fell to a prior that
happened to score better. This is exactly the caveat in `owid.py`'s docstring,
and `bfg-manual-resolutions.md` pre-flagged it ("q7/q8 ground truth is
tracker- and vintage-dependent").

**Interpretation.** The plan's objective is *live* forecast quality, with the
replay benchmark as the measurement instrument. For live forecasting #34 is
unambiguously correct — the model now sees the real cumulative figure instead
of guessing. q7's replay score is confounded by retrospective revision, a
limitation of the *instrument* on revised-data sources, not of #34. Net: 1
clean win (q8), 1 instrument-confounded case (q7); criterion 2 is met where the
ground truth is vintage-stable.

## Remaining
- Large COVID CSV (~70 MB) is downloaded per run; consider a cached/streamed
  fetch if COVID questions become hot (follow-up, not blocking).
- #34's proposed grapher-`.csv` URL is dead; the issue text would want updating
  to the GitHub-raw source (logging is the user's call).
- **Replay fidelity on revised data (q7):** to make historical replay faithful
  for OWID, the extractor could fetch the CSV *as of the cutoff* via OWID's git
  history (commit-pinned raw URL) rather than today's `main`. Improves benchmark
  fidelity only — live forecasting already wants the current series. Substantial;
  not built.
