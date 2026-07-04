# Issue #34 — postfix trajectory re-score, reproduced on `main`

Date: 2026-07-04 · Branch: `fix/eval-cleanup-and-issue34-rescore` (off `main` @ `c8962a6`)

Companion to [`issue-34-owid-dashboard-fetch.md`](issue-34-owid-dashboard-fetch.md),
which documents the OWID dashboard-fetch fix and its original re-score on
`feat/pipeline-tuning`. That analysis was ported to `main` as prose, but the
trajectory-CSV **artifacts** were left behind (disjoint history). This note
records a fresh, live re-run of the q7/q8 trajectories on the **ported** code
(the `source_id`-dispatched custom-scraper framework) and commits the missing
CSVs.

## What was run

Full historical-replay pipeline (search → filter → extract → insight →
forecast) at each cutoff, via `scripts/run_historical_trajectory.py`, live
OpenAI + Tavily. q7: 4 cutoffs (2025-02-24 → 02-27); q8: 5 cutoffs
(2025-02-24 → 04-17). ~$0.17 total across 9 cutoffs.

Artifacts (committed):
`data/investigations/q7_trajectory_forecasts.csv`,
`data/investigations/q8_trajectory_forecasts.csv`.

## Criterion 1 — records now produced ✅ (both)

The OWID custom scraper fired at every cutoff: each run has a `dashboard_csv`
doc and real mpox cumulative-case insight records (q7: 3 records at 02-24;
was **0** pre-fix). Forecasts move decisively off the uniform/prior.

## Criterion 2 — Brier vs the no-retrieval baseline: q8 win, q7 confounded

Scored with `bioscancast.stages.evaluation.scoring.multiclass_brier_score`
against the contemporaneous resolved bucket. `bioscancast_baseline` is the
per-run retrieval-free forecast — a like-for-like stand-in for the pre-fix
zero-record behaviour (falls back to a ~uniform prior, Brier 0.750).

| question | truth bucket | informed Brier (by cutoff) | baseline Brier | verdict |
|---|---|---|---|---|
| **q8** | `131,001+` | 0.131 → 0.000 → 0.000 → 0.000 → 0.000 | 0.750 | **clean win** |
| **q7** | `126,001-128,500` | 1.853 / 1.620 / 1.618 / 1.945 | 0.750 | **regression (confounded)** |

**q8** is a clean win: the resolution floor `131,001+` is robust across OWID
data vintages, so feeding the real cumulative series makes the model
confidently correct at all 5 cutoffs.

**q7 regresses, but it is an instrument artifact, not a pipeline defect.**
OWID has **retrospectively revised** the Feb-2025 mpox figure upward
(contemporaneous truth 126,441 → bucket `126,001-128,500`; OWID's current
series reads ~129,602 climbing → `128,501-131,000`). The scraper feeds the
model OWID's *current* series, so it confidently lands in the revised bucket
and misses the contemporaneous truth. Pre-fix it had no data and fell to a
prior that happened to score better. This is the exact caveat in the OWID
scraper's docstring and in `issue-34-owid-dashboard-fetch.md`.

These numbers reproduce the original `feat/pipeline-tuning` re-score
(q8 → 0.000; q7 → ~1.995 worst cutoff) on the ported code — i.e. the port
preserved behaviour.

## Status of #34

- Criterion 1 (records produced, forecasts off-prior): **met** for q7 and q8.
- Criterion 2 (Brier improves vs zero-record): **met where ground truth is
  vintage-stable** (q8); **confounded** by retrospective revision where it is
  not (q7).
- Criterion 3 (OWID COVID regression) is a code/test concern covered by
  `bioscancast/tests/test_owid_custom_scrapers.py`, not this run.

Because q7's criterion 2 is confounded by a documented data-vintage limitation
rather than satisfied outright, **whether to close #34 is a judgement call** —
left to the maintainer. A faithful-replay follow-up (fetch the OWID CSV *as of*
the cutoff via git-pinned history) is noted in the companion doc as the way to
un-confound q7's replay score; it improves benchmark fidelity only (live
forecasting already wants the current series).
