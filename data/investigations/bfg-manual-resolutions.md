# BFG question manual resolutions (expand scorable set 4 → 11)

Date: 2026-06-21 · Branch: `feat/pipeline-tuning`

The historical-replay benchmark could score only 4 of 11 BFG questions
(q1/q3/q7/q9); the rest were `unresolved`/`ambiguous`, capping tuning at
n=4. All target dates are now in the past, so the outcomes are knowable.
This note records the manually-researched resolutions written into
`bioscancast/stages/eval_stage/bioscancast_questions.csv`
(`question_status` → `resolved`, `resolved_option` set). Each resolution
cites an authoritative source; caveats are flagged for the two soft cases.

| id | question (metric) | resolution | confidence | basis |
|---|---|---|---|---|
| q2 | H5N1 US human cases by 2026-01-01 | **70-100** | high | flat at 70 since Apr 2024; no new cases through Dec 2025 |
| q4 | # US states w/ H5N1 human cases by 2025-02-28 | **15-20** | medium ⚠ | 15 states; bucket-boundary convention (see below) |
| q5 | ≥5 US H5N1 deaths by 2025-05-01? | **NO** | high | only 1 US death (Louisiana, Jan 2025) |
| q6 | new Marburg outbreak ≥5 cases before 2025-05-01? | **YES** | medium ⚠ | Tanzania 10 cases/10 deaths (2 confirmed + 8 probable) |
| q8 | cumulative global mpox by 2025-05-01 | **131,001+** | medium ⚠ | OWID ~140,102 for May 1 2025 |
| q10 | Sudan virus confirmed cases by 2025-06-30 | **9-15** | high | 12 confirmed (14 incl. probable); outbreak ended 26 Apr 2025 |
| q11 | most likely cause of the Congo outbreak | **Vector-borne** | high | 2025 Équateur (DRC) outbreak attributed to malaria |

## Per-question detail

- **q2 — 70-100.** US H5N1 human cases held at 70 (cumulative since Apr
  2024) through 2025; CDC FluView weeks 50–51 (Dec 2025) reported no new H5
  human infections. 70 → 70-100 bucket. Same plateau value as q1.
  Source: CDC FluView wk 51, CDC bird-flu situation summary.

- **q4 — 15-20 (caveat).** The dashboard showed **15 states** on 2025-02-28.
  The option set overlaps at 15 (`10-15` *and* `15-20` both include it) — the
  original "ambiguous" reason. Resolved by the half-open convention
  (10-14, 15-19, 20-24, 25+), placing 15 in the **15-20** option. A
  defensible decision for a flawed bucket boundary, not an unambiguous fact.

- **q5 — NO.** Exactly one US H5N1 human death (Louisiana, reported Jan 6
  2025); it remained the only one. < 5 by May 1 2025. Source: CDC press
  release m0106; Louisiana Dept of Health.

- **q6 — YES (caveat).** Tanzania declared a Marburg outbreak (Kagera) on
  20 Jan 2025; cumulative **10 cases / 10 deaths** (Jan–Mar 2025), outbreak
  over 13 Mar 2025. Exceeds ≥5 before May 1. **Caveat:** WHO counted
  2 confirmed + 8 probable — "≥5" holds only if probable cases count toward
  the threshold. Source: WHO DON 2025-DON554; Africa CDC.

- **q8 — 131,001+ (caveat).** Our World in Data shows ~**140,102**
  cumulative confirmed mpox cases globally for 2025-05-01, far above the
  131,001+ floor. **Data-quality caveat:** OWID revises retrospectively.
  q7's contemporaneous resolution note cites a "final case count of
  126,441" (→ 126,001-128,500), but OWID now shows ~130,142 for the same
  2025-02-28 date — which would fall in q7's *next* bucket (128,501-131,000).
  q8 clears 131,001+ under either the contemporaneous or revised series, so
  the resolution is robust; but q7/q8 ground truth is tracker- and
  vintage-dependent in a way the CDC/WHO-dashboard questions (q1/q3/q9) are
  not. Worth keeping in mind when scoring mpox questions.

- **q10 — 9-15.** 2025 Uganda Sudan virus disease outbreak: 12 confirmed
  cases (14 including 2 probable), 4 deaths; outbreak declared over 26 Apr
  2025. Within 9-15. Source: WHO DON 2025-DON566; NEJM "Sudan Virus Disease
  in Uganda, 2025".

- **q11 — Vector-borne.** The "current Congo outbreak" at question time was
  the 2025 Équateur province (DRC) event, attributed to **malaria** — a
  vector-borne (Anopheles-mosquito-transmitted) disease — so it maps to the
  `Vector-borne` option. Source: Wikipedia, "2025 Équateur province malaria
  outbreak" (per project lead).

## Sources

- CDC FluView wk 51 (2025): https://www.cdc.gov/fluview/surveillance/2025-week-51.html
- CDC first US H5 death: https://www.cdc.gov/media/releases/2025/m0106-h5-birdflu-death.html
- WHO Marburg DON (Tanzania): https://www.who.int/emergencies/disease-outbreak-news/item/2025-DON554
- Our World in Data — mpox: https://ourworldindata.org/mpox
- WHO Sudan virus DON: https://www.who.int/emergencies/disease-outbreak-news/item/2025-DON566
- NEJM Sudan virus Uganda 2025: https://www.nejm.org/doi/full/10.1056/NEJMc2508159
- 2025 Équateur province malaria outbreak: https://en.wikipedia.org/wiki/2025_%C3%89quateur_province_malaria_outbreak

## Caveats summary (for downstream tuning)

- **q4, q6** are soft resolutions (bucket-boundary convention; probable-case
  counting). **q8** depends on the mpox tracker vintage. Consider a
  sensitivity check that excludes q4/q6/q8 when a tuning result is marginal.
- These resolutions give ground truth only. Scoring still requires forecasts
  at each cutoff — i.e. regenerating trajectories for the newly-resolved
  questions (target dates: q2 2026-01-01, q5/q6/q8 2025-05-01, q10
  2025-06-30, q11 2025-03-31; `_parse_target_date` may need help for some).
