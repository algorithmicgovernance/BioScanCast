# Systematic review of issues #3, #4, #13 — findings

Date: 2026-05-28 · Branch: `feat/end-to-end-orchestrator`

## Method

- Drafted 10 live forecasting questions (`data/investigations/live_questions.csv`,
  `L1`–`L10`) mirroring the `bioscancast_questions.csv` schema, spanning
  range/binary/categorical across H5N1, the DRC+Uganda Bundibugyo Ebola
  outbreak, the Andes-virus cruise hantavirus cluster, mpox, and Marburg.
  Topics chosen to span dashboard-covered vs not.
- **Captured** the live search pool for each (`scripts/capture_search_pools.py`
  → `data/investigations/live_pools/<id>.json`). `total_cap=60`; all pools were
  29–49 results, so **none were truncated** — the captured pools are complete.
- **Hand-labeled** every result keep/drop (`live_pools/labels.json`).
- **Swept** filter thresholds (#13) and search-stage weights (#4) offline and
  deterministically over the cached pools (`scripts/sweep_filter_params.py`),
  scored against the labels. Zero API cost.
- **Ran the full pipeline live** end-to-end on 5 questions + 1 historical replay
  (`data/investigations/runs/`).

**Cost:** total ≈ **$0.025** across all 6 live/replay runs (well under the $5
cap). The offline sweep and dashboard checks were free.

## Headline finding (ties all three issues together)

`search_stage_score = 0.5·domain + 0.3·freshness + 0.2·(1/rank)` contains **no
relevance/topic-match term**. Combined with Tavily's recency bias, this means
that whenever the question's pathogen is *not* the dominant news story of the
moment, the organic pool is flooded with off-topic high-authority content that
ranks well on domain score alone.

In our snapshot (late May 2026) Ebola was *the* story, so:

| Question | On-topic organic results | Dominant noise in pool |
|----------|--------------------------|------------------------|
| L1/L2 H5N1 | ~0 (only the 2 dashboards) | Ebola/hantavirus/measles news, a nature.com TB paper at rank 0, Law360, Ukraine casualties, a King Charles death hoax |
| L7/L8 mpox | ~1 organic (fox6now WI cluster) + dashboards | Ebola + NHL/FIFA/NCAA/Belmont/French-Open picks, nature.com (nuclear-fuel cladding, semaglutide), biospace PR |
| L9 Marburg | ~0 (only the 2 dashboards) | entirely the Ebola/hantavirus roundup |
| L3/L4 Ebola | rich, genuine coverage | minimal |

No amount of filter-threshold tuning (#13) can recover relevance that the
search stage never surfaced. The dashboards (#3) are the only thing keeping the
off-peak questions afloat — but, per below, they frequently don't extract.

## Issue #4 — search-stage weights

**The weights barely matter; the missing dimension does.** On organic-ranking
quality (precision@P / MAP, micro-averaged, scored vs labels):

| Variant (domain/fresh/rank[/rel]) | prec@P | MAP |
|---|---|---|
| current 0.5/0.3/0.2 | 0.694 | 0.506 |
| no-freshness 0.6/0/0.4 | 0.694 | 0.528 |
| rank-heavy 0.4/0.1/0.5 | 0.694 | 0.580 |
| domain-heavy 0.7/0.1/0.2 | 0.694 | 0.506 |
| **+relevance 0.25/0.15/0.1/0.5** | 0.725 | 0.600 |
| **relevance-only 0/0/0/1** | **0.755** | **0.656** |

- Reordering domain/freshness/rank leaves top-P precision **unchanged** (0.694)
  — freshness is ~uniform (≈0.98–1.0) in live mode so its 0.3 weight is nearly
  dead signal, and domain score is too coarse to separate on-topic from
  off-topic within a tier.
- Adding a keyword-overlap **relevance** term is the only thing that moves the
  needle: MAP 0.506 → 0.656.

**Recommendation (#4):** add a relevance/keyword-overlap term to
`_compute_search_stage_score` (the same `keyword_overlap_score` the filter
already uses), e.g. `0.35·relevance + 0.30·domain + 0.10·freshness +
0.10·rank` (drop freshness's weight; it is near-useless live). Keep `rank` low.
This is the highest-leverage change of the three issues and prevents off-topic
authority from consuming the `total_cap` slots.

## Issue #13 — heuristic filter survival

**The keep threshold sits on a wide plateau; it is not the lever.** Organic-only,
micro-averaged, no-LLM (fail-closed) sweep:

| keep_threshold | precision | recall | organic survivors | official recall |
|---|---|---|---|---|
| 0.50 | 0.43 | 0.20 | 47 | 1.0 |
| 0.575 | 0.23 | 0.03 | 13 | 1.0 |
| **0.60 – 0.775** | **0.43–0.50** | **0.03** | **6 (flat)** | **1.0** |
| 0.80 | 0.60 | 0.03 | 5 | 1.0 |

- Thresholds **0.60 through 0.775 are identical** (6 survivors). The
  `0.72 → 0.65` change (commit `f63684b`) moved within this flat region — it
  did not, by itself, change organic survival. (It still matters as the cutoff
  below which the LLM-rescue path is fed; see below.)
- **Official-source recall is 1.0 at every threshold** — the ECDC/official
  organic sources we labeled keep always survive. The fix for the original
  "dropping CDC/WHO" symptom is holding.
- The two real recall sinks are structural, not the threshold:
  1. **Fail-closed LLM path.** 84 organic candidates land in the rerank→LLM
     band; with `llm_client=None` *all 84 are dropped*. This is by far the
     dominant dev-mode recall sink.
  2. **Unknown-tier credibility floor.** Much legitimate outbreak news
     (Forbes, LA Times, NBC) resolves to `source_tier="unknown"`
     (`domain_score=0.20`, `credibility≈0.35`). `priority_score` blends in
     `0.25·credibility`, which caps these below the 0.45 borderline regardless
     of keyword overlap, so they are rejected outright (278 of 379 organic).
     Aggressively reweighting toward keyword-overlap and lowering the borderline
     barely helped (recall 0.03 → 0.05) — the credibility floor and tier
     mis-classification dominate.

**Caveat on the metric:** our labels generously marked ~190 mostly-redundant
on-topic news items as keep, so absolute *recall* understates filter quality
(dedup/cap would collapse that tail anyway). **Precision (~0.5) and official
recall (1.0)** are the trustworthy numbers; precision ~0.5 means roughly half of
kept organic results are still off-topic — a real problem that the #4 relevance
term would address upstream.

**Recommendations (#13):**
- Keep `heuristic_keep_threshold` at **0.60–0.65** (anywhere in the plateau is
  equivalent); stop treating it as a tunable — it isn't, on this data.
- The bigger wins are upstream (#4 relevance term) and in **tier coverage**:
  promote major wire/national outlets out of `unknown` so their `domain_score`
  isn't a hard ceiling. Consider lowering the `0.25·credibility` blend weight in
  `compute_priority_score`.
- **Fail-closed fallback (the deferred decision):** the data now supports a
  decision. 84 organic candidates hit the band, but precision is already ~0.5,
  so a blanket "auto-keep all borderline ≥0.60" would push precision lower.
  Recommend a **targeted** soft fallback instead: when `llm_client=None`,
  auto-keep a borderline candidate only if it is `is_official_domain` **or**
  has high keyword-overlap relevance — recovering the on-topic tail without
  admitting the generic-news mass. Gate it behind a config flag so production
  (which has an LLM) is unchanged.

## Issue #3 — dashboard value vs organic search

**Dashboards are essential for off-peak pathogens but their value is bimodal at
extraction time.** End-to-end live runs:

| Q | pathogen | survivors | of which dashboard | insight records | note |
|---|---|---|---|---|---|
| L1 | H5N1 | 2 | 2 | 1 | CDC bird-flu *situation-summary* extracts (some content) |
| L3 | Ebola | 7 | 2 | 11 | rich organic; guard dropped 1 fabricated quote |
| L5 | hantavirus | 2 | 0 | 5 | organic-only (apnews+wapo) → good cruise-cluster facts |
| L7 | mpox | 3 | 3 | **0** | all 3 dashboards non-extractable (see below) |
| L9 | Marburg | 7 | 2 | 43 | WHO Marburg *factsheet* → many (historical) records |

- **Dashboards crowd out organic and then yield nothing when the URL is an
  interactive tracker / index / landing page.** L7 mpox is the worst case: the 3
  injected dashboards (`who.int/.../situation-reports` index,
  `cdc.gov/mpox/data-research` [a **404**], `ourworldindata.org/mpox`
  [JS viz]) extracted as 245/456/2358-char shells and produced **0 records** —
  while the one organic result with real mpox case data (fox6now Wisconsin
  cluster) was filtered out. The pipeline produced nothing usable for mpox.
- **Static fact-bearing dashboards do work**: the WHO Marburg factsheet (L9) and
  CDC bird-flu situation summary (L1) are prose pages and yielded records.
- **Staleness check** (all 11 URLs in `biosecurity_sources.py`):
  - **Broken:** `cdc.gov/mpox/data-research/index.html` → **404** (redirects to
    `monkeypox/...` which is also dead).
  - **Stale redirects (update to canonical):**
    `afro.who.int/health-topics/ebola-virus-disease` → `…/ebola-disease`;
    `cdc.gov/ebola/index.html` → `cdc.gov/ebola/about/index.html`.
  - Other 8 return 200.
- **Routing brittleness:** `DASHBOARD_LOOKUP` is matched by **exact** lowercased
  `pathogen` key. The CSV-natural topic "Marburg Virus Disease" yields
  `pathogen="marburg virus disease"`, which does **not** match the `marburg`
  key — L9 would have gotten **zero** on-topic results (we had to rename the
  topic to "Marburg" to make it route). Same risk for any multi-word pathogen.

**Recommendations (#3):**
- **Fix the broken/stale URLs** above. Prefer static, fact-bearing pages over
  interactive viz where possible: e.g. point mpox at a WHO mpox external
  situation-report landing or the OWID *data* endpoint, not the JS tracker.
- **Make routing tolerant** — substring/alias match (`"marburg" in pathogen`) or
  a normalization map, so multi-word pathogen names still route.
- **Re-evaluate the "extract dashboards as documents" assumption.** For
  interactive trackers the current value at the insight stage is ~0; either
  (a) attach a curated structured snapshot/PDF per dashboard, or (b) treat
  dashboards as resolution-source pointers rather than extractable documents,
  and don't let them consume `total_cap`/survival slots that organic
  fact-bearing sources need.
- Keep dashboards for **off-peak pathogens** (H5N1, mpox, Marburg) where they
  are the only on-topic authority; their problem is extractability, not
  relevance.

## Historical-replay confound (confirmed)

`q7` (mpox, resolved) replayed at cutoff 2025-02-17 survived 5 docs = 2
wayback-rewritten dashboards + 3 organic, but the 3 organic were **2024
preparedness/calendar pages** (whitehouse.gov, WHO event pages), not case-count
data → 1 insight record. This confirms `specs/tavily-historical-coverage.md`:
replay survival is dashboard/official-dominated with little fresh outbreak
signal, so **forecast-accuracy on resolved questions cannot drive filter/weight
tuning** — which is why we tuned against hand labels on live pools instead.

## Limitations

- Single point-in-time snapshot (late May 2026, Ebola-dominant). The relevance
  collapse for off-peak pathogens may be milder in a quieter news period.
- Labels are one annotator's judgment; gray-zone calls (opinion vs reportage,
  redundant news) affect absolute recall more than the qualitative conclusions.
- #4 truncation effect is not exercised here (pools < `total_cap`); the relevance
  term's benefit would be *larger* when pools exceed the cap.

## Suggested next actions (not done in this pass)

1. (#4) Add a relevance term to `_compute_search_stage_score` + unit test.
2. (#3) Fix the 1 broken + 2 stale dashboard URLs; make `DASHBOARD_LOOKUP`
   routing substring/alias-tolerant.
3. (#13) Implement the targeted `llm_client=None` soft fallback behind a flag.
4. (#13) Audit `tier_resolution` coverage so major wire/national outlets aren't
   `unknown`.

## Changes implemented (follow-up pass)

- **#4** — `_compute_search_stage_score` now includes a keyword-overlap
  relevance term (`0.45·relevance + 0.30·domain + 0.10·freshness + 0.15·rank`,
  sums to 1.0), reusing the filter's `keyword_overlap_score`/`build_query_terms`.
  Relevance is the dominant term; freshness kept low (near-uniform live).
- **#3** — fixed the broken CDC mpox URL (now the extractable
  `monkeypox/situation-summary` page; re-run yielded **2 records vs 0**), the
  two stale redirects (afro.who.int ebola-disease, cdc.gov/ebola/about), and
  made `DASHBOARD_LOOKUP` routing substring/alias-tolerant (`_resolve_pathogen_key`)
  so "marburg virus disease"→marburg, "monkeypox"→mpox, "bird flu"→h5n1.
- **#13 (tier coverage)** — added ~22 national/international outlets
  (CNN, NBC, CBS, ABC, NPR, USA Today, LA Times, Politico, Axios, Forbes,
  Bloomberg, FT, WSJ, Economist, Time, Atlantic, Ars Technica, Business Insider,
  …) to Tier 3 (`trusted_media`, domain_score 0.6) so legitimate outbreak
  reporting is no longer floored at `unknown`/0.2. Validation note: this raises
  recall of reputable reporting but, because the *filter's* `priority_score`
  still weights credibility heavily, it can also admit off-topic pieces from
  those outlets (e.g. an H5N1 run kept a CBS transcript) — the search-stage
  relevance term does not feed the filter's keep decision. Worth considering
  raising the filter's keyword-overlap weight / lowering its `0.25·credibility`
  blend as a follow-up.
- **#13 fail-closed fallback** — implemented option **C (targeted soft-keep)**
  behind a default-off flag `FILTER_CONFIG["no_llm_soft_fallback"]`
  (+ `no_llm_fallback_relevance_threshold`, default 0.5). When enabled and no LLM
  client is configured, a borderline ("llm_needed") candidate is kept iff it is
  an official domain OR its keyword-overlap relevance clears the threshold —
  approximating the LLM-rescue path for dev/offline runs. Default off, so
  production (always has an LLM client) is unchanged.
- All 455 tests pass; new tests added for the score formula, tolerant routing,
  tier coverage, and the soft-fallback flag.
