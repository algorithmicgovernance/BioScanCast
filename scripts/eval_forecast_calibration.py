"""Offline calibration + extremize analysis for the forecasting stage.

Pure math, no API calls. Reads the committed historical-replay trajectory
CSVs (data/investigations/q{1,3,7,9}_trajectory_forecasts.csv), and for the
evidence-based `bioscancast` forecast at each cutoff computes Brier, then
evaluates the `extremize` knob (normalize(p**(1+s))) under:

  * static strengths, and
  * time-ramped schedules keyed on days-to-resolution,

to test the hypothesis that extremizing should be turned on near resolution
(for slow-dynamics questions). Confirms the benchmark's under-confidence
finding and surfaces the key risk: extremize amplifies confident-but-wrong
forecasts (the pre-fix lurch cutoffs, and q7's genuine miss).

If post-fix replay artifacts exist under data/runs (the validate_* runs from
the extraction-prompt fix), their distributions overlay the stale pre-fix
cutoffs so the picture is closer to current behaviour.

    python scripts/eval_forecast_calibration.py
    python scripts/eval_forecast_calibration.py --near-days 5 --strength 1.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

INVEST = Path("data/investigations")
RUNS = Path("data/runs")

# Resolved bucket + resolution date for each scorable BFG question
# (expanded to 11 on 2026-06-21; see data/investigations/bfg-manual-resolutions.md).
TRUTH: Dict[str, str] = {
    "q1": "70-100", "q2": "70-100", "q3": "970-1000", "q4": "15-20",
    "q5": "NO", "q6": "YES", "q7": "126,001-128,500", "q8": "131,001+",
    "q9": "11+", "q10": "9-15", "q11": "Vector-borne",
}
TARGET: Dict[str, date] = {
    "q1": date(2025, 2, 28), "q2": date(2026, 1, 1), "q3": date(2025, 2, 28),
    "q4": date(2025, 2, 28), "q5": date(2025, 5, 1), "q6": date(2025, 5, 1),
    "q7": date(2025, 2, 28), "q8": date(2025, 5, 1), "q9": date(2025, 5, 1),
    "q10": date(2025, 6, 30), "q11": date(2025, 3, 31),
}
ALL_QIDS = [f"q{i}" for i in range(1, 12)]
# Pre-fix lurch cutoffs documented in the benchmark report.
LURCH = {("q3", "2025-02-23"), ("q9", "2025-04-04")}
# Post-fix replay artifacts (run-ids from the extraction-fix validation).
OVERLAY = {
    ("q3", "2025-02-23"): RUNS / "q3/validate_q3_0223/forecast.json",
    ("q9", "2025-04-04"): RUNS / "q9/validate_q9_0404/forecast.json",
    ("q1", "2025-02-25"): RUNS / "q1/validate_q1_0225/forecast.json",
}


def _norm(dist: Dict[str, float]) -> Dict[str, float]:
    total = sum(dist.values())
    return {k: v / total for k, v in dist.items()} if total > 0 else dist


def extremize(dist: Dict[str, float], strength: float) -> Dict[str, float]:
    if strength == 0.0:
        return dict(dist)
    return _norm({k: max(v, 1e-9) ** (1.0 + strength) for k, v in dist.items()})


def brier(dist: Dict[str, float], truth: str) -> float:
    return sum((p - (1.0 if opt == truth else 0.0)) ** 2 for opt, p in dist.items())


def _load_cutoffs(
    qid: str, overlay: bool, invest_dir: Path = INVEST
) -> List[Tuple[str, Dict[str, float], int, bool]]:
    """Return [(cutoff, dist, days_to_resolution, is_lurch)] for one question."""
    df = pd.read_csv(invest_dir / f"{qid}_trajectory_forecasts.csv", sep=";", decimal=",")
    df["probability"] = df["probability"].astype(float)
    out = []
    versions = sorted(
        v for v in df["forecast_version"].unique() if v.startswith("bioscancast@")
    )
    for v in versions:
        cutoff = v.split("@")[1]
        sub = df[df["forecast_version"] == v]
        dist = _norm({r["option"]: float(r["probability"]) for _, r in sub.iterrows()})
        if overlay and (qid, cutoff) in OVERLAY and OVERLAY[(qid, cutoff)].exists():
            fc = json.load(open(OVERLAY[(qid, cutoff)]))
            for d in fc["distributions"]:
                if d["forecast_source"] == "bioscancast":
                    dist = _norm(dict(d["probabilities"]))
        days = (TARGET[qid] - datetime.strptime(cutoff, "%Y-%m-%d").date()).days
        out.append((cutoff, dist, days, (qid, cutoff) in LURCH))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--near-days", type=int, default=5,
                   help="Cutoffs with days-to-resolution <= this are 'near'.")
    p.add_argument("--strength", type=float, default=1.0,
                   help="Extremize strength used in the near/far + ramp analyses.")
    p.add_argument("--gate", type=float, default=0.5,
                   help="Peak-probability threshold for the confidence-gated schedule.")
    p.add_argument("--no-overlay", action="store_true",
                   help="Ignore post-fix data/runs artifacts (pure committed CSVs).")
    p.add_argument("--invest-dir", default="data/investigations",
                   help="Directory of {qid}_trajectory_forecasts.csv. Use "
                   "data/investigations/postfix for the post-fix regeneration.")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    overlay = not args.no_overlay
    invest_dir = Path(args.invest_dir)

    rows = []  # (qid, cutoff, days, lurch, dist)
    for qid in ALL_QIDS:
        if not (invest_dir / f"{qid}_trajectory_forecasts.csv").exists():
            continue
        for cutoff, dist, days, lurch in _load_cutoffs(qid, overlay, invest_dir):
            rows.append((qid, cutoff, days, lurch, dist))

    overlaid = [k for k in OVERLAY if overlay and OVERLAY[k].exists()]
    print(f"Loaded {len(rows)} cutoffs across q1/q3/q7/q9 "
          f"(post-fix overlay: {sorted(overlaid) or 'none'})\n")

    # --- Per-cutoff table ---
    print(f"{'q':<4}{'cutoff':<12}{'days':>5}{'top':<16}{'truth_p':>8}"
          f"{'Brier0':>8}{f'Brier@{args.strength:g}':>9}  note")
    for qid, cutoff, days, lurch, dist in rows:
        top = max(dist, key=dist.get)
        tp = dist.get(TRUTH[qid], 0.0)
        b0 = brier(dist, TRUTH[qid])
        bs = brier(extremize(dist, args.strength), TRUTH[qid])
        note = "LURCH" if lurch else ("MISS" if top != TRUTH[qid] else "")
        print(f"{qid:<4}{cutoff:<12}{days:>5}{top:<16}{tp:>8.3f}"
              f"{b0:>8.3f}{bs:>9.3f}  {note}")

    # --- Under-confidence (baseline) ---
    correct = [(qid, d) for qid, _, _, _, d in rows if max(d, key=d.get) == TRUTH[qid]]
    if correct:
        mean_tp = sum(d[TRUTH[qid]] for qid, d in correct) / len(correct)
        print(f"\nUnder-confidence check: on the {len(correct)}/{len(rows)} cutoffs "
              f"where the top bucket is correct, mean prob on truth = {mean_tp:.3f} "
              f"(1.0 = perfectly confident).")

    # --- Static extremize sweep ---
    print("\nStatic extremize sweep (mean Brier; lower is better):")
    print(f"  {'strength':>9}{'all':>9}{'excl-lurch':>12}{'excl-miss':>11}")
    for s in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        all_b, nolurch, nomiss = [], [], []
        for qid, cutoff, days, lurch, dist in rows:
            b = brier(extremize(dist, s), TRUTH[qid])
            all_b.append(b)
            if not lurch:
                nolurch.append(b)
            if max(dist, key=dist.get) == TRUTH[qid]:
                nomiss.append(b)
        print(f"  {s:>9.1f}{sum(all_b)/len(all_b):>9.3f}"
              f"{sum(nolurch)/len(nolurch):>12.3f}{sum(nomiss)/len(nomiss):>11.3f}")

    # --- Time structure: does extremize help more near resolution? ---
    s = args.strength
    near = [(qid, dist) for qid, _, days, _, dist in rows if days <= args.near_days]
    far = [(qid, dist) for qid, _, days, _, dist in rows if days > args.near_days]
    def mean_delta(group):
        if not group:
            return None
        return sum(brier(extremize(d, s), TRUTH[q]) - brier(d, TRUTH[q])
                   for q, d in group) / len(group)
    print(f"\nTime structure of extremize (strength {s:g}; negative delta = improvement):")
    print(f"  near (<= {args.near_days}d): n={len(near)}  mean Brier delta = {mean_delta(near):+.3f}")
    print(f"  far  (>  {args.near_days}d): n={len(far)}  mean Brier delta = {mean_delta(far):+.3f}")

    # --- Time-ramped schedule vs static ---
    print("\nSchedule comparison (mean Brier over all cutoffs):")
    base = sum(brier(d, TRUTH[q]) for q, _, _, _, d in rows) / len(rows)
    static = sum(brier(extremize(d, s), TRUTH[q]) for q, _, _, _, d in rows) / len(rows)
    ramp = sum(
        brier(extremize(d, s if days <= args.near_days else 0.0), TRUTH[q])
        for q, _, days, _, d in rows
    ) / len(rows)
    n_gated = sum(1 for _, _, _, _, d in rows if max(d.values()) >= args.gate)
    gated = sum(
        brier(extremize(d, s if max(d.values()) >= args.gate else 0.0), TRUTH[q])
        for q, _, _, _, d in rows
    ) / len(rows)
    print(f"  off (strength 0):                  {base:.3f}")
    print(f"  static (strength {s:g} always):        {static:.3f}")
    print(f"  time-ramp (strength {s:g} <= {args.near_days}d):     {ramp:.3f}")
    print(f"  conf-gated (strength {s:g} if peak>={args.gate:g}): {gated:.3f}  "
          f"(extremizes {n_gated}/{len(rows)} cutoffs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
