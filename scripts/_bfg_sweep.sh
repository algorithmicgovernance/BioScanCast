#!/usr/bin/env bash
# Phase-1 evidence-only sweep across all 25 BFG summer-2026 questions.
# Sequential (shared 24h SQLite search cache dislikes concurrent writers).
# Stamps the run date so the live-drift window is auditable.
set -u
cd "$(dirname "$0")/.."

QCSV="bioscancast/stages/evaluation/bfg_summer_2026_questions.csv"
OCSV="bioscancast/stages/evaluation/bfg_summer_2026_options.csv"
LOG="${1:-sweep.log}"
STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "=== BFG evidence-only sweep started ${STAMP} ===" | tee "$LOG"
for i in $(seq 1 25); do
  qid="bfg_q${i}"
  echo "--- ${qid} $(date -u +%H:%M:%S) ---" | tee -a "$LOG"
  python -m bioscancast.main "$qid" --csv "$QCSV" --forecasts-csv "$OCSV" \
      --no-forecast >>"$LOG" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "    ${qid} OK" | tee -a "$LOG"
  else
    echo "    ${qid} FAILED rc=${rc}" | tee -a "$LOG"
  fi
done
echo "=== sweep complete $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
