#!/usr/bin/env bash
#
# End-to-end NHIS 2021 + 2024 run: convert raw extracts, then analyze each year
# through the general survey-semantics pipeline (scales + survey weights).
#
# Prerequisites (NOT committed — NHIS data and the replication script are yours):
#   - The replication script with ALL_QUESTION_TEXTS / ALL_ITEM_SCALES.
#   - Raw adult CSVs: adult21.csv (2021) and adult24.csv (2024).
#
# Override any path via environment variables; defaults assume a gitignored
# ./data/NHIS working dir at the repo root.
#
# Usage:
#   examples/nhis/run_nhis.sh                 # bge-m3 if MODEL set, else tfidf
#   EMBEDDING=tfidf examples/nhis/run_nhis.sh # force offline smoke test
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

DATA_DIR="${DATA_DIR:-$REPO/data/NHIS}"
SCRIPT="${SCRIPT:-$DATA_DIR/nhis_replication_2021_2024.py}"
ADULT21="${ADULT21:-$DATA_DIR/2021/adult21.csv}"
ADULT24="${ADULT24:-$DATA_DIR/2024/adult24.csv}"
OUTDIR="${OUTDIR:-$REPO/outputs/nhis}"

# Embedding backend: a local bge-m3 path (MODEL=...) uses sentence-transformers;
# otherwise fall back to the offline tfidf smoke test.
MODEL="${MODEL:-}"
if [[ -n "$MODEL" ]]; then
  EMBEDDING="${EMBEDDING:-sentence-transformers}"
  MODEL_ARGS=(--embedding "$EMBEDDING" --model "$MODEL")
else
  EMBEDDING="${EMBEDDING:-tfidf}"
  MODEL_ARGS=(--embedding "$EMBEDDING")
fi

echo "==> 1/3  Convert raw NHIS extracts -> tool-ready per-year files"
python "$HERE/convert_nhis.py" \
  --script   "$SCRIPT" \
  --data2021 "$ADULT21" \
  --data2024 "$ADULT24" \
  --outdir   "$DATA_DIR"

for YEAR in 2021 2024; do
  echo "==> 2/3  Analyze NHIS $YEAR ($EMBEDDING)"
  python -m survey_semantics.cli analyze-file "$DATA_DIR/$YEAR/nhis$YEAR.csv" \
    --prompt-file  "$DATA_DIR/$YEAR/nhis${YEAR}_prompts.csv" \
    --scale-file   "$DATA_DIR/$YEAR/nhis${YEAR}_scales.csv" \
    --weights-file "$DATA_DIR/$YEAR/nhis${YEAR}_weights.csv" \
    --id-col HHX \
    "${MODEL_ARGS[@]}" \
    --d-selection variance --variance-threshold 0.80 --max-components 0 \
    --pan-mild --empirical-percentiles 95 99 \
    --skip-umap \
    --outdir "$OUTDIR/$YEAR"
done

echo "==> 3/3  Done. Outputs under: $OUTDIR/{2021,2024}"
echo "    Ranked outliers:  $OUTDIR/<year>/nhis<year>__*_scores.csv"
