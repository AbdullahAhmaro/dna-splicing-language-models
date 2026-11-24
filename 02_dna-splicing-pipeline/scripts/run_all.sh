#!/usr/bin/env bash
# Phase-1 splice-site pipeline, start to finish, CPU only.
#
#   bash scripts/run_all.sh                               # 8,000 train / 2,000 test per model
#   MAX_TRAIN=-1 MAX_TEST=-1 bash scripts/run_all.sh      # the full dataset
#
# Order: 01 data -> (02 embed -> 03 heads) per model
#        -> 02b centre-pooled embeddings per model -> 12 previous-token variants
#        -> 03 heads per pooling variant
#        -> 04 baseline -> 07 pooling choice on validation -> 09 motif analysis
#        -> 13 hard negatives (training chromosomes) -> 03 --hardneg heads -> 07 --hardneg
#        -> 10 whole-gene scans of the 15 test-chromosome genes (long context; NT-v2
#           600-nt windows; the hard-negative heads) -> 11 scan report
#        -> 05 charts -> 06 summary.
# Everything is appended to results/run.log. A failing step is recorded and the
# run carries on, so one model failing does not lose the rest. Exit code is 1
# if any step failed. Embeddings already on disk for the same settings are
# reused, and an interrupted embedding resumes from its last checkpoint.
set -u -o pipefail

cd "$(dirname "$0")/.." || exit 1          # -> the pipeline folder (was dna-phase/)
mkdir -p results data
LOG=results/run.log
STEPS=results/run_steps.tsv

# Activate dna-venv from wherever this folder lives now. `source dna-venv/bin/activate`
# is not used here because activate hard-codes the absolute path the venv was created
# at: after the folder is moved or renamed it puts a missing directory on PATH and
# python silently falls back to the system interpreter (no torch). The venv's python
# finds its packages through dna-venv/pyvenv.cfg, so these two variables are enough.
VENV="$PWD/dna-venv"
[ -x "$VENV/bin/python" ] || { echo "cannot find $VENV/bin/python; run scripts/setup_local_env.sh" >&2; exit 1; }
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
unset PYTHONHOME
[ "$(python -c 'import sys; print(sys.prefix)')" = "$VENV" ] || { echo "python is not the dna-venv interpreter" >&2; exit 1; }
export PYTHONUNBUFFERED=1                  # progress lines reach the log immediately
export TOKENIZERS_PARALLELISM=false

MAX_TRAIN=${MAX_TRAIN:-8000}
MAX_TEST=${MAX_TEST:-2000}
MODELS=(
  "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
  "zhihan1996/DNABERT-2-117M"
)
VARIANTS=(center_tok center_w10 center_w25 center_w50 concat_c25_full
          prev_tok center_plus_prev center_plus_prev_concat)
# variants a per-position scan can apply (centre token and the token before it)
SCAN_VARIANTS=(center_tok prev_tok center_plus_prev center_plus_prev_concat)
NT="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
DB="zhihan1996/DNABERT-2-117M"

exec > >(tee -a "$LOG") 2>&1

START=$(date +%s)
echo
echo "########## run_all.sh started $(date '+%F %T')  MAX_TRAIN=$MAX_TRAIN MAX_TEST=$MAX_TEST"
echo "########## python $(python -V 2>&1 | cut -d' ' -f2), torch $(python -c 'import torch; print(torch.__version__)')"
printf '# started_epoch\t%s\n' "$START" > "$STEPS"

# Results from a previous run are kept aside so a failed model can't show stale numbers.
if [ -f results/phase1_results.json ]; then
  mv results/phase1_results.json results/phase1_results.prev.json
  echo "previous results moved to results/phase1_results.prev.json"
fi

FAILED=()
run_step() {
  local name=$1; shift
  echo
  echo "========== $(date '+%T') START $name"
  echo "\$ $*"
  local t0 rc dt
  t0=$(date +%s)
  "$@"
  rc=$?
  dt=$(( $(date +%s) - t0 ))
  printf '%s\t%s\t%s\n' "$name" "$rc" "$dt" >> "$STEPS"
  if [ "$rc" -eq 0 ]; then
    echo "========== $(date '+%T') OK     $name (${dt}s)"
  else
    echo "========== $(date '+%T') FAILED $name (exit $rc after ${dt}s); continuing"
    FAILED+=("$name")
  fi
  return "$rc"
}

run_step 01_get_data python scripts/01_get_data.py

for m in "${MODELS[@]}"; do
  short=${m##*/}
  if run_step "02_embed:$short" python scripts/02_embed.py --model "$m" \
       --max-train "$MAX_TRAIN" --max-test "$MAX_TEST"; then
    run_step "03_train_head:$short" python scripts/03_train_head.py --model "$m"
  else
    echo "skipping 03_train_head for $m: no embeddings"
    printf '%s\t%s\t%s\n' "03_train_head:$short" "skipped" 0 >> "$STEPS"
  fi
done

CENTER_OK=()
for m in "${MODELS[@]}"; do
  short=${m##*/}
  if run_step "02b_embed_center:$short" python scripts/02b_embed_center.py --model "$m" \
       --max-train "$MAX_TRAIN" --max-test "$MAX_TEST"; then
    CENTER_OK+=("$m")
  fi
done
# previous-token variants, re-pooled from the saved centre tokens (no forward pass)
run_step 12_repool_center python scripts/12_repool_center.py

for m in "${MODELS[@]}"; do
  short=${m##*/}
  if [[ " ${CENTER_OK[*]} " == *" $m "* ]]; then
    for v in "${VARIANTS[@]}"; do
      run_step "03_train_head:$short:$v" python scripts/03_train_head.py --model "$m" --variant "$v"
    done
  else
    echo "skipping variant heads for $m: no centre-pooled embeddings"
    for v in "${VARIANTS[@]}"; do
      printf '%s\t%s\t%s\n' "03_train_head:$short:$v" "skipped" 0 >> "$STEPS"
    done
  fi
done

run_step 04_baseline python scripts/04_baseline.py --max-test "$MAX_TEST"
run_step 07_select_variant python scripts/07_select_variant.py
run_step 09_motif_analysis python scripts/09_motif_analysis.py --max-test "$MAX_TEST" --max-train "$MAX_TRAIN"
# Hard negatives from training-chromosome genes (Ensembl the first time, then cached),
# heads trained with them (same recipe), and the 07 protocol for them.
HN_OK=0
if run_step 13_hard_negatives python scripts/13_hard_negatives.py; then
  HN_OK=1
  for m in "${MODELS[@]}"; do
    short=${m##*/}
    for v in "${SCAN_VARIANTS[@]}"; do
      run_step "03_train_head:$short:$v:hardneg" python scripts/03_train_head.py --model "$m" \
           --variant "$v" --hardneg
    done
  done
  run_step 07_select_variant:hardneg python scripts/07_select_variant.py --hardneg
fi
# Whole-gene scans of the 15 test-chromosome genes: Ensembl the first time, then the
# data/genes/ cache. The NT-v2 window scan is ~2 h the first time (token vectors are
# cached in data/scan/, so later runs and other heads take seconds).
if run_step 10_scan_gene python scripts/10_scan_gene.py; then
  # reference: the first-round centre-token head, for the head comparison in the report
  run_step 10_scan_gene:center_tok python scripts/10_scan_gene.py --variant center_tok
  run_step 10_scan_gene:window python scripts/10_scan_gene.py --context window --models "$NT"
  if [ "$HN_OK" -eq 1 ]; then
    run_step 10_scan_gene:hardneg:window python scripts/10_scan_gene.py --head hardneg \
         --context window --models "$NT"
    run_step 10_scan_gene:hardneg python scripts/10_scan_gene.py --head hardneg --models "$DB"
  fi
  run_step 11_scan_report python scripts/11_scan_report.py
else
  printf '%s\t%s\t%s\n' "11_scan_report" "skipped" 0 >> "$STEPS"
fi
run_step 05_chart    python scripts/05_chart.py
run_step 06_summary  python scripts/06_summary.py

total=$(( $(date +%s) - START ))
echo
echo "########## finished $(date '+%F %T') in $((total / 60))m $((total % 60))s"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "########## FAILED steps: ${FAILED[*]}"
  exit 1
fi
echo "########## all steps succeeded"
