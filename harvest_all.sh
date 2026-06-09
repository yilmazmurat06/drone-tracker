#!/bin/bash
set -Eeuo pipefail

ROOT="/Users/murat/Documents/GitHub/drone-tracker"
cd "$ROOT"

PY="/opt/anaconda3/bin/python3"
COMMON="--max-active 12 --seed-every 10 --min-len 25"
LOG_DIR="data/pseudo"
LOG_FILE="$LOG_DIR/harvest_all.log"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

trap 'echo "[ERROR] ${BASH_SOURCE[0]}:${LINENO} komut: ${BASH_COMMAND}" >&2' ERR

run_step() {
  local label="$1"
  shift
  local started_at finished_at duration

  started_at=$(date +%s)
  echo ""
  echo "########## ${label} ##########"
  "$@"
  finished_at=$(date +%s)
  duration=$((finished_at - started_at))
  echo "[OK] ${label} (${duration}s)"
}

assert_file() {
  local file_path="$1"
  if [[ ! -s "$file_path" ]]; then
    echo "[ERROR] Beklenen dosya yok ya da boş: $file_path" >&2
    exit 1
  fi
}

echo "[INFO] harvest_all başladı: $(date)"
echo "[INFO] log: $LOG_FILE"

assert_file "data/flight_01_084727.mp4"
assert_file "data/flight_02_084915.mp4"
assert_file "data/flight_03_085014.mp4"

run_step "flight_01 (exclude TEST 3084:3504) / collect" \
  "$PY" tools/collect_trajectories.py data/flight_01_084727.mp4 \
    --out data/pseudo/flight_01_raw.csv --exclude 3084:3504 $COMMON
run_step "flight_01 filter" \
  "$PY" tools/filter_trajectories.py data/flight_01_084727.mp4 \
    data/pseudo/flight_01_raw.csv --out data/pseudo/flight_01_clean.csv

run_step "flight_02 / collect" \
  "$PY" tools/collect_trajectories.py data/flight_02_084915.mp4 \
    --out data/pseudo/flight_02_raw.csv $COMMON
run_step "flight_02 filter" \
  "$PY" tools/filter_trajectories.py data/flight_02_084915.mp4 \
    data/pseudo/flight_02_raw.csv --out data/pseudo/flight_02_clean.csv

run_step "flight_03 / collect" \
  "$PY" tools/collect_trajectories.py data/flight_03_085014.mp4 \
    --out data/pseudo/flight_03_raw.csv $COMMON
run_step "flight_03 filter" \
  "$PY" tools/filter_trajectories.py data/flight_03_085014.mp4 \
    data/pseudo/flight_03_raw.csv --out data/pseudo/flight_03_clean.csv

echo ""
echo "########## TAMAMLANDI ##########"
wc -l data/pseudo/flight_0*_clean.csv