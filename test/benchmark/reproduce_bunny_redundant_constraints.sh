#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
OUTPUT_DIR="${1:-${ROOT_DIR}/../../assets/teaser_analysis}"
TRIALS="${2:-100}"
NUM_POINTS="${3:-20}"
OUTLIER_RATIO="${4:-0.2}"
NOISE_BOUND="${5:-0.01}"
SOLVER="${6:-CLARABEL}"

mkdir -p "${OUTPUT_DIR}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --target bunny_teaser_candidate_benchmark -j4

TRIAL_SUMMARY_CSV="${OUTPUT_DIR}/bunny_teaser_candidate_trials.csv"
TIMS_CSV="${OUTPUT_DIR}/bunny_teaser_candidate_tims.csv"
PRIMAL_RESULTS_CSV="${OUTPUT_DIR}/bunny_sdp_primal_results.csv"
SUMMARY_CSV="${OUTPUT_DIR}/bunny_sdp_summary.csv"

"${BUILD_DIR}/test/benchmark/bunny_teaser_candidate_benchmark" \
  "${TRIAL_SUMMARY_CSV}" "${TIMS_CSV}" "${TRIALS}" "${NUM_POINTS}" "${OUTLIER_RATIO}" \
  "${NOISE_BOUND}" "${ROOT_DIR}/test/teaser/data/bunny.pcd"

if [[ ! -d "${ROOT_DIR}/.venv-analysis" ]]; then
  python3 -m venv "${ROOT_DIR}/.venv-analysis"
fi

if ! "${ROOT_DIR}/.venv-analysis/bin/python" - <<'PY' >/dev/null 2>&1
import importlib.util
mods = ["cvxpy", "clarabel", "pandas", "matplotlib", "numpy"]
raise SystemExit(0 if all(importlib.util.find_spec(m) for m in mods) else 1)
PY
then
  "${ROOT_DIR}/.venv-analysis/bin/pip" install --upgrade pip
  "${ROOT_DIR}/.venv-analysis/bin/pip" install cvxpy clarabel pandas matplotlib numpy
fi

"${ROOT_DIR}/.venv-analysis/bin/python" "${ROOT_DIR}/test/benchmark/sdp_rank_montecarlo.py" \
  "${PRIMAL_RESULTS_CSV}" "${SUMMARY_CSV}" "${TRIAL_SUMMARY_CSV}" "${TIMS_CSV}" "${SOLVER}"

"${ROOT_DIR}/.venv-analysis/bin/python" "${ROOT_DIR}/test/benchmark/plot_bunny_redundant_constraints.py" \
  "${PRIMAL_RESULTS_CSV}" "${OUTPUT_DIR}"

echo "Generated:"
echo "  ${TRIAL_SUMMARY_CSV}"
echo "  ${TIMS_CSV}"
echo "  ${PRIMAL_RESULTS_CSV}"
echo "  ${SUMMARY_CSV}"
find "${OUTPUT_DIR}" -maxdepth 1 -type f \( -name 'bunny_*_hist.png' -o -name 'bunny_*_hist.pdf' \) | sort
