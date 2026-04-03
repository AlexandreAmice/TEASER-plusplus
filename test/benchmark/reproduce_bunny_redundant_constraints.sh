#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
OUTPUT_DIR="${1:-${ROOT_DIR}/../../assets/teaser_analysis}"
TRIALS="${2:-12}"
NUM_POINTS="${3:-12}"
OUTLIER_RATIO="${4:-0.2}"
NOISE_BOUND="${5:-0.01}"
MAX_ITERS="${6:-50}"

mkdir -p "${OUTPUT_DIR}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --target bunny_certifier_suboptimality_benchmark -j4

CERTIFIER_CSV="${OUTPUT_DIR}/bunny_certifier_suboptimality.csv"
RANK_CSV="${OUTPUT_DIR}/bunny_sdp_rank_distribution.csv"
GAP_PREFIX="${OUTPUT_DIR}/bunny_suboptimality_gap_distribution"
RANK_PREFIX="${OUTPUT_DIR}/bunny_sdp_rank_distribution"

"${BUILD_DIR}/test/benchmark/bunny_certifier_suboptimality_benchmark" \
  "${CERTIFIER_CSV}" "${TRIALS}" "${NUM_POINTS}" "${OUTLIER_RATIO}" "${NOISE_BOUND}" "${MAX_ITERS}" \
  "${ROOT_DIR}/test/teaser/data/bunny.pcd"

if [[ ! -d "${ROOT_DIR}/.venv-analysis" ]]; then
  python3 -m venv "${ROOT_DIR}/.venv-analysis"
  "${ROOT_DIR}/.venv-analysis/bin/pip" install --upgrade pip
  "${ROOT_DIR}/.venv-analysis/bin/pip" install cvxpy pandas matplotlib numpy
fi

"${ROOT_DIR}/.venv-analysis/bin/python" "${ROOT_DIR}/test/benchmark/sdp_rank_montecarlo.py" \
  "${RANK_CSV}" "${TRIALS}" "${NUM_POINTS}" "${OUTLIER_RATIO}" "${NOISE_BOUND}" SCS \
  "${ROOT_DIR}/test/teaser/data/bunny.pcd"

python3 "${ROOT_DIR}/test/benchmark/plot_bunny_redundant_constraints.py" \
  "${CERTIFIER_CSV}" "${RANK_CSV}" "${GAP_PREFIX}" "${RANK_PREFIX}"

echo "Generated:"
echo "  ${GAP_PREFIX}.png"
echo "  ${GAP_PREFIX}.pdf"
echo "  ${RANK_PREFIX}.png"
echo "  ${RANK_PREFIX}.pdf"
echo "  ${CERTIFIER_CSV}"
echo "  ${RANK_CSV}"
