#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
OUTPUT_DIR="${1:-${BUILD_DIR}/test/benchmark}"
TRIALS="${2:-20}"
NUM_VECTORS="${3:-12}"
OUTLIER_RATIO="${4:-0.2}"
MAX_ITERS="${5:-50}"

mkdir -p "${OUTPUT_DIR}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" \
  --target certification_redundant_constraints_montecarlo certification_redundant_constraints_benchmark \
  -j4

CSV_PATH="${OUTPUT_DIR}/certification_redundant_constraints_montecarlo.csv"
PREFIX="${OUTPUT_DIR}/certification_redundant_constraints"

"${BUILD_DIR}/test/benchmark/certification_redundant_constraints_montecarlo" \
  "${CSV_PATH}" "${TRIALS}" "${NUM_VECTORS}" "${OUTLIER_RATIO}" "${MAX_ITERS}"

python3 "${ROOT_DIR}/test/benchmark/plot_certification_redundant_constraints.py" \
  "${CSV_PATH}" "${PREFIX}"

echo "Generated:"
echo "  ${PREFIX}.png"
echo "  ${PREFIX}.pdf"
echo "  ${CSV_PATH}"
