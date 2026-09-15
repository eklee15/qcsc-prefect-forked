#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
BIN_DIR="${SCRIPT_DIR}/bin"

mkdir -p "${BIN_DIR}"

module list >/dev/null 2>&1 || true

CXX=""
for cmd in mpicxx mpic++ mpicxx.openmpi mpiCC; do
  if command -v "${cmd}" >/dev/null 2>&1; then
    CXX="${cmd}"
    break
  fi
done

if [ -z "${CXX}" ]; then
  echo "No MPI C++ compiler was found. Load an MPI module first (e.g. 'module load openmpi')." >&2
  exit 1
fi

CXXFLAGS="-O3 -std=c++17"

"${CXX}" ${CXXFLAGS} -o "${BIN_DIR}/get_counts_json" "${SRC_DIR}/get_counts_json.cpp"
"${CXX}" ${CXXFLAGS} -o "${BIN_DIR}/get_counts_hist" "${SRC_DIR}/get_counts_hist.cpp"

echo "Built binaries with ${CXX}:"
ls -l "${BIN_DIR}/get_counts_json" "${BIN_DIR}/get_counts_hist"
