#!/usr/bin/bash
# Step 2/3 wrapper: build + transpile the LUCJ circuit.
#
# The executable the Slurm job for the make_circuit step runs on the node. The
# #SBATCH header and HL_PYTHON / HL_JOB_NAME come from the rendered sbatch
# script (from the blocks); this file is the body.
#
# The circuit is transpiled against the target QRMI reports for the QPU at this
# moment, so this runs inside a quantum-partition job with --qpu set, same as
# the other steps. If the circuit file already exists it is reused (delete it to
# rebuild against the current target).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

JOB_NAME="${HL_JOB_NAME:-${SLURM_JOB_NAME:-h14_qpu}}"
CONFIG="jobs/${JOB_NAME}/config.yaml"
HL_PYTHON="${HL_PYTHON:-/gpfs/u/scratch/QCSC/shared/osama/sc_env/venvs/pyscf-3.11/bin/python}"

# Activate the compute environment (pyscf-3.11): sc_env/env.sh puts its OpenBLAS,
# GCC 15 and MPI libraries first on LD_LIBRARY_PATH and activates the venv.
SC_ENV="${SC_ENV:-/gpfs/u/scratch/QCSC/shared/osama/sc_env}"
export SC_ENV
if [ -f "${SC_ENV}/env.sh" ]; then
    set +u
    # shellcheck disable=SC1091
    source "${SC_ENV}/env.sh"
    set -u
fi

export TMPDIR="${TMPDIR:-${SCRIPT_DIR}/.tmp}"
mkdir -p "${TMPDIR}"

die() { echo "ERROR: $*" >&2; exit 1; }

echo "=== make_circuit: job context ==="
echo "Job name : ${JOB_NAME}"
echo "Job id   : ${SLURM_JOB_ID:-<none>}"
echo "Host     : $(hostname)"
echo "Python   : ${HL_PYTHON}"
echo "Config   : ${CONFIG}"

[ -x "${HL_PYTHON}" ] || die "interpreter not found: ${HL_PYTHON} (set HL_PYTHON)"
[ -f "${CONFIG}" ] || die "missing ${CONFIG}"

circuit=$(awk '/^circuit_data:/ {print $2}' "${CONFIG}" | tr -d '"')
if [ -f "${circuit}" ]; then
    echo "reusing existing ${circuit} (delete it to rebuild against the current target)"
    exit 0
fi

echo "=== make_circuit: building ${circuit} ==="
exec "${HL_PYTHON}" -u make_circuit.py "${CONFIG}"
