#!/usr/bin/bash
# Step 3/3 wrapper: sample the circuit on the QPU.
#
# The executable the Slurm job for the sampling step runs on the node. The
# #SBATCH header and HL_PYTHON / HL_JOB_NAME come from the rendered sbatch
# script (from the blocks); this file is the body.
#
# Spends QPU shots (qpu_shots in the config). sample_circuit.py refuses to
# overwrite an existing samples.json, so the flow moves prior results aside
# first when asked (reset). The QRMI Slurm plugin (quantum partition + --qpu)
# provides the QPU resource and credentials to HL_PYTHON.
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

echo "=== sampling: job context ==="
echo "Job name : ${JOB_NAME}"
echo "Job id   : ${SLURM_JOB_ID:-<none>}"
echo "Host     : $(hostname)"
echo "Python   : ${HL_PYTHON}"
echo "Config   : ${CONFIG}"

[ -x "${HL_PYTHON}" ] || die "interpreter not found: ${HL_PYTHON} (set HL_PYTHON)"
[ -f "${CONFIG}" ] || die "missing ${CONFIG}"

circuit=$(awk '/^circuit_data:/ {print $2}' "${CONFIG}" | tr -d '"')
[ -f "${circuit}" ] || die "circuit ${circuit} not built yet (run the make_circuit step first)"

echo "=== sampling: sampling ${JOB_NAME} on the QPU ==="
exec "${HL_PYTHON}" -u sample_circuit.py "${JOB_NAME}"
