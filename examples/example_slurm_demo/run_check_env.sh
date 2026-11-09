#!/usr/bin/bash
# Step 1/3 wrapper: the QRMI / environment preflight.
#
# This is the executable the Slurm job for the check_env step runs on the node.
# The #SBATCH header (partition, --qpu, --gres, nodes, walltime) and the
# HL_PYTHON / HL_JOB_NAME environment are supplied by the rendered sbatch script
# (from the blocks via flow.py); this file is just the body.
#
# HL_PYTHON is the compute interpreter (pyscf-3.11) that has ffsim / pyscf /
# qiskit / qrmi. The QRMI Slurm plugin (quantum partition + --qpu) provides the
# QPU resource and credentials to it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

JOB_NAME="${HL_JOB_NAME:-${SLURM_JOB_NAME:-h14_qpu}}"
CONFIG="jobs/${JOB_NAME}/config.yaml"
HL_PYTHON="${HL_PYTHON:-/gpfs/u/scratch/QCSC/shared/osama/sc_env/venvs/pyscf-3.11/bin/python}"

# Activate the compute environment (pyscf-3.11): sc_env/env.sh puts its OpenBLAS,
# GCC 15 and MPI libraries first on LD_LIBRARY_PATH and activates the venv.
# SC_ENV lets check_env.py recognize those libraries as this env's own.
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

echo "=== check_env: job context ==="
echo "Job name : ${JOB_NAME}"
echo "Job id   : ${SLURM_JOB_ID:-<none>}"
echo "Host     : $(hostname)"
echo "Python   : ${HL_PYTHON}"
echo "Config   : ${CONFIG}"

[ -x "${HL_PYTHON}" ] || die "interpreter not found: ${HL_PYTHON} (set HL_PYTHON)"
[ -f "${CONFIG}" ] || die "missing ${CONFIG}"

backend_type=$(awk '/^backend_type:/ {print $2}' "${CONFIG}" | tr -d '"')
[ "${backend_type}" = "qrmi" ] || die "backend_type in ${CONFIG} is '${backend_type}', not 'qrmi'"

echo "=== check_env: QRMI / environment check ==="
exec "${HL_PYTHON}" -u check_env.py --qrmi --config "${CONFIG}"
