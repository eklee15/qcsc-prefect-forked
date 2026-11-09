#!/usr/bin/bash
# Standalone sampler job -- the body SLURM runs on the compute node.
#
# Self-contained version of hydrogen-lattice's workflow/3_sample_qpu_sbatch.sh:
# every Python script it runs (check_env.py, build_hamiltonian.py,
# make_circuit.py, sample_circuit.py) and every data file (config, Hamiltonian,
# pre-built circuit) lives in THIS directory. It does NOT need the
# hydrogen-lattice repo checked out.
#
# The #SBATCH header (partition, --qpu, --gres, nodes, walltime, job name) is
# supplied by flow.py when it renders the wrapper script; this file is the
# executable that wrapper runs. flow.py sets the job's working directory to this
# directory, so the relative paths below resolve against it.
#
# The one thing that cannot be vendored into a directory is the Python
# environment: ffsim / pyscf / qiskit / qrmi are large binary packages. Point
# HL_PYTHON at an interpreter that has them (defaults to the project's conda
# env), or activate such an env before submitting.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

JOB_NAME="${SLURM_JOB_NAME:-h14_qpu}"
CONFIG="jobs/${JOB_NAME}/config.yaml"
CIRCUIT_ONLY="${CIRCUIT_ONLY:-}"

# Interpreter with ffsim/pyscf/qiskit/qrmi installed. The QRMI Slurm plugin
# (from the quantum partition + --qpu) still provides the QPU resource and
# credentials to whatever Python runs here.
HL_PYTHON="${HL_PYTHON:-/gpfs/u/scratch/QCSC/shared/ek/rpi_qcsc_demo/hl/hydrogen-lattice/.envs/envs/aimos/bin/python}"

# GPFS-backed scratch for pyscf/qiskit temp files.
export TMPDIR="${TMPDIR:-${SCRIPT_DIR}/.tmp}"
mkdir -p "${TMPDIR}"

die() { echo "ERROR: $*" >&2; exit 1; }

echo "=== Job context ==="
echo "Job name   : ${JOB_NAME}"
echo "Job id     : ${SLURM_JOB_ID:-<none>}"
echo "Batch host : $(hostname)"
echo "Work dir   : ${SCRIPT_DIR}"
echo "Python     : ${HL_PYTHON}"
echo "Config     : ${CONFIG}"

[ -x "${HL_PYTHON}" ] || die "interpreter not found: ${HL_PYTHON} (set HL_PYTHON)"
[ -f "${CONFIG}" ] || die "missing ${CONFIG}"

backend_type=$(awk '/^backend_type:/ {print $2}' "${CONFIG}" | tr -d '"')
[ "${backend_type}" = "qrmi" ] || die "backend_type in ${CONFIG} is '${backend_type}', not 'qrmi'"

echo "=== 1. QRMI / environment check ==="
"${HL_PYTHON}" -u check_env.py --qrmi --config "${CONFIG}" \
    || die "environment or QRMI check failed (see above)"

echo "=== 2. Circuit ==="
circuit=$(awk '/^circuit_data:/ {print $2}' "${CONFIG}" | tr -d '"')
if [ -f "${circuit}" ]; then
    echo "reusing existing ${circuit} (delete it to rebuild against the current target)"
else
    echo "building ${circuit}"
    "${HL_PYTHON}" -u make_circuit.py "${CONFIG}" || die "make_circuit.py failed"
fi

if [ -n "${CIRCUIT_ONLY}" ]; then
    echo "Circuit-only mode: built the circuit, no shots taken."
    exit 0
fi

echo "=== 3. Sampling ==="
"${HL_PYTHON}" -u sample_circuit.py "${JOB_NAME}"
status=$?

if [ ${status} -eq 0 ]; then
    echo "Completed job ${JOB_NAME}."
else
    echo "Job ${JOB_NAME} failed with exit status ${status}."
fi
exit ${status}
