"""Create the Prefect blocks the sampling flow runs from.

Three steps run as three separate Slurm jobs -- the QRMI/environment check,
the circuit build, and the sampling. Each is described by a trio of blocks:

  * CommandBlock          -- what to run (a logical command + an executable_key)
  * ExecutionProfileBlock -- how to run it (nodes, walltime, launcher, env)
  * HPCProfileBlock       -- where to run it (slurm, partition, --qpu, --gres,
                             and the executable_key -> real path mapping)

All three commands share one HPCProfileBlock (same cluster, same QPU) and map
into its executable_map. flow.py loads these blocks and hands them to
qcsc_prefect_executor.run_job_from_blocks, which renders + submits the sbatch.

Run this under an interpreter that has prefect + the qcsc_prefect_* packages
(the rpi_demo conda env). The Slurm jobs themselves run under the compute
interpreter (pyscf-3.11), selected by HL_PYTHON in each ExecutionProfileBlock's
environment -- see WRAPPERS below.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

# Target the running Prefect server (default: the local server on :4223) rather
# than ephemeral mode. Set before importing prefect so its settings pick it up.
# Override PREFECT_API_URL in the environment to point elsewhere.
os.environ.setdefault("PREFECT_API_URL", "http://127.0.0.1:4223/api")

from qcsc_prefect_blocks.common.blocks import (  # noqa: E402
    CommandBlock,
    ExecutionProfileBlock,
    HPCProfileBlock,
)

EXAMPLE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Block document names the flow loads by.
# --------------------------------------------------------------------------- #
HPC_BLOCK = "h14-hpc"

CMD_CHECK_ENV = "h14-cmd-check-env"
CMD_MAKE_CIRCUIT = "h14-cmd-make-circuit"
CMD_SAMPLE = "h14-cmd-sample"

EXEC_CHECK_ENV = "h14-exec-check-env"
EXEC_MAKE_CIRCUIT = "h14-exec-make-circuit"
EXEC_SAMPLE = "h14-exec-sample"

# executable_map keys (CommandBlock.executable_key -> HPCProfileBlock path)
KEY_CHECK_ENV = "check_env_exe"
KEY_MAKE_CIRCUIT = "make_circuit_exe"
KEY_SAMPLE = "sample_exe"

# The wrapper scripts Slurm runs on the node, one per step. Absolute so the
# sbatch preflight can stat them on the compute node.
WRAPPERS = {
    KEY_CHECK_ENV: str(EXAMPLE_DIR / "run_check_env.sh"),
    KEY_MAKE_CIRCUIT: str(EXAMPLE_DIR / "run_make_circuit.sh"),
    KEY_SAMPLE: str(EXAMPLE_DIR / "run_sampling.sh"),
}

# --------------------------------------------------------------------------- #
# Cluster settings (override with env vars).
# --------------------------------------------------------------------------- #
PARTITION = os.getenv("SLURM_PARTITION", "quantum")
QPU = os.getenv("SLURM_QPU", "ibm_rensselaer")
GRES = os.getenv("SLURM_GRES", "gpu:1")
ACCOUNT = os.getenv("SLURM_ACCOUNT", "").strip()  # quantum partition needs none
JOB_NAME = os.getenv("HL_JOB_NAME", "h14_qpu")

# The compute interpreter the Slurm jobs run with (has ffsim/pyscf/qiskit/qrmi).
# NOT the interpreter that runs this script -- that one (rpi_demo) only needs
# prefect + the qcsc packages.
COMPUTE_PYTHON = os.getenv(
    "HL_PYTHON",
    "/gpfs/u/scratch/QCSC/shared/osama/sc_env/venvs/pyscf-3.11/bin/python",
)

# Walltimes: the checks and the build are quick once the QPU is allocated; the
# sampling job can sit waiting on the IBM hardware queue, so give it the max.
WALLTIME_CHECK_ENV = os.getenv("WALLTIME_CHECK_ENV", "00:30:00")
WALLTIME_MAKE_CIRCUIT = os.getenv("WALLTIME_MAKE_CIRCUIT", "00:30:00")
WALLTIME_SAMPLE = os.getenv("WALLTIME_SAMPLE", "06:00:00")


def _env() -> dict[str, str]:
    """Environment exported into every Slurm script before the wrapper runs."""
    return {"HL_PYTHON": COMPUTE_PYTHON, "HL_JOB_NAME": JOB_NAME}


def build_blocks() -> list[tuple[Any, str]]:
    """Construct every (block, document_name) pair, without saving.

    One HPCProfileBlock (the cluster target, the QPU, the --gres directive and
    the executable_key -> wrapper-path map for all steps), plus a CommandBlock +
    ExecutionProfileBlock per step. The command_name must match between the two
    blocks of a step (run_job_from_blocks checks it).
    """
    pairs: list[tuple[Any, str]] = [
        (
            HPCProfileBlock(
                hpc_target="slurm",
                queue_cpu=PARTITION,
                queue_gpu=PARTITION,
                project_cpu=ACCOUNT,
                project_gpu=ACCOUNT,
                executable_map=dict(WRAPPERS),
                slurm_qpu=QPU,
                slurm_extra_directives=[f"--gres={GRES}"],
            ),
            HPC_BLOCK,
        ),
    ]

    steps = [
        (CMD_CHECK_ENV, EXEC_CHECK_ENV, "check_env", KEY_CHECK_ENV,
         "QRMI / environment preflight (check_env.py --qrmi)", WALLTIME_CHECK_ENV),
        (CMD_MAKE_CIRCUIT, EXEC_MAKE_CIRCUIT, "make_circuit", KEY_MAKE_CIRCUIT,
         "Build + transpile the LUCJ circuit (make_circuit.py)", WALLTIME_MAKE_CIRCUIT),
        (CMD_SAMPLE, EXEC_SAMPLE, "sample", KEY_SAMPLE,
         "Sample the circuit on the QPU (sample_circuit.py)", WALLTIME_SAMPLE),
    ]
    for cmd_name, exec_name, command, key, description, walltime in steps:
        pairs.append((
            CommandBlock(
                command_name=command,
                executable_key=key,
                description=description,
                default_args=[],
            ),
            cmd_name,
        ))
        pairs.append((
            ExecutionProfileBlock(
                profile_name=command,
                command_name=command,
                resource_class="gpu",  # selects queue_gpu/project_gpu
                num_nodes=1,
                mpiprocs=1,
                ompthreads=1,
                walltime=walltime,
                launcher="single",  # run the wrapper directly, no srun/mpirun
                environments=_env(),
            ),
            exec_name,
        ))
    return pairs


def _names() -> dict[str, Any]:
    return {
        "hpc": HPC_BLOCK,
        "check_env": {"command": CMD_CHECK_ENV, "execution": EXEC_CHECK_ENV},
        "make_circuit": {"command": CMD_MAKE_CIRCUIT, "execution": EXEC_MAKE_CIRCUIT},
        "sample": {"command": CMD_SAMPLE, "execution": EXEC_SAMPLE},
    }


def _report() -> None:
    print("Saved blocks:")
    print(f"  HPC         : {HPC_BLOCK}  (slurm, partition={PARTITION}, "
          f"--qpu={QPU}, --gres={GRES})")
    print(f"  check_env   : {CMD_CHECK_ENV} / {EXEC_CHECK_ENV}")
    print(f"  make_circuit: {CMD_MAKE_CIRCUIT} / {EXEC_MAKE_CIRCUIT}")
    print(f"  sample      : {CMD_SAMPLE} / {EXEC_SAMPLE}")
    print(f"  compute python (HL_PYTHON): {COMPUTE_PYTHON}")


def main() -> dict[str, Any]:
    """Create/overwrite all blocks synchronously (standalone / no running loop)."""
    for block, name in build_blocks():
        block.save(name, overwrite=True)
    _report()
    return _names()


async def asave_all() -> dict[str, Any]:
    """Create/overwrite all blocks from inside a running event loop (the flow).

    Block.save() returns a coroutine when a loop is already running and a value
    otherwise, so await only when needed.
    """
    for block, name in build_blocks():
        result = block.save(name, overwrite=True)
        if inspect.isawaitable(result):
            await result
    _report()
    return _names()


if __name__ == "__main__":
    main()
