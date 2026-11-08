from __future__ import annotations

import os
from pathlib import Path

from qcsc_prefect_blocks.common.blocks import (
    CommandBlock,
    ExecutionProfileBlock,
    HPCProfileBlock,
)

# Block names the flow loads by.
CMD_BLOCK = "cmd-example"
EXEC_BLOCK = "exec-example"
HPC_BLOCK = "hpc-example"


def _resolve_executable() -> str:
    """The script SLURM runs on the node: run_example.sh next to this file.

    Override with EXAMPLE_EXECUTABLE to point elsewhere.
    """
    env_path = os.getenv("EXAMPLE_EXECUTABLE", "").strip()
    if env_path:
        return str(Path(env_path).expanduser().resolve())
    return str((Path(__file__).resolve().parent / "run_example.sh").resolve())


def main() -> None:
    account = os.getenv("SLURM_ACCOUNT", "").strip()
    if not account:
        raise RuntimeError("Set SLURM_ACCOUNT (your cluster account) before running.")
    partition = os.getenv("SLURM_PARTITION", "normal").strip()
    executable = _resolve_executable()

    # 1. CommandBlock: logical command name + the executable_key the HPC block maps.
    CommandBlock(
        command_name="example",
        executable_key="example_exe",
        description="Run example.py on SLURM",
        default_args=[],  # static args appended to every run, e.g. ["--foo", "1"]
    ).save(CMD_BLOCK, overwrite=True)

    # 2. ExecutionProfileBlock: how to launch (resources + environment).
    ExecutionProfileBlock(
        profile_name="example-single",
        command_name="example",
        resource_class="cpu",
        num_nodes=1,
        mpiprocs=1,  # -> #SBATCH --ntasks-per-node
        ompthreads=1,  # -> #SBATCH --cpus-per-task
        walltime="00:05:00",
        launcher="single",  # run the executable directly (no srun/mpirun)
        mpi_options=[],
        modules=[],
        pre_commands=[],
        environments={"OMP_NUM_THREADS": "1"},  # -> export lines before the exec
    ).save(EXEC_BLOCK, overwrite=True)

    # 3. HPCProfileBlock: cluster target + map executable_key -> real path.
    HPCProfileBlock(
        hpc_target="slurm",
        queue_cpu=partition,
        queue_gpu=partition,
        project_cpu=account,
        project_gpu=account,
        executable_map={"example_exe": executable},
    ).save(HPC_BLOCK, overwrite=True)

    print(f"Saved blocks: {CMD_BLOCK}, {EXEC_BLOCK}, {HPC_BLOCK}")
    print(f"  SLURM_ACCOUNT={account}")
    print(f"  SLURM_PARTITION={partition}")
    print(f"  executable={executable}")


if __name__ == "__main__":
    main()
