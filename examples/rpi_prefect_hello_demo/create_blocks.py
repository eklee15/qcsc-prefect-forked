from __future__ import annotations

import os
from pathlib import Path

from qcsc_prefect_blocks.common.blocks import CommandBlock, ExecutionProfileBlock, HPCProfileBlock


def _resolve_demo_executable() -> str:
    env_path = os.getenv("RPI_DEMO_EXECUTABLE", "").strip()
    if env_path:
        return str(Path(env_path).expanduser().resolve())
    return str((Path(__file__).resolve().parent / "hello_demo.sh").resolve())


def main() -> None:
    account = os.getenv("SLURM_ACCOUNT", "").strip()
    if not account:
        raise RuntimeError(
            "Set SLURM_ACCOUNT (your account/project on the cluster) before running create_blocks.py."
        )

    partition = os.getenv("SLURM_PARTITION", "default").strip()
    executable = _resolve_demo_executable()

    CommandBlock(
        command_name="hello-demo",
        executable_key="hello_demo",
        description="Simple hello script for RPI + Prefect block demo",
        default_args=[],
    ).save("cmd-rpi-hello-demo", overwrite=True)

    ExecutionProfileBlock(
        profile_name="hello-single-node",
        command_name="hello-demo",
        resource_class="cpu",
        #resource_class="local",
        num_nodes=1,
        mpiprocs=1,
        ompthreads=1,
        walltime="00:15:00",
        launcher="single",
        modules=[],
        environments={},
    ).save("exec-rpi-hello-single", overwrite=True)

    HPCProfileBlock(
        hpc_target="slurm",
        queue_cpu=partition,
        queue_gpu=partition,
        project_cpu=account,
        project_gpu=account,
        executable_map={"hello_demo": executable},
    ).save("hpc-rpi", overwrite=True)

    print("Saved blocks: cmd-rpi-hello-demo, exec-rpi-hello-single, hpc-rpi")
    print(f"  SLURM_ACCOUNT={account}")
    print(f"  SLURM_PARTITION={partition}")
    print(f"  executable={executable}")


if __name__ == "__main__":
    main()
