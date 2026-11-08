from __future__ import annotations

import inspect
from pathlib import Path

from prefect import flow
from qcsc_prefect_adapters.slurm.builder import SlurmJobRequest
from qcsc_prefect_blocks.common.blocks import (
    CommandBlock,
    ExecutionProfileBlock,
    HPCProfileBlock,
)
from qcsc_prefect_core.models.execution_profile import ExecutionProfile
from qcsc_prefect_executor.slurm.run import run_slurm_job


async def _resolve(value):
    if inspect.isawaitable(value):
        return await value
    return value


@flow(name="example-slurm-demo")
async def example_slurm_flow(
    *,
    command_block_name: str = "cmd-example",
    execution_profile_block_name: str = "exec-example",
    hpc_profile_block_name: str = "hpc-example",
    work_dir: str = "./work/example_slurm_demo",
    script_filename: str = "example.sbatch",
    user_args: list[str] | None = None,
):
    cmd = await _resolve(CommandBlock.load(command_block_name))
    profile = await _resolve(ExecutionProfileBlock.load(execution_profile_block_name))
    hpc = await _resolve(HPCProfileBlock.load(hpc_profile_block_name))

    executable = hpc.executable_map.get(cmd.executable_key)
    if not executable:
        raise KeyError(f"Executable key '{cmd.executable_key}' not in HPCProfileBlock.")

    arguments = list(cmd.default_args)
    if user_args:
        arguments.extend(user_args)

    exec_profile = ExecutionProfile(
        command_key=cmd.command_name,
        num_nodes=profile.num_nodes,
        mpiprocs=profile.mpiprocs,
        ompthreads=profile.ompthreads,
        walltime=profile.walltime,
        launcher=profile.launcher,
        mpi_options=list(profile.mpi_options),
        modules=list(profile.modules),
        environments=dict(profile.environments),
        arguments=arguments,
    )

    req = SlurmJobRequest(
        partition=hpc.queue_cpu,
        account=hpc.project_cpu,
        executable=executable,
    )

    result = await run_slurm_job(
        work_dir=Path(work_dir).expanduser().resolve(),
        script_filename=script_filename,
        exec_profile=exec_profile,
        req=req,
        watch_poll_interval=5.0,
        timeout_seconds=300,
        metrics_artifact_key="example-slurm-demo-metrics",
    )
    return {
        "job_id": result.job_id,
        "exit_status": result.exit_status,
        "state": result.state,
        "work_dir": str(Path(work_dir).expanduser().resolve()),
    }


if __name__ == "__main__":
    import asyncio

    print(asyncio.run(example_slurm_flow()))
