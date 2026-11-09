#!/usr/bin/env python3
"""Prefect flow: sample an H14 LUCJ circuit on a QPU via Slurm, from blocks.

Four Prefect tasks, in order:

  1. create_blocks  -- save the Command/Execution/HPC Prefect blocks (in-process)
  2. check_env      -- Slurm job: QRMI / environment preflight (check_env.py)
  3. make_circuit   -- Slurm job: build + transpile the circuit (make_circuit.py)
  4. run_sampling   -- Slurm job: sample on the QPU (sample_circuit.py)

Tasks 2-4 are each a SEPARATE Slurm job, submitted through the blocks via
qcsc_prefect_executor.run_job_from_blocks, which renders the sbatch (partition,
--qpu, --gres, walltime, env) from the HPC + Execution + Command blocks and
waits for it.

Two interpreters are in play:
  * This flow + create_blocks run under an orchestration interpreter that has
    prefect + the qcsc_prefect_* packages (the rpi_demo conda env).
  * Each Slurm job runs its wrapper under the compute interpreter that has
    ffsim / pyscf / qiskit / qrmi (pyscf-3.11), selected by HL_PYTHON set in the
    ExecutionProfileBlock environment by create_blocks.py.

Run it from this directory with the orchestration interpreter:

    python flow.py                      # full run: check -> circuit -> sample
    CIRCUIT_ONLY=1 python flow.py       # stop after make_circuit, no shots
    RESET=1 python flow.py              # move prior samples aside first

Override cluster settings with env vars (HL_JOB_NAME, SLURM_PARTITION,
SLURM_QPU, SLURM_GRES, HL_PYTHON, WALLTIME_*). create_blocks.py reads the same
vars when it builds the blocks.
"""

from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from prefect import flow, task
from prefect.logging import get_run_logger
from qcsc_prefect_executor.from_blocks import run_job_from_blocks

import create_blocks as blocks

EXAMPLE_DIR = Path(__file__).resolve().parent
JOB_NAME = os.getenv("HL_JOB_NAME", "h14_qpu")

# Seconds between Slurm status polls while waiting on a job.
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "15"))


def _artifact_key(step: str) -> str:
    """A Prefect artifact key (lowercase letters, digits and dashes only)."""
    raw = f"{JOB_NAME}-{step}-metrics".lower()
    return re.sub(r"[^a-z0-9]+", "-", raw).strip("-")


def _log_dir(step: str) -> Path:
    """Per-step working dir for the rendered sbatch + its output.out/err.

    The wrappers cd back to EXAMPLE_DIR for the actual work; this only keeps
    each step's script and logs separate.
    """
    path = EXAMPLE_DIR / "work" / JOB_NAME / step
    path.mkdir(parents=True, exist_ok=True)
    return path


async def _submit_step(
    *,
    step: str,
    command_block: str,
    execution_block: str,
    logger,
) -> dict[str, Any]:
    """Submit one step as its own Slurm job and wait for it."""
    work_dir = _log_dir(step)
    logger.info(f"[{step}] submitting Slurm job (logs in {work_dir})")
    result = await run_job_from_blocks(
        command_block_name=command_block,
        execution_profile_block_name=execution_block,
        hpc_profile_block_name=blocks.HPC_BLOCK,
        work_dir=work_dir,
        script_filename=f"{step}.slurm",
        watch_poll_interval=POLL_INTERVAL,
        metrics_artifact_key=_artifact_key(step),
    )
    state = str(result.state).split()[0].rstrip("+") if result.state else ""
    status = {
        "step": step,
        "job_id": result.job_id,
        "exit_status": result.exit_status,
        "state": state,
    }
    logger.info(f"[{step}] {status}")
    if state != "COMPLETED" or result.exit_status != 0:
        raise RuntimeError(f"step {step!r} did not complete cleanly: {status}")
    return status


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #
@task(name="create_blocks")
async def create_blocks_task() -> dict[str, Any]:
    """Create/overwrite the Prefect blocks the other tasks load by."""
    logger = get_run_logger()
    result = await blocks.asave_all()
    logger.info(f"created blocks: {result}")
    return result


@task(name="check_env")
async def check_env_task() -> dict[str, Any]:
    """Slurm job: QRMI / environment preflight."""
    return await _submit_step(
        step="check_env",
        command_block=blocks.CMD_CHECK_ENV,
        execution_block=blocks.EXEC_CHECK_ENV,
        logger=get_run_logger(),
    )


@task(name="make_circuit")
async def make_circuit_task() -> dict[str, Any]:
    """Slurm job: build + transpile the LUCJ circuit."""
    return await _submit_step(
        step="make_circuit",
        command_block=blocks.CMD_MAKE_CIRCUIT,
        execution_block=blocks.EXEC_MAKE_CIRCUIT,
        logger=get_run_logger(),
    )


@task(name="run_sampling")
async def run_sampling_task() -> dict[str, Any]:
    """Slurm job: sample the circuit on the QPU."""
    return await _submit_step(
        step="run_sampling",
        command_block=blocks.CMD_SAMPLE,
        execution_block=blocks.EXEC_SAMPLE,
        logger=get_run_logger(),
    )


def _reset_prior_samples(logger) -> None:
    """Move previous sample outputs aside so a repeat full run can proceed."""
    samples_dir = EXAMPLE_DIR / "jobs" / JOB_NAME / "samples"
    existing = [
        samples_dir / name
        for name in ("samples.json", "qpu_job.yaml", "qpu_task_logs.txt")
        if (samples_dir / name).exists()
    ]
    if not existing:
        return
    backup = samples_dir / f"prev_{datetime.now():%Y%m%d_%H%M%S}"
    backup.mkdir(parents=True, exist_ok=True)
    for path in existing:
        path.rename(backup / path.name)
    logger.info(f"reset: moved {len(existing)} prior sample file(s) to {backup}")


# --------------------------------------------------------------------------- #
# Flow
# --------------------------------------------------------------------------- #
@flow(name="sample-qpu-slurm")
async def sample_qpu_flow(
    *,
    circuit_only: bool = False,
    reset: bool = False,
) -> dict[str, Any]:
    """Run the four tasks: create_blocks -> check_env -> make_circuit -> sample.

    Args:
        circuit_only: stop after make_circuit, taking no QPU shots.
        reset: move any previous sample outputs aside before sampling.
    """
    logger = get_run_logger()

    results: dict[str, Any] = {}
    results["create_blocks"] = await create_blocks_task()
    results["check_env"] = await check_env_task()
    results["make_circuit"] = await make_circuit_task()

    if circuit_only:
        logger.info("circuit-only mode: built the circuit, no shots taken.")
        results["run_sampling"] = {"step": "run_sampling", "skipped": True}
        return results

    if reset:
        _reset_prior_samples(logger)
    results["run_sampling"] = await run_sampling_task()

    logger.info(f"done: {results}")
    return results


if __name__ == "__main__":
    circuit_only = os.getenv("CIRCUIT_ONLY", "").strip().lower() in {"1", "true", "yes"}
    reset = os.getenv("RESET", "").strip().lower() in {"1", "true", "yes"}
    print(asyncio.run(sample_qpu_flow(circuit_only=circuit_only, reset=reset)))
