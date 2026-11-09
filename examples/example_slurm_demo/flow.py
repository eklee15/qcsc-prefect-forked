#!/usr/bin/env python3
"""Self-contained Prefect flow: submit the QPU sampler to Slurm.

This example is standalone -- it does NOT need the hydrogen-lattice repo. Every
script the job runs (check_env.py, build_hamiltonian.py, make_circuit.py,
sample_circuit.py), the Slurm runner (run_sampling.sh), the job config
(jobs/<JOB_NAME>/config.yaml) and the data (mol_data/, circuit/) all live in
THIS directory, copied from hydrogen-lattice and trimmed to the sampling path.

This file renders a batch script, submits it with ``sbatch``, waits for the job,
and reports the result -- its only dependency is ``prefect``. The #SBATCH header
it emits:

    #SBATCH --partition=quantum
    #SBATCH --nodes=1
    #SBATCH --time=06:00:00
    #SBATCH --qpu=ibm_rensselaer
    #SBATCH --gres=gpu:1
    #SBATCH --job-name=<JOB_NAME>

The job runs run_sampling.sh, which does check_env -> make_circuit ->
sample_circuit, with the working directory set to this directory.

Run it from this directory:

    python flow.py                      # full sampling (spends QPU shots)
    CIRCUIT_ONLY=1 python flow.py       # circuit-only: build only, no shots

The one thing that cannot be vendored into a directory is the Python
environment: ffsim / pyscf / qiskit / qrmi are large binary packages. Point
run_sampling.sh at an interpreter that has them via HL_PYTHON (it defaults to
the project conda env). Override Slurm settings via env vars (HL_JOB_NAME,
SLURM_WALLTIME, SLURM_QPU, ...).
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from prefect import flow
from prefect.artifacts import create_table_artifact
from prefect.logging import get_run_logger

# --------------------------------------------------------------------------- #
# Configuration -- edit these defaults or override with environment variables.
# --------------------------------------------------------------------------- #
# Everything the job needs (scripts, config, data) lives in THIS directory;
# the example does not depend on the hydrogen-lattice repo.
EXAMPLE_DIR = Path(__file__).resolve().parent
SBATCH_SCRIPT = EXAMPLE_DIR / "run_sampling.sh"

JOB_NAME = os.getenv("HL_JOB_NAME", "h14_qpu")
PARTITION = os.getenv("SLURM_PARTITION", "quantum")
QPU = os.getenv("SLURM_QPU", "ibm_rensselaer")
GRES = os.getenv("SLURM_GRES", "gpu:1")
NODES = int(os.getenv("SLURM_NODES", "1"))
WALLTIME = os.getenv("SLURM_WALLTIME", "06:00:00")  # the quantum partition MaxTime

POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "15"))  # seconds between status checks
MAX_LOG = 10_000  # chars of stdout/stderr to echo into the Prefect log

# Slurm states that mean the job is no longer going to change.
_TERMINAL = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "TIMEOUT",
}


# --------------------------------------------------------------------------- #
# Submission helpers (inlined -- previously the qcsc_prefect_* packages).
# --------------------------------------------------------------------------- #
async def _run(*args: str, cwd: Path | None = None) -> str:
    """Run a command, return decoded stdout, raise with diagnostics on failure."""
    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=str(cwd) if cwd else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out_b, err_b = await proc.communicate()
    out = (out_b or b"").decode(errors="replace")
    err = (err_b or b"").decode(errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(args)} rc={proc.returncode}\n"
            f"stdout:\n{out}\nstderr:\n{err}"
        )
    return out


def _render_script(*, work_dir: Path, circuit_only: bool) -> str:
    """Build the batch script text that wraps run_sampling.sh."""
    directives = [
        f"#SBATCH --partition={PARTITION}",
        f"#SBATCH --nodes={NODES}",
        f"#SBATCH --time={WALLTIME}",
        f"#SBATCH --output={work_dir}/output.out",
        f"#SBATCH --error={work_dir}/output.err",
        f"#SBATCH --qpu={QPU}",
        f"#SBATCH --gres={GRES}",
        f"#SBATCH --job-name={JOB_NAME}",
    ]
    exe = str(SBATCH_SCRIPT)
    circuit_only_env = "export CIRCUIT_ONLY=1\n" if circuit_only else ""
    return (
        "#!/bin/bash\n"
        + "\n".join(directives)
        + f"\n\ncd {work_dir}\n\n"
        + circuit_only_env
        + f'if [ ! -x "{exe}" ]; then\n'
        + f'  echo "executable not found or not executable: {exe}" >&2\n'
        + "  exit 127\n"
        + "fi\n\n"
        + f'"{exe}"\n'
    )


async def _submit(script_path: Path, *, cwd: Path) -> str:
    """Submit with ``sbatch --parsable`` and return the job id."""
    out = (await _run("sbatch", "--parsable", str(script_path), cwd=cwd)).strip()
    job_id = out.split(";", 1)[0].strip()
    if not job_id:
        raise RuntimeError(f"could not parse job id from sbatch output: {out!r}")
    return job_id


async def _wait(job_id: str, logger) -> dict[str, Any]:
    """Poll scontrol (then sacct) until the job reaches a terminal state."""
    while True:
        status = await _query(job_id)
        if status:
            state = status["State"].split()[0].rstrip("+")
            if state in _TERMINAL:
                return status
            logger.info(f"job {job_id}: {state}")
        await asyncio.sleep(POLL_INTERVAL)


async def _query(job_id: str) -> dict[str, Any] | None:
    """Return normalized status for ``job_id`` from scontrol, falling back to sacct."""
    # scontrol works while the job is pending/running and needs no accounting.
    try:
        out = await _run("scontrol", "show", "job", "-o", job_id)
        for line in out.strip().splitlines():
            kv = dict(t.split("=", 1) for t in line.split() if "=" in t)
            if kv.get("JobId") == job_id and kv.get("JobState"):
                return {
                    "JobID": job_id,
                    "State": kv.get("JobState", "UNKNOWN"),
                    "ExitCode": kv.get("ExitCode", "0:0"),
                    "Elapsed": kv.get("RunTime", "0:00"),
                    "NodeList": kv.get("NodeList", ""),
                }
    except Exception:
        pass
    # After the job leaves the controller, sacct has the final record.
    try:
        out = await _run(
            "sacct", "-j", job_id,
            "--format=JobID,State,ExitCode,Elapsed,NodeList",
            "--parsable2", "--noheader",
        )
        for line in out.strip().splitlines():
            f = line.split("|")
            if len(f) >= 5 and f[0].strip() == job_id and f[1].strip():
                return {
                    "JobID": f[0], "State": f[1], "ExitCode": f[2],
                    "Elapsed": f[3], "NodeList": f[4],
                }
    except Exception:
        pass
    return None


def _tail(path: Path) -> str:
    if not path.exists():
        return ""
    text = path.read_text(errors="replace")
    return text if len(text) <= MAX_LOG else text[:MAX_LOG] + f"... (+{len(text) - MAX_LOG} chars)"


# --------------------------------------------------------------------------- #
# The flow.
# --------------------------------------------------------------------------- #
def _reset_prior_samples(logger) -> None:
    """Move any previous sample outputs aside so a new full run can proceed.

    sample_circuit.py refuses to overwrite an existing samples.json / qpu_job.yaml
    (so a prior result is never silently lost). This backs those up into a
    timestamped ``prev_<ts>/`` subfolder rather than deleting them.
    """
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


@flow(name="sample-qpu-slurm")
async def sample_qpu_flow(
    *,
    circuit_only: bool = False,
    reset: bool = False,
) -> dict[str, Any]:
    """Submit the sampler to Slurm, wait for it, and report the result.

    Args:
        circuit_only: build the circuit and stop, taking no shots.
        reset: before submitting, move any previous sample outputs aside so a
            repeat full run is not blocked by sample_circuit.py's overwrite guard.
    """
    logger = get_run_logger()
    work_dir = EXAMPLE_DIR  # the job cd's here; config and data are relative to it

    if not SBATCH_SCRIPT.exists():
        raise FileNotFoundError(f"sampler script not found: {SBATCH_SCRIPT}")

    if reset and not circuit_only:
        _reset_prior_samples(logger)

    script_path = work_dir / "sample_qpu.sbatch"
    script_path.write_text(_render_script(work_dir=work_dir, circuit_only=circuit_only))
    logger.info(f"wrote {script_path}")

    job_id = await _submit(script_path, cwd=work_dir)
    logger.info(f"submitted job {job_id} to partition {PARTITION} (--qpu={QPU})")

    status = await _wait(job_id, logger)

    if logs := _tail(work_dir / "output.out"):
        logger.info(logs)
    if errs := _tail(work_dir / "output.err"):
        logger.error(errs)

    exit_code = status.get("ExitCode", "-1:0").split(":", 1)[0]
    exit_status = int(exit_code) if exit_code.isdigit() else -1
    state = status.get("State", "").split()[0].rstrip("+")

    await create_table_artifact(
        key="sample-qpu-slurm-metrics",
        table=[list(status.keys()), list(status.values())],
    )

    result = {
        "job_name": JOB_NAME,
        "job_id": job_id,
        "exit_status": exit_status,
        "state": state,
        "work_dir": str(work_dir),
    }
    # Surface non-success loudly: a walltime kill reports ExitCode 0:0 but
    # State TIMEOUT, so State is the signal that actually matters.
    if state != "COMPLETED" or exit_status != 0:
        raise RuntimeError(f"slurm job did not complete cleanly: {result}")

    logger.info(f"done: {result}")
    return result


if __name__ == "__main__":
    circuit_only = os.getenv("CIRCUIT_ONLY", "").strip().lower() in {"1", "true", "yes"}
    print(asyncio.run(sample_qpu_flow(circuit_only=circuit_only)))
