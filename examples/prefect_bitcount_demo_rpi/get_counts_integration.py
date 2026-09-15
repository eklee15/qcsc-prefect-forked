from __future__ import annotations

import json
from array import array
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from prefect import task
from prefect.blocks.core import Block
from pydantic import Field
from qcsc_prefect_executor.from_blocks import run_job_from_blocks

BITLEN = 10


def _make_job_work_dir(base_work_dir: Path) -> Path:
    base_work_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    job_dir = base_work_dir / f"job_{timestamp}_{uuid4().hex[:8]}"
    job_dir.mkdir(parents=True, exist_ok=False)
    return job_dir


def _read_counts(job_work_dir: Path, bitlen: int) -> dict[str, int]:
    hist_path = job_work_dir / "hist_u64.bin"
    if hist_path.exists():
        hist = array("Q")
        with hist_path.open("rb") as f:
            hist.frombytes(f.read())
        expected = 1 << bitlen
        if len(hist) != expected:
            raise ValueError(f"Unexpected histogram size: expected {expected}, got {len(hist)}")
        return {format(i, f"0{bitlen}b"): int(v) for i, v in enumerate(hist) if v > 0}

    json_path = job_work_dir / "output.json"
    if json_path.exists():
        raw_counts = json.loads(json_path.read_text(encoding="utf-8"))
        return {format(int(k), f"0{bitlen}b"): int(v) for k, v in raw_counts.items()}

    raise FileNotFoundError(f"Neither hist_u64.bin nor output.json was generated in {job_work_dir}")


class BitCounter(Block):
    """
    Facade block for legacy tutorial style.

    `BitCounter.load(...).get(bitstrings)` internally resolves:
    - CommandBlock
    - ExecutionProfileBlock
    - HPCProfileBlock
    """

    _block_type_name = "Bit Counter"
    _block_type_slug = "bit-counter"

    root_dir: str = Field(title="Root Directory")
    command_block_name: str = Field(default="cmd-bitcount-hist", title="Command Block Name")
    execution_profile_block_name: str = Field(
        default="exec-bitcount-mpi", title="Execution Profile Block Name"
    )
    hpc_profile_block_name: str = Field(
        default="hpc-miyabi-bitcount", title="HPC Profile Block Name"
    )
    script_filename: str = Field(default="bitcount_facade.pbs", title="Script Filename")
    metrics_artifact_key: str = Field(
        default="miyabi-bitcount-facade-metrics", title="Metrics Artifact Key"
    )
    bitlen: int = Field(default=BITLEN, gt=0, title="Bit Length")
    user_args: list[str] = Field(default_factory=list, title="Command Args")

    async def get(
        self,
        bitstrings: list[str],
    ) -> dict[str, int]:
        return await _get_inner(self, bitstrings)


@task(name="get_counts_mpi")
async def _get_inner(
    counter: BitCounter,
    bitstrings: list[str],
) -> dict[str, int]:
    base_work_dir = Path(counter.root_dir).expanduser().resolve()
    job_work_dir = _make_job_work_dir(base_work_dir)

    u32_values = array("I", (int(bits, 2) for bits in bitstrings))
    with (job_work_dir / "input.bin").open("wb") as f:
        u32_values.tofile(f)

    result = await run_job_from_blocks(
        command_block_name=counter.command_block_name,
        execution_profile_block_name=counter.execution_profile_block_name,
        hpc_profile_block_name=counter.hpc_profile_block_name,
        work_dir=job_work_dir,
        script_filename=counter.script_filename,
        user_args=list(counter.user_args),
        watch_poll_interval=5.0,
        timeout_seconds=1800,
        metrics_artifact_key=counter.metrics_artifact_key,
    )
    if result.exit_status != 0:
        raise RuntimeError(f"BitCounter job failed: exit_status={result.exit_status}")

    return _read_counts(job_work_dir, counter.bitlen)
