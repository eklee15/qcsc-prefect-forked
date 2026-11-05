"""SBD solver block backed by qcsc-prefect block execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np
from prefect import get_run_logger, task
from prefect.blocks.core import Block
from pydantic import Field
from pyscf.tools import fcidump
from qcsc_prefect_executor.from_blocks import run_job_from_blocks


@dataclass(frozen=True)
class SBDResult:
    """Result of an SBD calculation."""

    energy: float
    """The SCI energy."""

    orbital_occupancies: tuple[np.ndarray, np.ndarray]
    """The average orbital occupancies."""

    carryover_bitstrings: np.ndarray
    """The 2D array of bool representations of carryover bitstrings."""

    rdm1: np.ndarray | None = None
    """Spin-summed 1-particle reduced density matrix."""

    rdm2: np.ndarray | None = None
    """Spin-summed 2-particle reduced density matrix."""

    @property
    def sci_state(self):
        raise NotImplementedError("SBD Prefect integration doesn't reconstruct sci_state object.")


def _make_job_work_dir(base_work_dir: Path) -> Path:
    base_work_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    job_dir = base_work_dir / f"job_{timestamp}_{uuid4().hex[:8]}"
    job_dir.mkdir(parents=True, exist_ok=False)
    return job_dir


def _build_solver_args(solver: "SBDSolverJob") -> list[str]:
    args = [
        "--task_comm_size",
        str(solver.task_comm_size),
        "--adet_comm_size",
        str(solver.adet_comm_size),
        "--bdet_comm_size",
        str(solver.bdet_comm_size),
        "--block",
        str(solver.block),
        "--iteration",
        str(solver.iteration),
        "--tolerance",
        str(solver.tolerance),
        "--carryover_ratio",
        str(solver.carryover_ratio),
        "--dump_matrix_form_wf",
        "matrixformwf.txt",
        "--rdm",
        "0",
    ]
    if solver.solver_mode == "gpu":
        args.extend(["--adetfile", "AlphaDets.bin", "--carryoverfile", "carryover.txt"])
    if solver.user_args:
        args.extend(list(solver.user_args))
    return args


def _prep_files(
    *,
    work_dir: Path,
    ci_strings: tuple[np.ndarray, np.ndarray],
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
) -> None:
    logger = get_run_logger()

    # Write PySCF FCI dump file.
    logger.debug("Writing fcidump.txt file.")
    fcidump.from_integrals(
        str(work_dir / "fcidump.txt"),
        one_body_tensor,
        two_body_tensor,
        norb,
        nelec,
    )

    # Write alpha determinant list consumed by SBD binary.
    logger.debug("Writing AlphaDets.bin file.")
    alpha_det = np.asarray(ci_strings[0], dtype=np.int64).reshape(-1)
    if np.any(alpha_det < 0):
        raise ValueError("Alpha determinants must be non-negative integers.")
    max_ci = 1 << norb
    if np.any(alpha_det >= max_ci):
        raise ValueError(f"Alpha determinant is out of range for norb={norb}.")

    bytes_per_config = (norb + 7) // 8
    with (work_dir / "AlphaDets.bin").open("wb") as fp:
        for ci in alpha_det:
            fp.write(int(ci).to_bytes(bytes_per_config, byteorder="big", signed=False))


def _read_files(
    *,
    work_dir: Path,
    norb: int,
) -> SBDResult:
    logger = get_run_logger()

    def _read_occupations(spin: str) -> np.ndarray:
        """Read per-orbital occupations for one spin.

        Prefer the flow-native ``occ_{spin}.txt`` (one value per orbital).
        Fall back to the standalone solver's ``1pRDM_{spin}.txt`` (a full
        1-particle RDM written as ``i j value`` rows), whose diagonal
        (i == j) holds the occupations.
        """
        occ_path = work_dir / f"occ_{spin}.txt"
        if occ_path.exists():
            logger.debug("Reading %s.", occ_path.name)
            return np.atleast_1d(np.loadtxt(occ_path, dtype=np.float64))
        rdm_path = work_dir / f"1pRDM_{spin}.txt"
        if rdm_path.exists():
            logger.debug("occ_%s.txt missing; deriving from %s diagonal.", spin, rdm_path.name)
            occ = np.zeros(norb, dtype=np.float64)
            with rdm_path.open() as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    i, j = int(parts[0]), int(parts[1])
                    if i == j and 0 <= i < norb:
                        occ[i] = float(parts[2])
            return occ
        raise FileNotFoundError(f"{occ_path} not found (and no 1pRDM_{spin}.txt fallback).")

    occa = _read_occupations("a")
    occb = _read_occupations("b")

    logger.debug("Reading carryover.bin file.")
    bytes_per_config = (norb + 7) // 8
    carryover_path = work_dir / "carryover.bin"
    data = (
        np.fromfile(carryover_path, dtype=np.uint8)
        if carryover_path.exists()
        else np.empty(0, dtype=np.uint8)
    )
    if data.size == 0:
        carryover = np.empty((0, norb), dtype=bool)
    else:
        if data.size % bytes_per_config != 0:
            raise ValueError(
                "carryover.bin size is not aligned with expected bytes-per-config; "
                f"norb={norb}, bytes_per_config={bytes_per_config}, raw_size={data.size}"
            )
        carryover = np.unpackbits(data, bitorder="big").reshape(-1, bytes_per_config * 8)[:, :norb]
        carryover = carryover.astype(bool)

    energy_path = work_dir / "davidson_energy.txt"
    if energy_path.exists():
        logger.debug("Reading davidson_energy.txt file.")
        energy = float(np.loadtxt(energy_path).item())
    else:
        # Standalone solver writes energies.txt: one-body, two-body, total.
        energies_path = work_dir / "energies.txt"
        logger.debug("davidson_energy.txt missing; reading total energy from energies.txt.")
        vals = np.atleast_1d(np.loadtxt(energies_path, dtype=np.float64))
        energy = float(vals[-1])

    return SBDResult(
        energy=energy,
        orbital_occupancies=(occa, occb),
        carryover_bitstrings=carryover,
        rdm1=None,
        rdm2=None,
    )


class SBDSolverJob(Block):
    """Prefect block facade for SBD execution through qcsc-prefect blocks."""

    _block_type_name = "SBD Solver Job"
    _block_type_slug = "sbd_solver_job"

    root_dir: str = Field(
        title="Root Directory",
        description="Root directory where per-job work directories are created.",
    )
    command_block_name: str = Field(
        default="cmd-sbd-diag",
        title="Command Block Name",
        description="Prefect CommandBlock document name.",
    )
    execution_profile_block_name: str = Field(
        default="exec-sbd-mpi",
        title="Execution Profile Block Name",
        description="Prefect ExecutionProfileBlock document name.",
    )
    hpc_profile_block_name: str = Field(
        default="hpc-miyabi-sbd",
        title="HPC Profile Block Name",
        description="Prefect HPCProfileBlock document name.",
    )
    script_filename: str = Field(default="sbd_solver.pbs", title="Script Filename")
    metrics_artifact_key: str = Field(default="miyabi-sbd-metrics", title="Metrics Artifact Key")
    timeout_seconds: float = Field(default=7200.0, title="Timeout Seconds")
    user_args: list[str] = Field(default_factory=list, title="Additional User Args")

    task_comm_size: int = Field(
        default=1,
        gt=0,
        title="Task Comm Size",
        description=(
            "Size of task communicator. Controls distribution of Hamiltonian column operations."
        ),
    )
    adet_comm_size: int = Field(
        default=1,
        gt=0,
        title="Adet Comm Size",
        description="Number of alpha-determinant partitions.",
    )
    bdet_comm_size: int = Field(
        default=1,
        gt=0,
        title="Bdet Comm Size",
        description="Number of beta-determinant partitions.",
    )
    block: int = Field(
        default=10,
        gt=0,
        title="Block",
        description="Maximum Davidson subspace size.",
    )
    iteration: int = Field(
        default=2,
        gt=0,
        title="Iteration",
        description="Number of Davidson restarts.",
    )
    tolerance: float = Field(
        default=1e-4,
        gt=0.0,
        title="Tolerance",
        description="Convergence threshold for Davidson residual norm.",
    )
    carryover_ratio: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        title="Carryover Ratio",
        description="Ratio of bitstrings retained as carryover candidates.",
    )
    solver_mode: Literal["cpu", "gpu"] = Field(
        default="cpu",
        title="Solver Mode",
        description="SBD execution mode.",
    )

    async def run(
        self,
        ci_strings: tuple[np.ndarray, np.ndarray],
        one_body_tensor: np.ndarray,
        two_body_tensor: np.ndarray,
        norb: int,
        nelec: tuple[int, int],
    ) -> SBDResult:
        """Run SBD solver job and return parsed outputs."""
        return await _run_sbd_inner(
            solver=self,
            ci_strings=ci_strings,
            one_body_tensor=one_body_tensor,
            two_body_tensor=two_body_tensor,
            norb=norb,
            nelec=nelec,
        )


@task(name="solve_eigenstate")
async def _run_sbd_inner(
    *,
    solver: SBDSolverJob,
    ci_strings: tuple[np.ndarray, np.ndarray],
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
) -> SBDResult:
    base_work_dir = Path(solver.root_dir).expanduser().resolve()
    job_work_dir = _make_job_work_dir(base_work_dir)

    _prep_files(
        work_dir=job_work_dir,
        ci_strings=ci_strings,
        one_body_tensor=one_body_tensor,
        two_body_tensor=two_body_tensor,
        norb=norb,
        nelec=nelec,
    )

    result = await run_job_from_blocks(
        command_block_name=solver.command_block_name,
        execution_profile_block_name=solver.execution_profile_block_name,
        hpc_profile_block_name=solver.hpc_profile_block_name,
        work_dir=job_work_dir,
        script_filename=solver.script_filename,
        user_args=_build_solver_args(solver),
        watch_poll_interval=5.0,
        timeout_seconds=solver.timeout_seconds,
        metrics_artifact_key=solver.metrics_artifact_key,
    )
    if result.exit_status != 0:
        raise RuntimeError(f"SBDSolverJob failed: exit_status={result.exit_status}")

    return _read_files(work_dir=job_work_dir, norb=norb)
