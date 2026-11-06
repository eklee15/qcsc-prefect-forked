# Hydrogen-lattice (HL) variant of lucj.py.
#
# Instead of building the LUCJ ansatz from CCSD t2 amplitudes (see lucj.py),
# this module loads a pre-transpiled circuit from a QPY file and hands it to
# the sampling path of the flow. The QPY circuit is produced outside this
# repo (hydrogen-lattice/python_scripts/make_circuit.py) and consumed the same
# way as in that project's sample_circuit.py:
#
#     from qiskit import qpy
#     with open(circuit_path, "rb") as fd:
#         circuit = qpy.load(fd)[0]
#
# The circuit is already an ISA circuit for a specific backend (e.g. 120
# physical qubits for ibm_rensselaer) with a single classical ``meas``
# register, so unlike create_lucj_circuit it must NOT be transpiled again.

import os
from pathlib import Path

import numpy as np
from prefect import get_run_logger, task
from qcsc_workflow_utility.chem import (
    ElectronicProperties,
    NpStrict1DArrayF64,
    NpStrict2DArrayF64,
)
from qiskit import qpy
from qiskit.circuit import QuantumCircuit

MODULE_RNG = np.random.default_rng(seed=4520)

# Default pre-transpiled circuit. Override with SBD_HL_CIRCUIT or by passing
# ``circuit_path`` to create_lucj_circuit.
DEFAULT_CIRCUIT_PATH = (
    Path(__file__).resolve().parents[1] / "circuit" / "h14_sto-6g_14e14o_d-0.95252.qpy"
)


def _resolve_circuit_path(circuit_path: str | os.PathLike | None) -> Path:
    """Pick the QPY path: explicit arg, else $SBD_HL_CIRCUIT, else the default."""
    if circuit_path is not None:
        return Path(circuit_path).expanduser()
    env = os.getenv("SBD_HL_CIRCUIT", "").strip()
    if env:
        return Path(env).expanduser()
    return DEFAULT_CIRCUIT_PATH


@task
def initialize_ucj_parameters(
    elec_props: ElectronicProperties,
    aa_indices: list[tuple[int, int]],
    ab_indices: list[tuple[int, int]],
    num_walkers: int,
    randomization_factor: float,
    n_lucj_layers: int,
) -> NpStrict2DArrayF64:
    """Return one placeholder parameter vector per walker.

    The HL circuit is fixed (loaded from QPY), so there are no ansatz
    parameters to optimize. This keeps the same signature as
    ``lucj.initialize_ucj_parameters`` so the flow loop is unchanged; the
    returned values are unused by ``create_lucj_circuit`` below.
    """
    return [np.zeros(0, dtype=np.float64) for _ in range(num_walkers)]


@task
def create_lucj_circuit(
    ucj_parameter: NpStrict1DArrayF64,
    elec_props: ElectronicProperties,
    aa_indices: list[tuple[int, int]],
    ab_indices: list[tuple[int, int]],
    n_lucj_layers: int,
    use_reset_mitigation: bool,
    circuit_path: str | os.PathLike | None = None,
) -> QuantumCircuit:
    """Load the pre-transpiled HL circuit from a QPY file.

    Drop-in replacement for ``lucj.create_lucj_circuit``: it accepts the same
    arguments (which are ignored, since the circuit is pre-built) and returns a
    ``QuantumCircuit``. Reset mitigation is not supported here because the
    stored circuit already fixes its classical registers.
    """
    logger = get_run_logger()

    if use_reset_mitigation:
        raise NotImplementedError(
            "use_reset_mitigation is not supported for pre-transpiled HL circuits; "
            "the QPY circuit's classical registers are fixed at build time."
        )

    path = _resolve_circuit_path(circuit_path)
    if not path.exists():
        raise FileNotFoundError(
            f"HL circuit QPY file not found: {path}. "
            "Set SBD_HL_CIRCUIT or pass circuit_path to create_lucj_circuit."
        )

    with path.open("rb") as fd:
        circuits = qpy.load(fd)
    if not circuits:
        raise ValueError(f"No circuits found in QPY file {path}.")
    circuit = circuits[0]

    logger.info(
        "Loaded HL circuit from %s: %s qubits, %s clbits, ops=%s",
        path,
        circuit.num_qubits,
        circuit.num_clbits,
        dict(circuit.count_ops()),
    )
    return circuit
