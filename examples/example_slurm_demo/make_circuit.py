#!/usr/bin/env python3

# Build the transpiled LUCJ circuit for a 1D hydrogen chain.
# Script version of make_circuit/H_30_make_circuits_nighthawk.ipynb. Parameters
# come from the configuration file (default: config_template.yaml at the repo
# root, the same file the SQD jobs are seeded from).
#
# 1. read the configuration file
# 2. check whether the Hamiltonian files named in it exist
# 3. if not, build them (build_hamiltonian.py)
# 4. load the saved Hamiltonian and build the circuit from it: HF state +
#    spin-balanced LUCJ seeded by the CCSD amplitudes, transpiled for the
#    backend and saved as QPY.

import argparse
import importlib.util
import json
from pathlib import Path

import ffsim
from qiskit.providers import BackendV2, Options

from build_hamiltonian import (
    DEFAULT_CONFIG,
    HAMILTONIAN_DEFAULTS,
    REPO_ROOT,
    check_writable,
    ensure_hamiltonian,
    load_config,
    load_hamiltonian,
    repo_path,
)

# Used for keys the configuration file does not set. The transpiler and the
# generic backend are seeded from the config's random_seed.
CIRCUIT_DEFAULTS = {
    # ibm:     a real device through QiskitRuntimeService. TOKEN and INSTANCE
    #          are read from the .env file at the repo root.
    # generic: a GenericBackendV2 on a grid_num_rows x grid_num_columns square
    #          grid, which needs no credentials (for testing offline).
    # qrmi:    the QPU named backend_name through QRMI, transpiling against the
    #          target it reports. Only inside a Slurm job started with
    #          --qpu=<backend_name> (workflow/3_sample_qpu_sbatch.sh): the QRMI
    #          plugin provides the resource and its credentials.
    "backend_type": "ibm",
    "ibm_channel": "ibm_quantum_platform",
    # Square-grid shape: used to build the generic backend, and to place qubits
    # in layout_plot (Nighthawk is 12 x 10, indexed row-major)
    "grid_num_rows": 12,
    "grid_num_columns": 10,
    "generic_basis_gates": ["cp", "xx_plus_yy", "p", "x", "swap"],
    # Alpha-alpha pairs are the nearest-neighbour orbitals (p, p + 1). The first
    # n_ab_pairs diagonal pairs (p, p) are *requested* as alpha-beta
    # interactions; the layout pass may drop some from the end if the hardware
    # cannot fit them.
    "n_ab_pairs": 10,
    # Iterations for compressing the CCSD amplitudes into the ansatz (optimize_ucj)
    "ucj_maxiter": 100,
    # Plot of the qubit layout on the device; None to skip
    "layout_plot": None,
}


def load_env(filename=".env"):
    """Read KEY=VALUE pairs from the .env file at the repo root into a dict."""
    for directory in [Path.cwd(), *Path.cwd().parents, REPO_ROOT]:
        path = directory / filename
        if path.is_file():
            break
    else:
        raise FileNotFoundError(f"No {filename} found in {Path.cwd()} or any parent")

    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip("\"'")
    return env


def qrmi_resource(name):
    """The QRMI resource for the QPU called `name`.

    QRMI lists only the resources the Slurm job was started with (--qpu=...),
    so outside such a job there are none.
    """
    from qrmi.primitives import QRMIService

    resources = QRMIService().resources()
    names = [resource.metadata().get("backend_name") for resource in resources]
    for resource, resource_name in zip(resources, names):
        if resource_name == name:
            return resource
    raise RuntimeError(
        f"QRMI resource {name!r} is not available (available: {names or 'none'}); "
        f"run inside a Slurm job started with --qpu={name}"
    )


class TargetBackend(BackendV2):
    """A backend that only carries a Target, for transpiling.

    generate_lucj_pass_manager, and the qiskit preset pass manager it builds,
    read nothing from a backend but its target and coupling map. Circuits are
    run through QRMI's SamplerV2, never through this backend.
    """

    def __init__(self, target, name):
        super().__init__(name=name)
        self._target = target

    @property
    def target(self):
        return self._target

    @property
    def max_circuits(self):
        return None

    @classmethod
    def _default_options(cls):
        return Options()

    def run(self, run_input, **options):
        raise NotImplementedError("TargetBackend is for transpiling only; sample with QRMI's SamplerV2")


def get_backend(config):
    if config["backend_type"] == "qrmi":
        from qrmi.primitives.ibm import get_backend as qrmi_get_backend

        resource = qrmi_resource(config["backend_name"])
        # qrmi >= 0.26.3: get_target(resource) is deprecated in favour of
        # get_backend(resource).target.
        return TargetBackend(qrmi_get_backend(resource).target, config["backend_name"])

    if config["backend_type"] == "ibm":
        from qiskit_ibm_runtime import QiskitRuntimeService

        env = load_env()
        service = QiskitRuntimeService(
            channel=config["ibm_channel"],
            token=env["TOKEN"],
            instance=env["INSTANCE"],
        )
        return service.backend(config["backend_name"])

    if config["backend_type"] == "generic":
        from qiskit.providers.fake_provider import GenericBackendV2
        from qiskit.transpiler import CouplingMap

        cmap = CouplingMap.from_grid(
            num_rows=config["grid_num_rows"], num_columns=config["grid_num_columns"]
        )
        return GenericBackendV2(
            num_qubits=cmap.size(),
            basis_gates=config["generic_basis_gates"],
            coupling_map=cmap,
            seed=config["random_seed"],
        )

    raise ValueError(f"Unknown backend_type: {config['backend_type']!r}")


def check_jax_available(config):
    """Stop early if optimize_ucj needs jax and it is not installed.

    On AiMOS (ppc64le) there is no jaxlib, and ffsim is patched to import jax
    only when a jax-based routine is called. Compressing the CCSD amplitudes is
    one, so without this check the run would fail only after the backend fetch
    and the layout search.
    """
    if config["optimize_ucj"] and importlib.util.find_spec("jax") is None:
        raise RuntimeError(
            "optimize_ucj: True needs jax, which is not installed in this "
            "environment; set optimize_ucj: False"
        )


def grid_coords(num_rows, num_columns):
    """[row, col] coordinates for a square lattice indexed row-major.

    Matches CouplingMap.from_grid(num_rows, num_columns), where qubit q
    sits at (q // num_columns, q % num_columns).
    """
    return [
        [row + 1, col + 1]
        for row in range(num_rows)
        for col in range(num_columns)
    ]


def make_circuit(mol_data, config, overwrite):
    from qiskit import qpy
    from qiskit.circuit import QuantumCircuit, QuantumRegister
    from ffsim.qiskit import generate_lucj_pass_manager

    circuit_path = repo_path(config["circuit_data"])
    check_writable(circuit_path, overwrite)
    check_jax_available(config)

    norb = mol_data.norb
    nelec = mol_data.nelec
    if mol_data.ccsd_t1 is None or mol_data.ccsd_t2 is None:
        raise ValueError("Molecular data has no CCSD amplitudes to seed the ansatz")

    backend = get_backend(config)
    print(f"Backend: {backend.name} ({backend.num_qubits} qubits)")

    # Define interactions
    pairs_aa = [(p, p + 1) for p in range(norb - 1)]
    # Every orbital on a square lattice; every ab_pair_stride-th (4 for heavy-hex) otherwise
    pairs_ab = [(p, p) for p in range(0, norb, config.get("ab_pair_stride", 1))][: config["n_ab_pairs"]]

    # The pass manager picks a layout and returns the alpha-beta pairs that fit
    # on it, which may be fewer than requested.
    pass_manager, pairs_ab = generate_lucj_pass_manager(
        backend=backend,
        norb=norb,
        connectivity=config["connectivity"],
        interaction_pairs=(pairs_aa, pairs_ab),
        two_qubit_error_threshold=config["two_qubit_error_threshold"],
        readout_error_threshold=config["readout_error_threshold"],
        optimization_level=config["optimization_level"],
        seed_transpiler=config["random_seed"],
    )
    print(f"Alpha-beta pairs kept: {len(pairs_ab)} of {config['n_ab_pairs']} requested")

    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        mol_data.ccsd_t2,
        t1=mol_data.ccsd_t1,
        n_reps=config["n_reps"],
        interaction_pairs=(pairs_aa, pairs_ab),
        optimize=config["optimize_ucj"],
        options=dict(maxiter=config["ucj_maxiter"]),
    )

    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)
    circuit.measure_all()

    # Transpile the circuit with configured pass manager
    transpiled = pass_manager.run(circuit)

    ops = dict(transpiled.count_ops())
    depth_2q = transpiled.depth(lambda x: x.operation.num_qubits == 2)
    print(f"Transpiled gate counts: {ops}")
    print(f"Two-qubit depth: {depth_2q}")

    with open(circuit_path, "wb") as fd:
        qpy.dump(transpiled, fd)
    print(f"Saved circuit to {circuit_path}")

    # Record what the circuit was actually built from, since pairs_ab and the
    # layout are chosen by the transpiler rather than fixed by the config.
    summary = {
        "backend": backend.name,
        "norb": norb,
        "nelec": list(nelec),
        "n_reps": config["n_reps"],
        "pairs_aa": pairs_aa,
        "pairs_ab": pairs_ab,
        "initial_layout": transpiled.layout.initial_index_layout(filter_ancillas=True),
        "count_ops": ops,
        "two_qubit_depth": depth_2q,
    }
    summary_path = circuit_path.with_suffix(".json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved circuit summary to {summary_path}")

    if config["layout_plot"]:
        from qiskit.visualization import plot_circuit_layout

        coords = None
        if config["connectivity"] == "square":
            coords = grid_coords(config["grid_num_rows"], config["grid_num_columns"])
        fig = plot_circuit_layout(
            circuit=transpiled,
            backend=backend,
            view="virtual",
            qubit_coordinates=coords,
        )
        plot_path = repo_path(config["layout_plot"])
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=150)
        print(f"Saved layout plot to {plot_path}")

    return transpiled


def main():
    parser = argparse.ArgumentParser(
        description="Build the transpiled LUCJ circuit for a 1D hydrogen chain."
    )
    parser.add_argument("config", nargs="?", default=DEFAULT_CONFIG,
                        help="configuration file (default: config_template.yaml at the repo root)")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing circuit file (the Hamiltonian is never rebuilt "
                             "here; use build_hamiltonian.py --overwrite for that)")
    args = parser.parse_args()

    # 1. Read the configuration file
    config = load_config(args.config, {**HAMILTONIAN_DEFAULTS, **CIRCUIT_DEFAULTS})

    # Fail before the Hamiltonian build rather than after it
    check_writable(repo_path(config["circuit_data"]), args.overwrite)
    check_jax_available(config)

    # 2-3. Build the Hamiltonian if its files do not exist yet
    ensure_hamiltonian(config)

    # 4. Build the circuit from the saved Hamiltonian
    mol_data = load_hamiltonian(config)
    make_circuit(mol_data, config, args.overwrite)


if __name__ == "__main__":
    main()
