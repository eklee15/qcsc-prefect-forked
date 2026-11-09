#!/usr/bin/env python3

# Build the Hamiltonian for a 1D hydrogen chain and save it under mol_data/.
# Parameters come from the configuration file (default: config_template.yaml
# at the repo root, the same file the SQD jobs are seeded from).
#
# RHF + CCSD (+ CISD) with pyscf, saved as ffsim MolecularData (.json.xz),
# plus the .npz (config key mol_data) and FCIDUMP (fci_dump) read by
# launch_sqd.py and the SBD solver.
#
# The .json.xz is the source of truth: if it exists, nothing is recomputed and
# only a missing .npz or FCIDUMP is regenerated from it. make_circuit.py calls
# ensure_hamiltonian() itself, so running this script first is optional.

import argparse
from pathlib import Path

import numpy as np
import yaml

import ffsim

# Standalone example: everything (scripts, config, data) lives in this one
# directory, so the "repo root" is simply the directory containing this file.
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "config_template.yaml"
AXES = {"x": 0, "y": 1, "z": 2}

# Used for keys the configuration file does not set.
HAMILTONIAN_DEFAULTS = {
    # CISD only adds cisd_energy to the output files as a reference value;
    # nothing downstream needs it.
    "run_cisd": False,
    # Path of the ffsim MolecularData file; null means mol_data with its
    # .npz suffix replaced by .json.xz
    "mol_data_json": None,
}


def repo_path(path):
    """Resolve a path from the config against the repository root."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path, defaults):
    """Read the configuration file, filling in defaults for keys it does not set."""
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    print(f"Loaded configuration from {path}")

    unset = {key: value for key, value in defaults.items() if key not in config}
    if unset:
        print("Not in config, using defaults: "
              + ", ".join(f"{key}={value!r}" for key, value in unset.items()))
    return {**unset, **config}


def check_writable(path, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)


def mol_data_json_path(config):
    if config["mol_data_json"]:
        return repo_path(config["mol_data_json"])
    return repo_path(config["mol_data"]).with_suffix(".json.xz")


def hamiltonian_paths(config):
    """The (.json.xz, .npz, FCIDUMP) output paths named in the config."""
    return (
        mol_data_json_path(config),
        repo_path(config["mol_data"]),
        repo_path(config["fci_dump"]),
    )


def build_mol_data(config):
    """Run RHF, CCSD and optionally CISD for the hydrogen chain."""
    from pyscf import gto, scf

    axis = AXES[config["chain_axis"]]
    atom = []
    for k in range(config["n_atoms"]):
        coords = [0.0, 0.0, 0.0]
        coords[axis] = k * config["bond_length"]
        atom.append(("H", tuple(coords)))

    mol = gto.M(
        atom=atom,
        basis=config["basis"],
        unit="Angstrom",
        spin=config["spin"],
        symmetry=config["symmetry"] or False,
        verbose=3,
    )
    mf = scf.RHF(mol).run()
    if not mf.converged:
        raise RuntimeError("RHF did not converge")

    mol_data = ffsim.MolecularData.from_scf(mf, active_space=config["active_space"])
    mol_data.run_ccsd(store_t1=True, store_t2=True)
    if config["run_cisd"]:
        mol_data.run_cisd()
    return mol_data


def save_npz(mol_data, npz_path):
    """Write the .npz in the format of dump_data.py, as read by launch_sqd.py."""
    hamiltonian = mol_data.hamiltonian
    np.savez(
        npz_path,
        nelec=mol_data.nelec,
        norb=mol_data.norb,
        e_core=hamiltonian.constant,
        h1e=hamiltonian.one_body_tensor,
        h2e=hamiltonian.two_body_tensor,
        hf_energy=mol_data.hf_energy,
        cisd_energy=mol_data.cisd_energy,
        ccsd_energy=mol_data.ccsd_energy,
    )
    print(f"Saved {npz_path}")


def save_fcidump(mol_data, fcidump_path):
    mol_data.to_fcidump(fcidump_path)
    print(f"Saved {fcidump_path}")


def ensure_hamiltonian(config, overwrite=False):
    """Build and save the Hamiltonian, unless the files already exist."""
    json_path, npz_path, fcidump_path = hamiltonian_paths(config)

    if json_path.is_file() and not overwrite:
        missing = [path for path in (npz_path, fcidump_path) if not path.is_file()]
        if not missing:
            print(f"Hamiltonian files already exist: {json_path}, {npz_path}, {fcidump_path}")
            return
        print(f"Regenerating {', '.join(map(str, missing))} from {json_path}")
        mol_data = ffsim.MolecularData.from_json(json_path, compression="lzma")
        for path in missing:
            path.parent.mkdir(parents=True, exist_ok=True)
        if npz_path in missing:
            save_npz(mol_data, npz_path)
        if fcidump_path in missing:
            save_fcidump(mol_data, fcidump_path)
        return

    # Check before the (slow) calculation, so it does not run only to be discarded.
    # Without --overwrite this refuses an .npz/FCIDUMP left without its .json.xz,
    # since nothing says they came from the same geometry as the new build.
    for path in (json_path, npz_path, fcidump_path):
        check_writable(path, overwrite)

    mol_data = build_mol_data(config)
    mol_data.to_json(json_path, compression="lzma")
    print(f"Saved {json_path}")
    save_npz(mol_data, npz_path)
    save_fcidump(mol_data, fcidump_path)


def load_hamiltonian(config):
    """Load the saved ffsim MolecularData (.json.xz)."""
    json_path = mol_data_json_path(config)
    print(f"Loading molecular data from {json_path}")
    mol_data = ffsim.MolecularData.from_json(json_path, compression="lzma")

    print(f"norb = {mol_data.norb}, nelec = {mol_data.nelec}")
    print(f"HF energy:   {mol_data.hf_energy:.10f} Ha")
    print(f"CCSD energy: {mol_data.ccsd_energy:.10f} Ha")
    if mol_data.cisd_energy is not None:
        print(f"CISD energy: {mol_data.cisd_energy:.10f} Ha")
    return mol_data


def main():
    parser = argparse.ArgumentParser(
        description="Build and save the Hamiltonian for a 1D hydrogen chain."
    )
    parser.add_argument("config", nargs="?", default=DEFAULT_CONFIG,
                        help="configuration file (default: config_template.yaml at the repo root)")
    parser.add_argument("--overwrite", action="store_true",
                        help="rebuild the Hamiltonian even if its files exist, replacing them")
    args = parser.parse_args()

    config = load_config(args.config, HAMILTONIAN_DEFAULTS)

    ensure_hamiltonian(config, args.overwrite)
    load_hamiltonian(config)


if __name__ == "__main__":
    main()
