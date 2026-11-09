#!/usr/bin/env python3
"""Check that a Python environment can run this project's Python code.

Covers build_hamiltonian.py and make_circuit.py, and the Python side of the SQD
loop (launch_sqd.py, sbd_solver.py, data_handlers.py). Written for the project
environment on AiMOS (env/aimos/), where there is no jaxlib for ppc64le and
ffsim and qiskit-addon-sqd are patched to import jax only when a jax routine is
actually called. Run it inside the environment you want to check, e.g.
`source env/aimos/env.sh` first; it only reads the repo and writes to a temp dir.
With HL_PYENV=sc_env (Osama's self-built stack, real jax; env.sh exports
SC_ENV) the libraries under $SC_ENV/lib and the gcc-15 runtime are that
environment's own, not foreign.

  1  Interpreter, and the installed version of every package the scripts
     import, next to the version they were tested with. An older version is a
     warning, not a failure: what matters is whether the APIs in 2 exist.
  2  The specific APIs the scripts call. generate_lucj_pass_manager in
     particular is recent, so an older ffsim may lack it.
  3  Import every name the scripts take from ffsim and qiskit-addon-sqd, and
     report whether that pulled in jax. Without jax installed, success here is
     the check that the lazy-jax patches are in place.
  4  (Linux) Which BLAS and C++ runtime numpy actually loaded. env/aimos/env.sh
     keeps the SBD solver's OpenBLAS and gcc-15 libraries off LD_LIBRARY_PATH,
     but a login script may still add them. Any such library loaded from outside
     the Python environment and the system directories is a failure.
  5  Load the repo's mol_data .json.xz. It was written by ffsim 0.0.84 on a
     laptop; an older ffsim may not read it.
  6  (--smoke) Run both circuit scripts end to end on an H4 chain against a
     generic backend: RHF, CCSD, LUCJ, transpile, QPY round trip. Takes seconds
     and needs no IBM credentials. optimize_ucj is used only if jax is present.
  7  (--ibm) Fetch the config's backend_name from IBM Quantum. Needs TOKEN and
     INSTANCE in .env, and network access, which a compute node may not have.
  8  (--qrmi) Find the config's backend_name among the QRMI resources and fetch
     its target. Only inside a Slurm job started with --qpu=<backend_name>, where
     the QRMI plugin supplies the resource and credentials; this is how AiMOS
     reaches the QPU (workflow/3_sample_qpu_sbatch.sh runs it first).
  9  What the memory profiling can use on this node, informational only:
     whether workflow/sbd_rank_time.sh will measure the SBD ranks with GNU time
     or fall back to polling /proc, whether the driver can reset its peak RSS
     (profiling.py), and which cgroup version and memory files
     workflow/mem_monitor.sh will find. Run it on a compute node to know.

Usage:
    python workflow/check_env.py
    python workflow/check_env.py --smoke --ibm
    python workflow/check_env.py --qrmi --config jobs/<job>/config.yaml
    python workflow/check_env.py --all-packages > env-packages.txt
"""

import argparse
import importlib.util
import inspect
import os
import platform
import sys
import tempfile
import traceback
from importlib import import_module, metadata

# Standalone example: scripts, config and data all live in this one directory.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

U = "  "
results = []

# (distribution, import name, needed for, version tested with)
PACKAGES = [
    ("numpy", "numpy", "all scripts", "2.5.2"),
    ("PyYAML", "yaml", "all scripts", "6.0.3"),
    ("scipy", "scipy", "all scripts", None),
    ("pyscf", "pyscf", "build_hamiltonian.py, SQD", "2.14.0"),
    ("ffsim", "ffsim", "circuit scripts", "0.0.84"),
    ("qiskit", "qiskit", "make_circuit.py", "2.5.2"),
    ("qiskit-ibm-runtime", "qiskit_ibm_runtime", "backend_type: ibm only", "0.49.0"),
    ("qrmi", "qrmi", "QPU sampling on AiMOS", "0.24.0"),
    ("matplotlib", "matplotlib", "layout_plot only", None),
    ("jax", "jax", "optimize_ucj: True only (absent on AiMOS)", None),
]
OPTIONAL = {"qiskit-ibm-runtime", "qrmi", "matplotlib", "jax"}

# System library locations; anything else outside the Python environment is
# foreign. (The SBD solver's OpenBLAS / gcc-15 libraries are reachable under
# several paths, e.g. QCSCrsdd/scratch-shared and QCSCzmng/scratch-shared are
# the same directory, so match on location, not on an account name.)
SYSTEM_LIB_DIRS = ("/usr/", "/lib/", "/lib64/")


def record(name, ok, detail=""):
    results.append((name, ok))
    print(f"{U}[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def older_than(installed, tested):
    try:
        from packaging.version import Version
    except ImportError:
        return False
    try:
        return Version(installed) < Version(tested)
    except Exception:
        return False


def check_packages():
    print("1. Interpreter and packages")
    print(f"{U}python      : {sys.executable}")
    print(f"{U}version     : {platform.python_version()} ({platform.machine()})")
    print(f"{U}VIRTUAL_ENV : {os.environ.get('VIRTUAL_ENV', '(none)')}")
    print(f"{U}CONDA_PREFIX: {os.environ.get('CONDA_PREFIX', '(none)')}")
    print()
    print(f"{U}{'package':<20} {'installed':<12} {'tested with':<12} needed for")
    broken = []
    for dist, module, needed_for, tested in PACKAGES:
        try:
            installed = metadata.version(dist)
        except metadata.PackageNotFoundError:
            installed = None
        note = ""
        if installed is None:
            note = "  <-- MISSING" + (" (optional)" if dist in OPTIONAL else "")
        else:
            try:
                import_module(module)
            except Exception as exc:
                installed, note = None, f"  <-- installed but import fails: {exc!r}"
            if installed and tested and older_than(installed, tested):
                note = "  <-- older than tested"
        print(f"{U}{dist:<20} {installed or '-':<12} {tested or '-':<12} {needed_for}{note}")
        if installed is None and dist not in OPTIONAL:
            broken.append((dist, needed_for))
    # Only the closed loop's Prefect server needs sqlite3 (prefect server start,
    # or the temporary one a flow starts without PREFECT_API_URL); a client
    # reporting to workflow/prefect_server.sh does not. sc_env's Python lacks it.
    try:
        import sqlite3  # noqa: F401
        sqlite_note = "available"
    except ImportError:
        sqlite_note = "MISSING: Prefect flows need a running server (workflow/prefect_server.sh)"
        if not os.environ.get("PREFECT_API_URL"):
            sqlite_note += "; PREFECT_API_URL is not set"
    print(f"{U}{'sqlite3 (stdlib)':<20} {sqlite_note}")
    print()
    for dist, needed_for in broken:
        record(f"{dist} importable", False, needed_for)


def has_param(func, name):
    try:
        return name in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def check_apis():
    print("2. APIs the scripts call")

    def probe(name, check):
        try:
            ok = bool(check())
            record(name, ok)
        except Exception as exc:
            record(name, False, repr(exc))

    import ffsim
    probe("ffsim.MolecularData.from_scf(active_space=...)",
          lambda: has_param(ffsim.MolecularData.from_scf, "active_space"))
    probe("ffsim.MolecularData.run_ccsd(store_t1=, store_t2=)",
          lambda: has_param(ffsim.MolecularData.run_ccsd, "store_t1")
          and has_param(ffsim.MolecularData.run_ccsd, "store_t2"))
    probe("ffsim.MolecularData.to_json/from_json(compression=)",
          lambda: has_param(ffsim.MolecularData.to_json, "compression")
          and has_param(ffsim.MolecularData.from_json, "compression"))
    probe("ffsim.MolecularData.to_fcidump", lambda: hasattr(ffsim.MolecularData, "to_fcidump"))
    probe("ffsim.UCJOpSpinBalanced.from_t_amplitudes(optimize=, options=)",
          lambda: has_param(ffsim.UCJOpSpinBalanced.from_t_amplitudes, "optimize")
          and has_param(ffsim.UCJOpSpinBalanced.from_t_amplitudes, "options"))

    def lucj_pass_manager():
        from ffsim.qiskit import generate_lucj_pass_manager
        return all(has_param(generate_lucj_pass_manager, p) for p in
                   ("connectivity", "two_qubit_error_threshold", "readout_error_threshold"))
    probe("ffsim.qiskit.generate_lucj_pass_manager", lucj_pass_manager)
    probe("ffsim.qiskit.PrepareHartreeFockJW / UCJOpSpinBalancedJW",
          lambda: hasattr(ffsim.qiskit, "PrepareHartreeFockJW")
          and hasattr(ffsim.qiskit, "UCJOpSpinBalancedJW"))

    def generic_backend():
        from qiskit.providers.fake_provider import GenericBackendV2
        return has_param(GenericBackendV2, "coupling_map")
    probe("qiskit GenericBackendV2", generic_backend)

    def index_layout():
        from qiskit.transpiler import TranspileLayout
        return hasattr(TranspileLayout, "initial_index_layout")
    probe("qiskit TranspileLayout.initial_index_layout", index_layout)
    print()


def check_imports():
    print("3. Imports the scripts make, and whether they load jax")
    jax_installed = importlib.util.find_spec("jax") is not None
    # Every name launch_sqd.py, sbd_solver.py and data_handlers.py take from
    # qiskit-addon-sqd. fermion.py is the module that imported jax at load time
    # before the patch.
    # Trimmed for this standalone example: only the imports the sampling
    # pipeline (build_hamiltonian -> make_circuit -> sample_circuit) makes.
    # The upstream repo also probes the SQD/SBD solver stack here
    # (qiskit_addon_sqd.*, sbd_solver, data_handlers), which this example does
    # not carry or use, so those checks are removed.
    imports = [
        ("ffsim", ["MolecularData", "UCJOpSpinBalanced"]),
        ("ffsim.qiskit", ["generate_lucj_pass_manager", "PrepareHartreeFockJW"]),
        ("build_hamiltonian", ["ensure_hamiltonian"]),
        ("make_circuit", ["make_circuit"]),
    ]
    for module, names in imports:
        label = f"from {module} import {', '.join(names)}" if names else f"import {module}"
        try:
            mod = import_module(module)
            missing = [name for name in names if not hasattr(mod, name)]
            record(label, not missing, f"missing: {', '.join(missing)}" if missing else "")
        except Exception as exc:
            record(label, False, repr(exc))

    loaded = "jax" in sys.modules
    if jax_installed:
        print(f"{U}jax is installed; imports above loaded it: {loaded}")
    else:
        # Only meaningful without jax: every import above must have succeeded
        # without it, and nothing may have slipped a stand-in into sys.modules.
        record("jax not installed and not loaded", not loaded,
               "" if not loaded else "something registered a 'jax' module")
    print()


def check_shared_libraries():
    print("4. Shared libraries numpy loaded")
    maps = "/proc/self/maps"
    if not os.path.exists(maps):
        print(f"{U}skipped: no {maps} on {platform.system()}")
        print()
        return
    import numpy as np

    a = np.random.default_rng(0).random((64, 64))
    a @ a  # make sure the BLAS backend is actually loaded
    with open(maps) as f:
        paths = sorted({line.split()[-1] for line in f if "/" in line})
    keywords = ("blas", "lapack", "libstdc++", "libgfortran", "libgomp")
    relevant = [p for p in paths if any(k in os.path.basename(p).lower() for k in keywords)]
    for path in relevant:
        print(f"{U}{path}")
    own = [os.path.realpath(sys.prefix) + os.sep]
    if os.environ.get("SC_ENV"):
        # sc_env (exported by env/aimos/env.sh): built against its own lib/ and
        # GCC 15, found through LD_LIBRARY_PATH
        own += [os.path.realpath(d) + os.sep for d in (os.path.join(os.environ["SC_ENV"], "lib"),
                                                       "/gpfs/u/software/dcs-rhel8/gcc/15.1.0")]
    foreign = [p for p in relevant
               if not os.path.realpath(p).startswith(tuple(own)) and not p.startswith(SYSTEM_LIB_DIRS)]
    record("numpy uses this environment's BLAS / C++ runtime", not foreign,
           f"{len(foreign)} loaded from outside {sys.prefix}: check LD_LIBRARY_PATH" if foreign else "")
    print()


def check_repo_mol_data(config_path):
    print("5. Read the repo's molecular data")
    from build_hamiltonian import HAMILTONIAN_DEFAULTS, load_config, mol_data_json_path
    import ffsim

    config = load_config(config_path, HAMILTONIAN_DEFAULTS)
    json_path = mol_data_json_path(config)
    if not json_path.is_file():
        print(f"{U}{json_path} does not exist; build_hamiltonian.py would build it")
        print()
        return
    try:
        mol_data = ffsim.MolecularData.from_json(json_path, compression="lzma")
        has_amplitudes = mol_data.ccsd_t1 is not None and mol_data.ccsd_t2 is not None
        record(f"load {json_path.name}", has_amplitudes,
               f"norb={mol_data.norb}, nelec={mol_data.nelec}"
               + ("" if has_amplitudes else ", but no CCSD amplitudes"))
    except Exception as exc:
        record(f"load {json_path.name}", False, repr(exc))
    print()


def smoke_test(config_path):
    print("6. End-to-end smoke test (H4, generic backend)")
    from build_hamiltonian import HAMILTONIAN_DEFAULTS, ensure_hamiltonian, load_config, load_hamiltonian
    from make_circuit import CIRCUIT_DEFAULTS, make_circuit
    from qiskit import qpy

    workdir = tempfile.mkdtemp(prefix="check_env_")
    print(f"{U}writing to {workdir}")
    # Start from the real config so its keys are exercised, then shrink the
    # problem and keep every output out of the repo.
    config = load_config(config_path, {**HAMILTONIAN_DEFAULTS, **CIRCUIT_DEFAULTS})
    optimize = config["optimize_ucj"] and importlib.util.find_spec("jax") is not None
    config.update(
        n_atoms=4,
        active_space=None,
        mol_data=os.path.join(workdir, "h4.npz"),
        mol_data_json=None,
        fci_dump=os.path.join(workdir, "h4.fcidump"),
        circuit_data=os.path.join(workdir, "h4.qpy"),
        backend_type="generic",
        n_ab_pairs=4,
        optimize_ucj=optimize,
        ucj_maxiter=10,
        layout_plot=None,
    )
    print(f"{U}optimize_ucj = {optimize}")
    try:
        ensure_hamiltonian(config)
        mol_data = load_hamiltonian(config)
        transpiled = make_circuit(mol_data, config, overwrite=False)
        with open(config["circuit_data"], "rb") as fd:
            loaded = qpy.load(fd)[0]
        same = loaded.count_ops() == transpiled.count_ops()
        record("build Hamiltonian, build circuit, QPY round trip", same,
               "" if same else "reloaded circuit differs from the one written")
    except Exception:
        traceback.print_exc()
        record("build Hamiltonian, build circuit, QPY round trip", False, "see traceback above")
    print()


def ibm_test(config_path):
    print("7. Fetch the IBM backend")
    from build_hamiltonian import HAMILTONIAN_DEFAULTS, load_config
    from make_circuit import CIRCUIT_DEFAULTS, get_backend

    config = load_config(config_path, {**HAMILTONIAN_DEFAULTS, **CIRCUIT_DEFAULTS})
    config["backend_type"] = "ibm"
    try:
        backend = get_backend(config)
        record(f"QiskitRuntimeService.backend({config['backend_name']!r})", True,
               f"{backend.num_qubits} qubits")
    except Exception as exc:
        record(f"QiskitRuntimeService.backend({config['backend_name']!r})", False, repr(exc))
    print()


def qrmi_test(config_path):
    print("8. Find the QPU through QRMI")
    from build_hamiltonian import HAMILTONIAN_DEFAULTS, load_config
    from make_circuit import CIRCUIT_DEFAULTS, qrmi_resource

    config = load_config(config_path, {**HAMILTONIAN_DEFAULTS, **CIRCUIT_DEFAULTS})
    name = config["backend_name"]
    try:
        from qrmi.primitives.ibm import get_backend

        resource = qrmi_resource(name)
        # qrmi >= 0.26.3: get_target(resource) is deprecated in favour of
        # get_backend(resource).target.
        target = get_backend(resource).target
        record(f"QRMI resource {name!r} and its target", True,
               f"{target.num_qubits} qubits, metadata {resource.metadata()}")
    except Exception as exc:
        record(f"QRMI resource {name!r} and its target", False, repr(exc))
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=os.path.join(REPO_ROOT, "config_template.yaml"),
                        help="configuration file (default: config_template.yaml at the repo root)")
    parser.add_argument("--smoke", action="store_true",
                        help="run both circuit scripts end to end on an H4 chain (generic backend)")
    parser.add_argument("--ibm", action="store_true",
                        help="also fetch backend_name from IBM Quantum (needs .env and network)")
    parser.add_argument("--qrmi", action="store_true",
                        help="also find backend_name through QRMI (inside a --qpu Slurm job)")
    parser.add_argument("--all-packages", action="store_true",
                        help="only list every installed distribution, like pip freeze")
    args = parser.parse_args()

    if args.all_packages:
        for dist in sorted(metadata.distributions(), key=lambda d: d.metadata["Name"].lower()):
            print(f"{dist.metadata['Name']}=={dist.version}")
        return

    check_packages()
    if any(not ok for _, ok in results):
        print("Required packages are missing; skipping the remaining checks.")
    else:
        check_apis()
        check_imports()
        check_shared_libraries()
        check_repo_mol_data(args.config)
        if args.smoke:
            smoke_test(args.config)
        if args.ibm:
            ibm_test(args.config)
        if args.qrmi:
            qrmi_test(args.config)

    failed = [name for name, ok in results if not ok]
    print(f"{len(results) - len(failed)} passed, {len(failed)} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
