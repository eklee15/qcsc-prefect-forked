# example_slurm_demo — sample a quantum circuit on a QPU via Prefect + Slurm

A **self-contained** Prefect example: it submits a Slurm job that samples a
transpiled LUCJ circuit for an H14 hydrogen chain on the `ibm_rensselaer` QPU
(through the QRMI Slurm plugin), waits for the job, and reports the result.

Everything the job needs lives in **this one directory** — it does **not**
require the `hydrogen-lattice` repo to be checked out:

```
example_slurm_demo/
├── flow.py                 # Prefect flow: render sbatch → submit → poll → report
├── run_sampling.sh         # the job body Slurm runs: check_env → make_circuit → sample_circuit
├── check_env.py            # environment / QRMI preflight (trimmed to the sampling path)
├── build_hamiltonian.py    # H-chain Hamiltonian (RHF/CCSD via pyscf)
├── make_circuit.py         # build + transpile the LUCJ circuit
├── sample_circuit.py       # sample the circuit on the QPU, write counts
├── jobs/h14_qpu/config.yaml # the job config (geometry, backend, shots, paths)
├── mol_data/               # pre-built Hamiltonian (.json.xz / .npz / .fcidump)
└── circuit/                # pre-built transpiled circuit (.qpy) + summary (.json)
```

## Requirements

The directory vendors all the **code and data**, but it cannot vendor the
**Python environment** — `ffsim`, `pyscf`, `qiskit`, `qrmi` and `prefect` are
large binary packages. You need an interpreter that has them. On AiMOS the
project conda env already does:

```
/gpfs/u/scratch/QCSC/shared/ek/rpi_qcsc_demo/hl/hydrogen-lattice/.envs/envs/aimos/bin/python
```

`flow.py` uses that interpreter by default (via `HL_PYTHON` in
`run_sampling.sh`); point `HL_PYTHON` elsewhere to use a different env.

The flow also needs the QPU: it submits to the `quantum` partition with
`--qpu=ibm_rensselaer`, and the QRMI Slurm plugin supplies the QPU resource and
credentials to the job — no token needed.

## Run it

Copy-paste from this directory. `PY` is the interpreter that has the packages;
the same one both submits the flow and runs inside the Slurm job.

```bash
cd /gpfs/u/scratch/QCSC/shared/ek/qcsc-prefect-forked/examples/example_slurm_demo
PY=/gpfs/u/scratch/QCSC/shared/ek/rpi_qcsc_demo/hl/hydrogen-lattice/.envs/envs/aimos/bin/python

# 1. Dry run — build the circuit only, spend NO QPU shots (recommended first):
CIRCUIT_ONLY=1 "$PY" flow.py

# 2. Full run — sample 100,000 shots on the QPU:
"$PY" flow.py

# Re-running the full flow? Add RESET=1 to move the previous results aside
# first (sample_circuit.py refuses to overwrite an existing samples.json):
RESET=1 "$PY" flow.py
```

The flow prints its result, e.g.:

```
{'job_name': 'h14_qpu', 'job_id': '4910209', 'exit_status': 0,
 'state': 'COMPLETED', 'work_dir': '.../example_slurm_demo'}
```

The counts land in `jobs/h14_qpu/samples/`:

```
jobs/h14_qpu/samples/samples.json      # {bitstring: count}, 100k shots
jobs/h14_qpu/samples/qpu_job.yaml      # job record + timing
jobs/h14_qpu/samples/qpu_task_logs.txt # the QPU's own task logs
```

## Notes

- **Run again?** `sample_circuit.py` refuses to overwrite existing results (so a
  prior run is never silently lost). Pass `RESET=1` and the flow moves the
  previous `samples.json` / `qpu_job.yaml` / `qpu_task_logs.txt` into a
  timestamped `jobs/h14_qpu/samples/prev_<ts>/` folder before submitting.
- **Rebuild the circuit?** Delete `circuit/h14_sto-6g_14e14o_d-0.95252.qpy`;
  `make_circuit.py` rebuilds it (transpiled against the QPU's current target)
  on the next run. The Hamiltonian in `mol_data/` is reused if present.
- **QPU queue time.** The Slurm job can sit in the IBM hardware queue for
  minutes to over an hour depending on load; the flow waits the whole time. The
  walltime is `06:00:00` (the `quantum` partition max) so a long queue does not
  time out. Watch it with `squeue -u "$USER"` or
  `tail -f output.out` (written next to `flow.py`).

## Configuration

Defaults live at the top of `flow.py` and in `jobs/h14_qpu/config.yaml`, and are
overridable by environment variable:

| Env var          | Default                                  | Meaning                         |
|------------------|------------------------------------------|---------------------------------|
| `HL_PYTHON`      | the AiMOS conda env above                | interpreter the job runs with   |
| `HL_JOB_NAME`    | `h14_qpu`                                 | selects `jobs/<name>/config.yaml` |
| `SLURM_PARTITION`| `quantum`                                 | Slurm partition                 |
| `SLURM_QPU`      | `ibm_rensselaer`                          | `#SBATCH --qpu`                 |
| `SLURM_GRES`     | `gpu:1`                                    | `#SBATCH --gres`                |
| `SLURM_WALLTIME` | `06:00:00`                                | `#SBATCH --time`                |
| `CIRCUIT_ONLY`   | *(unset)*                                 | set to `1` to build only, no shots |
| `RESET`          | *(unset)*                                 | set to `1` to move prior results aside before a full run |

For example, a short test with a different job name:

```bash
HL_JOB_NAME=h14_qpu SLURM_WALLTIME=01:00:00 CIRCUIT_ONLY=1 "$PY" flow.py
```
