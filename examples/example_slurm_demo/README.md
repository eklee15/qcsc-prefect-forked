# example_slurm_demo — sample a quantum circuit on a QPU via Prefect + Slurm

A Prefect example that samples a transpiled LUCJ circuit for an H14 hydrogen
chain on the `ibm_rensselaer` QPU (through the QRMI Slurm plugin). The flow is
driven by **Prefect blocks** and runs as **four Prefect tasks**, the last three
each submitting their **own separate Slurm job**:

1. **create_blocks** — save the Command / Execution / HPC Prefect blocks
2. **check_env** — Slurm job: QRMI / environment preflight (`check_env.py`)
3. **make_circuit** — Slurm job: build + transpile the circuit (`make_circuit.py`)
4. **run_sampling** — Slurm job: sample on the QPU (`sample_circuit.py`)

```
example_slurm_demo/
├── flow.py                 # Prefect flow: the four tasks above
├── create_blocks.py        # task 1: build + save the Command/Execution/HPC blocks
├── run_check_env.sh        # task 2 job body: QRMI / environment preflight
├── run_make_circuit.sh     # task 3 job body: build + transpile the circuit
├── run_sampling.sh         # task 4 job body: sample on the QPU
├── check_env.py            # environment / QRMI preflight (trimmed to the sampling path)
├── build_hamiltonian.py    # H-chain Hamiltonian (RHF/CCSD via pyscf)
├── make_circuit.py         # build + transpile the LUCJ circuit
├── sample_circuit.py       # sample the circuit on the QPU, write counts
├── jobs/h14_qpu/config.yaml # the job config (geometry, backend, shots, paths)
├── mol_data/               # pre-built Hamiltonian (.json.xz / .npz / .fcidump)
└── circuit/                # pre-built transpiled circuit (.qpy) + summary (.json)
```

## How it works: blocks + two environments

`create_blocks.py` saves three kinds of Prefect block (from
`qcsc_prefect_blocks.common.blocks`):

- **CommandBlock** — *what* to run: a logical command + an `executable_key`.
- **ExecutionProfileBlock** — *how* to run it: nodes, walltime, launcher, env.
- **HPCProfileBlock** — *where*: `slurm`, partition, `--qpu`, `--gres`, and the
  `executable_key → wrapper-path` map.

`flow.py` loads those blocks and hands each step to
`qcsc_prefect_executor.run_job_from_blocks`, which renders the `sbatch` script
(partition, `--qpu`, `--gres`, walltime, environment) and submits + waits for it.

Two interpreters are involved:

| Role | Environment | Needs |
|------|-------------|-------|
| **Orchestration** — run `flow.py` / `create_blocks.py`, save/load blocks, submit sbatch | **rpi_demo** conda env | `prefect` + the `qcsc_prefect_*` packages |
| **Compute** — each Slurm job's body (ffsim / pyscf / qiskit / qrmi) | **pyscf-3.11** venv | `ffsim`, `pyscf`, `qiskit`, `qrmi` |

The flow runs under rpi_demo. Each Slurm wrapper sources `sc_env/env.sh` to
activate the **pyscf-3.11** compute venv (set by `HL_PYTHON` in the
ExecutionProfileBlock), so the heavy numerics run there — not under rpi_demo.

```
# orchestration interpreter (has prefect + qcsc_prefect_*)
RPI=/gpfs/u/home/QCSC/QCSCnkle/scratch-shared/ek/miniconda3/envs/rpi_demo/bin/python
# compute interpreter (has ffsim/pyscf/qiskit/qrmi); used inside the Slurm jobs
HL_PYTHON=/gpfs/u/scratch/QCSC/shared/osama/sc_env/venvs/pyscf-3.11/bin/python
```

The flow needs the QPU: it submits to the `quantum` partition with
`--qpu=ibm_rensselaer`, and the QRMI Slurm plugin supplies the QPU resource and
credentials to the job — no token needed.

## Run it

From this directory, with the **orchestration** interpreter (rpi_demo). Prefect
needs a place to store blocks; the simplest is ephemeral mode with a fast,
node-local `PREFECT_HOME` (the GPFS home is too slow for the ephemeral server):

```bash
cd /gpfs/u/scratch/QCSC/shared/ek/qcsc-prefect-forked/examples/example_slurm_demo
RPI=/gpfs/u/home/QCSC/QCSCnkle/scratch-shared/ek/miniconda3/envs/rpi_demo/bin/python

export PREFECT_HOME=/tmp/pf_$USER
export PREFECT_SERVER_ALLOW_EPHEMERAL_MODE=true
export PREFECT_API_DATABASE_CONNECTION_URL="sqlite+aiosqlite:///$PREFECT_HOME/prefect.db"
export PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS=60
unset PREFECT_PROFILE PREFECT_API_URL

# 1. Dry run — create_blocks → check_env → make_circuit, spend NO QPU shots:
CIRCUIT_ONLY=1 "$RPI" flow.py

# 2. Full run — also run_sampling: 100,000 shots on the QPU:
"$RPI" flow.py

# Re-running the full flow? Add RESET=1 to move previous results aside first
# (sample_circuit.py refuses to overwrite an existing samples.json):
RESET=1 "$RPI" flow.py
```

The flow returns a dict of per-task results, each with `job_id`, `state` and
`exit_status`, e.g.:

```
{'create_blocks': {...},
 'check_env':    {'step': 'check_env',    'job_id': '4912xxx', 'state': 'COMPLETED', 'exit_status': 0},
 'make_circuit': {'step': 'make_circuit', 'job_id': '4912yyy', 'state': 'COMPLETED', 'exit_status': 0},
 'run_sampling': {'step': 'run_sampling', 'job_id': '4912zzz', 'state': 'COMPLETED', 'exit_status': 0}}
```

Each step's `sbatch` script and `output.out` / `output.err` land in
`work/<job name>/<step>/`. The counts land in `jobs/h14_qpu/samples/`:

```
jobs/h14_qpu/samples/samples.json      # {bitstring: count}, 100k shots
jobs/h14_qpu/samples/qpu_job.yaml      # job record + timing
jobs/h14_qpu/samples/qpu_task_logs.txt # the QPU's own task logs
```

## Notes

- **Run again?** `sample_circuit.py` refuses to overwrite existing results (so a
  prior run is never silently lost). Pass `RESET=1` and the flow moves the
  previous `samples.json` / `qpu_job.yaml` / `qpu_task_logs.txt` into a
  timestamped `jobs/h14_qpu/samples/prev_<ts>/` folder before sampling.
- **Rebuild the circuit?** Delete `circuit/h14_sto-6g_14e14o_d-0.95252.qpy`;
  the make_circuit step rebuilds it (transpiled against the QPU's current
  target). The Hamiltonian in `mol_data/` is reused if present.
- **QPU queue time.** The sampling Slurm job can sit in the IBM hardware queue
  for minutes to over an hour; the flow waits the whole time. Its walltime is
  `06:00:00` (the `quantum` partition max). Watch with `squeue -u "$USER"` or
  `tail -f work/h14_qpu/run_sampling/output.out`.

## Configuration

Cluster settings are read by `create_blocks.py` (so they are baked into the
blocks) and overridable by environment variable:

| Env var             | Default                                   | Meaning                             |
|---------------------|-------------------------------------------|-------------------------------------|
| `HL_PYTHON`         | the pyscf-3.11 venv python                | compute interpreter the jobs run with |
| `HL_JOB_NAME`       | `h14_qpu`                                  | selects `jobs/<name>/config.yaml`   |
| `SLURM_PARTITION`   | `quantum`                                  | Slurm partition                     |
| `SLURM_QPU`         | `ibm_rensselaer`                           | `#SBATCH --qpu`                     |
| `SLURM_GRES`        | `gpu:1`                                     | `#SBATCH --gres` (via extra directive) |
| `SLURM_ACCOUNT`     | *(unset)*                                  | `#SBATCH --account` (quantum needs none) |
| `WALLTIME_CHECK_ENV`| `00:30:00`                                 | walltime for the check_env job      |
| `WALLTIME_MAKE_CIRCUIT`| `00:30:00`                              | walltime for the make_circuit job   |
| `WALLTIME_SAMPLE`   | `06:00:00`                                 | walltime for the sampling job       |
| `CIRCUIT_ONLY`      | *(unset)*                                  | set to `1` to stop after make_circuit, no shots |
| `RESET`             | *(unset)*                                  | set to `1` to move prior results aside before sampling |

`create_blocks.py` can also be run on its own (under rpi_demo) to (re)create the
blocks without running the flow:

```bash
"$RPI" create_blocks.py
```
