# Prefect BitCount Demo (RPI / Slurm)

This example provides two execution styles for the RPI (AiMOS) Slurm cluster:

- `flow_tutorial_style_rpi.py`: legacy `counter.get(bitstrings)` style with optional HPC profile override
- `flow_optimized.py`: block-driven execution (works on Miyabi, Fugaku, and Slurm)

## Files

Paths are relative to the repository root.

- `examples/prefect_bitcount_demo_rpi/create_blocks.py`
- `examples/prefect_bitcount_demo_rpi/bitcount_blocks.rpi.example.toml`
- `examples/prefect_bitcount_demo_rpi/flow_tutorial_style_rpi.py`
- `examples/prefect_bitcount_demo_rpi/flow_optimized.py`
- `examples/prefect_bitcount_demo_rpi/quantum_sampling.py`
- `examples/prefect_bitcount_demo_rpi/get_counts_integration.py`
- `examples/prefect_bitcount_demo_rpi/build_on_rpi.sh`

For a local Slurm walkthrough using `slurm-docker-cluster`, see
[`docs/tutorials/create_qcsc_workflow_for_local_slurm.md`](../../docs/tutorials/create_qcsc_workflow_for_local_slurm.md).

## Run on the RPI cluster (Slurm)

The RPI (AiMOS) cluster uses Slurm, with the QRMI spank plugin providing QPU
reservations through `#SBATCH --qpu=<name>`.

### Prerequisites

- A Slurm login/frontend node where `sbatch`, `sacct`, and `sinfo` are on `PATH`
- The QRMI spank plugin configured in `/etc/slurm/plugstack.conf`
- An MPI C++ compiler (`mpicxx`) available, e.g. OpenMPI
- A reachable Prefect API (Prefect Cloud or `prefect server start`)

Register the block types once:

```bash
cd /path/to/qcsc-prefect
prefect block register -m qcsc_prefect_blocks.common.blocks
```

For `--quantum-source real-device`, the sampling step needs `prefect-qiskit`,
which is not part of `uv.lock`:

```bash
uv pip install prefect-qiskit
```

### 1. Build the MPI binaries

```bash
cd /path/to/qcsc-prefect
./examples/prefect_bitcount_demo_rpi/build_on_rpi.sh
```

This probes for `mpicxx`/`mpic++` and writes `bin/get_counts_hist` and
`bin/get_counts_json` next to the example.

### 2. Create the Slurm blocks

```bash
cp examples/prefect_bitcount_demo_rpi/bitcount_blocks.rpi.example.toml \
   examples/prefect_bitcount_demo_rpi/bitcount_blocks.rpi.toml
```

Edit `bitcount_blocks.rpi.toml` and set `work_dir` to an absolute path on a
filesystem shared with the compute nodes, e.g. `/path/to/qcsc-prefect/work/prefect_bitcount_rpi`.

```bash
python3 examples/prefect_bitcount_demo_rpi/create_blocks.py \
  --config examples/prefect_bitcount_demo_rpi/bitcount_blocks.rpi.toml
```

On Slurm, `project` is the account (`#SBATCH --account`) and `queue` is the
partition (`#SBATCH --partition`). Adjust both for your cluster; the example
config uses account `root` and partition `normal`. You can also supply them
through the environment (`SLURM_ACCOUNT`, `SLURM_PARTITION`) or the
`--project` / `--queue` flags.

This creates:

- `cmd-bitcount-hist`
- `exec-bitcount-rpi`
- `hpc-rpi-bitcount`
- `rpi-bitcount-options`

### 3. Run the tutorial-style flow (recommended starting point)

Create the legacy tutorial-style assets against the Slurm config:

```bash
python3 examples/prefect_bitcount_demo_rpi/create_blocks.py \
  --config examples/prefect_bitcount_demo_rpi/bitcount_blocks.rpi.toml \
  --hpc-target slurm \
  --create-legacy-tutorial-assets
```

That creates these backward-compatible names:

- BitCounter block: `rpi-tutorial`
- Prefect Variable: `rpi-tutorial`

Without IBM Quantum Runtime (useful to test just the Slurm/MPI path):

```bash
python3 examples/prefect_bitcount_demo_rpi/flow_tutorial_style_rpi.py \
  --quantum-source random \
  --random-seed 24
```

Against a real device:

```bash
python3 examples/prefect_bitcount_demo_rpi/flow_tutorial_style_rpi.py \
  --quantum-source real-device
```

You can override only the `HPCProfileBlock` at runtime when the stored
execution profile is already compatible with the target:

```bash
python3 examples/prefect_bitcount_demo_rpi/flow_tutorial_style_rpi.py \
  --bitcounter-block rpi-tutorial \
  --options-variable rpi-tutorial \
  --quantum-source random \
  --random-seed 24 \
  --hpc-profile-block-override hpc-rpi-bitcount
```

`flow_tutorial_style_rpi.py` is the legacy-compatible path, so it is not the
recommended demo route for backend switching — use `flow_optimized.py` for that.

### 4. Run the block-driven optimized flow

Without IBM Quantum Runtime (useful to test just the Slurm/MPI path):

```bash
python3 examples/prefect_bitcount_demo_rpi/flow_optimized.py \
  --quantum-source random \
  --random-seed 24 \
  --command-block cmd-bitcount-hist \
  --execution-profile-block exec-bitcount-rpi \
  --hpc-profile-block hpc-rpi-bitcount \
  --options-variable rpi-bitcount-options
```

Against a real device:

```bash
python3 examples/prefect_bitcount_demo_rpi/flow_optimized.py \
  --quantum-source real-device \
  --runtime-block ibm-runner \
  --command-block cmd-bitcount-hist \
  --execution-profile-block exec-bitcount-rpi \
  --hpc-profile-block hpc-rpi-bitcount \
  --options-variable rpi-bitcount-options
```

Result shape:

```
{'mode': 'optimized', 'job_id': '53', 'shots': 100000,
 'num_unique_bitstrings': 471,
 'work_dir': '.../work/prefect_bitcount_rpi/job_20260914_033858_c2451eee'}
```

Sampling a 10-qubit GHZ state on real hardware, `0000000000` and `1111111111`
dominate, with the remaining counts spread over several hundred noise states.
In `random` mode all 1024 bitstrings appear instead.

### Generated job script

`create_blocks.py` stores `hpc_target="slurm"`, so the executor renders a
`.slurm` script and submits it with `sbatch`:

```bash
#!/bin/bash
#SBATCH --partition=normal
#SBATCH --account=root
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=<work_dir>/output.out
#SBATCH --error=<work_dir>/output.err
#SBATCH --qpu=ibm_kingston
...
srun "${QCSC_PREFECT_EXECUTABLE}"
```

### Choosing `slurm_qpu`

`slurm_qpu` must name a resource in `/etc/slurm/qrmi_config.json` that is
actually usable from the compute nodes. On this cluster the
`qiskit-runtime-service` resources (`ibm_kingston`, `ibm_fez`,
`ibm_marrakesh`) acquire successfully.

The `direct-access` resources (`test_heron`, `test_eagle`) do **not**: their
`QRMI_IBM_QS_*` endpoint is not provisioned, so the plugin logs

```
error: spank_qrmi, Failed to create a QRMI instance, test_heron_QRMI_IBM_QS_ENDPOINT environment variable is not set
error: spank_qrmi, failed to acquire resource: test_heron
error: spank_qrmi, No QPU resource available
```

Note that this failure is **non-fatal**: the plugin is loaded as `optional`, so
the batch step still runs and the job reports `COMPLETED` with exit code 0 even
though no QPU was reserved. Check `output.err` to confirm the reservation
actually succeeded — an empty `output.err` means it did.

Omit `slurm_qpu` entirely if the job does not need a QPU reservation.

### Troubleshooting

**Job runs but `output.err` shows `No QPU resource available`** — see
[Choosing `slurm_qpu`](#choosing-slurm_qpu).

**`No MPI C++ compiler was found`** — load an MPI module (`module load openmpi`)
before running `build_on_rpi.sh`.

**Executable preflight failure (exit 127)** — the job script checks the
executable before launching. Run `build_on_rpi.sh`, and make sure `bin/` is on a
filesystem visible to the compute nodes.

**Job stuck in `PENDING`** — check partition capacity with
`sinfo -p normal -o "%P %a %D %t"`.
