# RPI Prefect Block Hello Demo

This example demonstrates Prefect Blocks for SLURM job orchestration on an RPI cluster. It shows how to define command, execution, and HPC profile blocks, then submit and monitor real SLURM jobs through Prefect flows.

### Prerequisites
* Run on a AiMOS (RPI Cluster) frontend/login node `dcsfen0[1-2]`
* slurm commdnas (eg, `srun` and `sbatch`) are available in PATH
* Prefect API is reachable

### Set up environment (one time setup)
```bash
# Install Conda
./install_conda.sh
conda config --add channels conda-forge
# Create your environment with [custom_env_name]
conda env create -f conda_env.yml -n custom_env_name --force 
```

### 1. Enable your conda environment and start server
```bash
conda activate custom_env_name
prefect server start --background
```

### 2. Register block types
```bash
prefect block register -m qcsc_prefect_blocks.common.blocks
```

### 3. Create demo blocks

```bash
export SLURM_ACCOUNT=qntm
export SLURM_PARTITION=quantum
python ~/barn/qcsc-prefect-forked/examples/rpi_prefect_hello_demo/create_blocks.py
```

Expected output:
```
Saved blocks: cmd-rpi-hello-demo, exec-rpi-hello-single, hpc-rpi
  SLURM_ACCOUNT=qntm
  SLURM_PARTITION=quantum
  executable=/gpfs/u/barn/QNTM/QNTMnkle/qcsc-prefect-forked/examples/rpi_prefect_hello_demo/hello_demo.sh
```

### 4. Run the flow

```bash
cd ~/barn/qcsc-prefect-forked/examples/rpi_prefect_hello_demo/
python -c "import asyncio; from flow import rpi_prefect_block_hello_flow; result = asyncio.run(rpi_prefect_block_hello_flow()); print('Result:', result)"
```

Expected output:
```
Result: {'job_id': '29', 'exit_status': 0, 'state': 'COMPLETED', 'work_dir': '/shared/qcsc-prefect-forked/work/rpi_prefect_block_hello'}
```

## Understanding the Flow

The `rpi_prefect_block_hello_flow()` does the following:

1. **Load Blocks**: Retrieves CommandBlock, ExecutionProfileBlock, and HPCProfileBlock from Prefect state
2. **Validate**: Ensures command and profile are compatible
3. **Resolve Executable**: Maps executable key to actual file path
4. **Build Profile**: Creates ExecutionProfile from block settings
5. **Create Request**: Constructs SlurmJobRequest with partition and account
6. **Submit Job**: Calls `sbatch` with generated SBATCH script
7. **Monitor**: Polls `scontrol` for job status until completion
8. **Return Results**: Returns job ID, exit status, state, and work directory

## Troubleshooting

**Error: "Set SLURM_ACCOUNT before running"**
```bash
export SLURM_ACCOUNT=qntm
python examples/rpi_prefect_hello_demo/create_blocks.py
```

**Job is stuck in PENDING state**
Check partition availability:
```bash
sinfo -p quantum -o "%P %a %D %t"
```

**No output files generated**
Verify work directory permissions:
```bash
ls -la /shared/qcsc-prefect-forked/work/rpi_prefect_block_hello/
```

## Notes

- Job outputs are stored in the work directory (`./work/rpi_prefect_block_hello` by default)
- Each run creates a new `hello_demo.sbatch` script
- General SLURM job states: `PENDING`, `RUNNING`, `COMPLETED`, `FAILED`, `TIMEOUT`, `CANCELLED`
- Default timeout is 600 seconds (10 minutes); adjust in flow parameters if needed
- The demo uses `launcher="single"` for single-node execution; use `mpirun` or `srun` for multi-node
