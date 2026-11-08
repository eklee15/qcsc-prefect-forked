#!/usr/bin/env python3
"""Minimal example workload to run on the SLURM cluster via Prefect blocks.

It just prints some environment/argument info and exits 0, so the SLURM job
completes and the Prefect flow can report success. Replace the body with your
real computation.
"""

import os
import socket
import sys
import time


def main() -> int:
    print(f"example.py running on host={socket.gethostname()}")
    print(f"argv={sys.argv[1:]}")
    print(f"SLURM_JOB_ID={os.getenv('SLURM_JOB_ID')}")
    print(f"SLURM_NTASKS={os.getenv('SLURM_NTASKS')}")
    print(f"OMP_NUM_THREADS={os.getenv('OMP_NUM_THREADS')}")

    print("example.py sleeping for 20 seconds...", flush=True)
    time.sleep(20)
    print("example.py finished sleeping")

    print("example.py done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
