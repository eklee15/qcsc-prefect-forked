#!/bin/bash
# Wrapper executed by SLURM on the compute node. It selects the Python
# interpreter and runs example.py. Using a wrapper (rather than making
# example.py the executable directly) keeps the interpreter/env choice
# explicit and lets you add `module load` / `source activate` lines here.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Interpreter: override with EXAMPLE_PYTHON, else fall back to python3 on PATH.
PYTHON="${EXAMPLE_PYTHON:-python3}"

exec "$PYTHON" "$SCRIPT_DIR/example.py" "$@"
