#!/usr/bin/env python3

# Sample a job's transpiled circuit on a QPU through QRMI and save the counts
# where launch_sqd.py reads them.
#
#   python python_scripts/sample_circuit.py <jobname>
#
# Reads jobs/<jobname>/config.yaml:
#   backend_name   the QPU, which the Slurm job must have been started with
#                  (--qpu=<backend_name>); QRMI only lists those resources
#   circuit_data   the transpiled circuit (QPY) from make_circuit.py
#   qpu_shots      number of shots (default 100000)
#   dd_enable,     dynamical decoupling on idle qubits (default off), with the
#   dd_sequence_type  Runtime sequence name, e.g. "XpXm" (default "XX")
#   sample_data,   counts are written to <sample_data>/<sample_files[0]> as a
#   sample_files   {bitstring: count} JSON dict, and the job record next to it
#                  as qpu_job.yaml (not .json: launch_sqd.py and check_inputs.py
#                  treat .json files in a samples directory as counts), with
#                  the QPU's own task logs as qpu_task_logs.txt
# Run from the repo root, inside such a Slurm job (workflow/3_sample_qpu_sbatch.sh).
#
# qpu_job.yaml records the time taken in two ways:
#   timing  wall clock of each step of this script, in seconds. wait_s runs from
#           submission until the job reaches a final state, so it includes the
#           time spent in the QPU's queue: QRMI reports no queued state for
#           this kind of resource, so the two cannot be told apart here.
#   qpu     when the QPU was executing, from the execution spans in the result
#           metadata. execution_s is the time covered by at least one span;
#           spans can overlap (classical processing alongside execution), so
#           summing them would count some of it twice. This is not the billed
#           usage (quantum_seconds), which QRMI's Python interface does not expose.

import json
import os
import sys
import time
from datetime import datetime, timezone

import yaml
from dotenv import load_dotenv
from qiskit import qpy
from qrmi.primitives.ibm import SamplerV2, get_backend

from build_hamiltonian import repo_path
from make_circuit import qrmi_resource

DEFAULT_SHOTS = 100_000
DEFAULT_DD_SEQUENCE = "XX"
# Seconds between status checks while the job runs; QRMI's own result() uses 1
POLL_INTERVAL = 1.0


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def execution_spans(metadata):
    """The QPU execution windows in a primitive result's metadata, or None if it has none."""
    spans = (metadata or {}).get("execution", {}).get("execution_spans")
    if not spans:
        return None
    windows = sorted((span.start, span.stop) for span in spans)
    # Time covered by at least one span: merge the overlapping ones
    covered = 0.0
    start, stop = windows[0]
    for next_start, next_stop in windows[1:]:
        if next_start > stop:
            covered += (stop - start).total_seconds()
            start, stop = next_start, next_stop
        else:
            stop = max(stop, next_stop)
    covered += (stop - start).total_seconds()
    return {
        "execution_s": covered,
        "first_start_utc": windows[0][0].isoformat(),
        "last_stop_utc": max(w[1] for w in windows).isoformat(),
        "spans": [{"start_utc": span.start.isoformat(), "stop_utc": span.stop.isoformat(),
                   "duration_s": span.duration, "shots": int(span.size)} for span in spans],
    }


# Parse input arguments
if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <jobname>")
    sys.exit(1)

jobname = sys.argv[1]

# Load configuration file
config_path = os.path.join("jobs", jobname, "config.yaml")
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
print(f"Loaded configuration from {config_path}")

backend_name = config["backend_name"]
shots = int(config.get("qpu_shots", DEFAULT_SHOTS))
dd = {
    "enable": bool(config.get("dd_enable", False)),
    "sequence_type": config.get("dd_sequence_type", DEFAULT_DD_SEQUENCE),
}
circuit_path = repo_path(config["circuit_data"])
sample_files = config["sample_files"]
if isinstance(sample_files, str):
    sample_files = [sample_files]
samples_dir = repo_path(config["sample_data"])
samples_path = samples_dir / sample_files[0]
record_path = samples_dir / "qpu_job.yaml"

# Refuse before spending QPU time, not after
for path in (samples_path, record_path):
    if path.exists():
        print(f"{path} already exists; move it away to sample again")
        sys.exit(1)
samples_dir.mkdir(parents=True, exist_ok=True)

# QRMI resource for the QPU. load_dotenv lets a .env supply QRMI settings when
# the Slurm plugin does not (it normally does).
timing = {}
step_start = script_start = time.perf_counter()


def step_done(name):
    """Record the wall time since the previous step ended as timing[<name>_s]."""
    global step_start
    now = time.perf_counter()
    timing[f"{name}_s"] = now - step_start
    step_start = now


load_dotenv()
qrmi = qrmi_resource(backend_name)
print(f"QRMI resource: {qrmi.metadata()}")
# qrmi >= 0.26.3: get_target(qrmi) is deprecated in favour of get_backend(qrmi).target.
target = get_backend(qrmi).target
step_done("load_target")

with open(circuit_path, 'rb') as fd:
    transpiled_circuit = qpy.load(fd)[0]
print(f"Loaded circuit from {circuit_path}: {transpiled_circuit.num_qubits} qubits, "
      f"{dict(transpiled_circuit.count_ops())}")
if transpiled_circuit.num_qubits > target.num_qubits:
    raise ValueError(f"circuit has {transpiled_circuit.num_qubits} qubits but "
                     f"{backend_name} has {target.num_qubits}")
step_done("load_circuit")

# QRMI's Options dataclass has only default_shots and run_options: a top-level
# dynamical_decoupling key raises TypeError. run_options is sent unchanged as the
# Runtime payload's "options", so Runtime SamplerV2 options belong inside it, and
# it is the server, not QRMI, that validates them.
options = {
        "default_shots": shots,
        "run_options": {"dynamical_decoupling": dd},
        }
print(f"Dynamical decoupling: {dd}")
sampler = SamplerV2(qrmi, options=options)

submitted = utc_now()
job = sampler.run([(transpiled_circuit,)])
print(f">>> Job ID: {job.job_id()}")
print(f">>> Job Status: {job.status()}")
step_done("submit")
# Wait here rather than in result(), so that waiting for the QPU and
# downloading the result are timed separately
while not job.in_final_state():
    time.sleep(POLL_INTERVAL)
finished = utc_now()
step_done("wait")
try:
    result = job.result()
except Exception:
    print(qrmi.task_logs(job.job_id()))
    raise
if job.errored():
    print(qrmi.task_logs(job.job_id()))
    raise RuntimeError(f"QPU job {job.job_id()} errored")
print(f">>> {result}")
pub_result = result[0]
counts = pub_result.data.meas.get_counts()
step_done("fetch")

with open(samples_path, "w") as f:
    json.dump(counts, f)
print(f"Saved {sum(counts.values())} shots ({len(counts)} distinct bitstrings) to {samples_path}")
step_done("save")
timing["total_s"] = time.perf_counter() - script_start

# The span format has not been seen from every QPU; a surprise in it must not
# lose the record of a job that has already run
try:
    qpu = execution_spans(result.metadata)
except Exception as exc:
    qpu = {"error": f"could not read execution spans: {exc!r}"}
if qpu is None:
    print("Note: the result metadata has no execution spans; qpu_job.yaml records no QPU time")
else:
    print(f"QPU execution time: {qpu.get('execution_s')} s; wall time {timing['total_s']:.1f} s")

job_record = {
    "job_id": job.job_id(),
    "backend": backend_name,
    "shots": shots,
    "dynamical_decoupling": dd,
    "num_bits": pub_result.data.meas.num_bits,
    "distinct_bitstrings": len(counts),
    "circuit": str(circuit_path),
    "counts_file": str(samples_path),
    "counts_file_bytes": samples_path.stat().st_size,
    "submitted_utc": submitted,
    "finished_utc": finished,
    "timing": timing,
    "qpu": qpu,
    "result_metadata": repr(result.metadata),
}
with open(record_path, "w") as f:
    yaml.safe_dump(job_record, f, sort_keys=False)
print(f"Saved job record to {record_path}")

# The QPU's own logs for the job, kept in case they say more than the above
logs_path = samples_dir / "qpu_task_logs.txt"
try:
    logs_path.write_text(qrmi.task_logs(job.job_id()))
    print(f"Saved task logs to {logs_path}")
except Exception as exc:
    print(f"Warning: could not fetch the task logs: {exc!r}")
