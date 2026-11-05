# Workflow for observability demo on Miyabi
#
# Author: Naoki Kanazawa (knzwnao@jp.ibm.com)

import io
import os
import uuid
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
from prefect.artifacts import create_table_artifact
from prefect.context import FlowRunContext
from prefect.settings import get_current_settings

try:
    from prefect_aws.s3 import S3Bucket
except Exception:
    S3Bucket = None  # type: ignore


def _flow_scoped_subdir() -> str:
    ctx = FlowRunContext.get()
    if ctx and ctx.flow and ctx.flow_run:
        return os.path.join(ctx.flow.name, ctx.flow_run.name)
    return os.path.join("no-context", uuid.uuid4().hex)


def _local_storage_dir() -> Path:
    settings = get_current_settings()
    return Path(settings.home) / "storage"


def save_ndarray(
    file_prefix: str,
    **arrays: np.ndarray,
) -> str:

    subdir = _flow_scoped_subdir()

    # --- Try S3 first ---
    if S3Bucket is not None:
        try:
            s3_client = S3Bucket.load("s3-sqd")
            s3_client.bucket_folder = subdir

            with io.BytesIO() as buf:
                np.savez_compressed(buf, **arrays, allow_pickle=False)
                buf.seek(0)
                key = s3_client.upload_from_file_object(
                    buf,
                    to_path=f"{file_prefix}_{uuid.uuid4().hex}.npz",
                )
            return str(key)
        except Exception:
            pass

    # --- Fallback to local ---
    base = _local_storage_dir() / "sqd_data" / subdir
    base.mkdir(parents=True, exist_ok=True)
    local_path = base / f"{file_prefix}_{uuid.uuid4().hex}.npz"

    with open(local_path, "wb") as f:
        with io.BytesIO() as buf:
            np.savez_compressed(buf, **arrays, allow_pickle=False)
            buf.seek(0)
            f.write(buf.read())

    return f"file://{local_path}"


def load_ndarray(
    file_path: str,
    key: str,
) -> np.ndarray:
    # --- Local ---
    if file_path.startswith("file://"):
        local_path = file_path[len("file://") :]
        with open(local_path, "rb") as f:
            data = f.read()
        with io.BytesIO(data) as buf:
            buf.seek(0)
            arr = np.load(buf, allow_pickle=False).get(key)
        assert arr is not None
        return arr

    # --- S3 (backward compatible) ---
    s3_client = S3Bucket.load("s3-sqd")
    with io.BytesIO() as buf:
        s3_client.download_object_to_file_object(file_path, buf)
        buf.seek(0)
        arr = np.load(buf, allow_pickle=False).get(key)
    assert arr is not None
    return arr


def extend_table_artifact(
    artifact_key: str,
    new_table: list[dict[str, Any]] | list[list[Any]],
    index: int = 0,
) -> UUID:
    """Record a telemetry table under ``artifact_key``.

    Historically this read the existing artifact and appended to it
    (read-modify-update). That is unreliable against the Prefect server in
    use: a just-created artifact is not immediately returned by
    ``read_artifacts`` (eventual consistency), so the read came back empty and
    ``artifacts[index]`` raised ``IndexError``.

    Instead, create a new version of the artifact under the same key on each
    call. Prefect keys are versioned, so repeated creates accumulate as
    versions (the UI shows the latest), and nothing depends on reading an
    artifact back immediately after writing it.

    ``index`` is retained for backwards compatibility and is unused.
    """
    return create_table_artifact(
        table=list(new_table),
        key=artifact_key,
        description="SQD intermediate data.",
    )
