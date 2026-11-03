"""
store.py
========
Storage abstraction — works in ``local`` mode (default) or ``s3`` mode.

Set STORAGE_MODE=local  → read/write under LOCAL_STORE directory.
Set STORAGE_MODE=s3     → use boto3 to read/write S3_BUCKET.

Both modes expose the same interface:
    save_result(df, key)  → stores a prediction result CSV
    load_result(key)      → returns a DataFrame
    list_results()        → returns list of stored keys
    upload_raw(path, key) → upload a raw file (CSV, model, etc.)
"""

from __future__ import annotations

import io
import os
import pathlib
from datetime import datetime, timezone

import boto3
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
STORAGE_MODE = os.getenv("STORAGE_MODE", "local").lower()
LOCAL_STORE = pathlib.Path(os.getenv("LOCAL_STORE", "local_store"))
S3_BUCKET = os.getenv("S3_BUCKET", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
RESULTS_PREFIX = "results/"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _timestamp_key() -> str:
    """Generate a timestamped key like ``results/2024-01-15T12-30-00.csv``."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    return f"{RESULTS_PREFIX}{ts}.csv"


def _s3_client():
    return boto3.client("s3", region_name=AWS_REGION)


# ---------------------------------------------------------------------------
# Local storage
# ---------------------------------------------------------------------------

class _LocalStore:
    def __init__(self, root: pathlib.Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, df: pd.DataFrame, key: str) -> str:
        path = self.root / key.lstrip("/")
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return str(path)

    def load(self, key: str) -> pd.DataFrame:
        path = self.root / key.lstrip("/")
        if not path.exists():
            raise FileNotFoundError(f"Local store: {path} not found.")
        return pd.read_csv(path)

    def list(self, prefix: str = RESULTS_PREFIX) -> list[str]:
        base = self.root / prefix.lstrip("/")
        if not base.exists():
            return []
        return sorted(
            str(p.relative_to(self.root)).replace("\\", "/")
            for p in base.glob("**/*.csv")
        )

    def upload_raw(self, local_path: str | pathlib.Path, key: str) -> str:
        src = pathlib.Path(local_path)
        dst = self.root / key.lstrip("/")
        dst.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(src, dst)
        return str(dst)


# ---------------------------------------------------------------------------
# S3 storage
# ---------------------------------------------------------------------------

class _S3Store:
    def __init__(self, bucket: str, region: str):
        self.bucket = bucket
        self.region = region

    def save(self, df: pd.DataFrame, key: str) -> str:
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        _s3_client().put_object(
            Bucket=self.bucket,
            Key=key,
            Body=csv_buf.getvalue().encode("utf-8"),
            ContentType="text/csv",
        )
        return f"s3://{self.bucket}/{key}"

    def load(self, key: str) -> pd.DataFrame:
        obj = _s3_client().get_object(Bucket=self.bucket, Key=key)
        return pd.read_csv(io.BytesIO(obj["Body"].read()))

    def list(self, prefix: str = RESULTS_PREFIX) -> list[str]:
        paginator = _s3_client().get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return sorted(keys)

    def upload_raw(self, local_path: str | pathlib.Path, key: str) -> str:
        _s3_client().upload_file(str(local_path), self.bucket, key)
        return f"s3://{self.bucket}/{key}"


# ---------------------------------------------------------------------------
# Public API — thin wrapper that delegates to the right backend
# ---------------------------------------------------------------------------

def _get_backend():
    if STORAGE_MODE == "s3":
        if not S3_BUCKET:
            raise EnvironmentError(
                "STORAGE_MODE=s3 but S3_BUCKET is not set in .env"
            )
        return _S3Store(S3_BUCKET, AWS_REGION)
    return _LocalStore(LOCAL_STORE)


def save_result(df: pd.DataFrame, key: str | None = None) -> str:
    """
    Persist a prediction result DataFrame.

    Returns the storage location (file path or s3:// URI).
    """
    if key is None:
        key = _timestamp_key()
    return _get_backend().save(df, key)


def load_result(key: str) -> pd.DataFrame:
    """Load a previously saved prediction result."""
    return _get_backend().load(key)


def list_results() -> list[str]:
    """Return a sorted list of all stored result keys."""
    return _get_backend().list()


def upload_raw(local_path: str | pathlib.Path, key: str) -> str:
    """Upload a raw file (model, CSV, etc.) to storage."""
    return _get_backend().upload_raw(local_path, key)
