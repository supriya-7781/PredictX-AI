"""
handler.py — AWS Lambda entry point
====================================
Triggered by an AWS IoT Rule that forwards MQTT messages on
``predictx/sensors/+`` to this function.

Event format (from IoT Rule SQL):
    SELECT *, topic() AS _topic, timestamp() AS _iot_ts FROM 'predictx/sensors/+'

The handler:
  1. Deserialises the sensor payload
  2. Runs Isolation Forest inference (model bundled in the Lambda ZIP)
  3. Writes the result row to S3 under ``results/``
  4. Logs to CloudWatch

Local testing:
    python lambda_function/handler.py
"""

from __future__ import annotations

import json
import os
import io
import pathlib
import sys
import logging
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# When running inside Lambda the package is at /var/task.
# When testing locally we add the project src/ to sys.path.
# ---------------------------------------------------------------------------
_LAMBDA_ROOT = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _LAMBDA_ROOT.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
import boto3

# ---------------------------------------------------------------------------
# Model paths
# When deployed to Lambda, the model files must be included in the ZIP
# alongside handler.py (i.e. models/ is inside the ZIP root).
# ---------------------------------------------------------------------------
_MODEL_PATH = _LAMBDA_ROOT / "models" / "isolation_forest.pkl"
_SCALER_PATH = _LAMBDA_ROOT / "models" / "scaler.pkl"

# Fall back to the project-level models/ when running locally
if not _MODEL_PATH.exists():
    _MODEL_PATH = _PROJECT_ROOT / "models" / "isolation_forest.pkl"
    _SCALER_PATH = _PROJECT_ROOT / "models" / "scaler.pkl"

FEATURE_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# ---------------------------------------------------------------------------
# S3 config
# ---------------------------------------------------------------------------
S3_BUCKET = os.getenv("S3_BUCKET", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
RESULTS_PREFIX = "results/"


# ---------------------------------------------------------------------------
# Lazy-loaded model artefacts
# ---------------------------------------------------------------------------
_model = None
_scaler = None


def _load_model():
    global _model, _scaler
    if _model is None:
        logger.info(f"Loading model from {_MODEL_PATH}")
        _scaler = joblib.load(_SCALER_PATH)
        _model = joblib.load(_MODEL_PATH)
    return _scaler, _model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(payload: dict) -> dict:
    """Run Isolation Forest on a single sensor reading."""
    scaler, model = _load_model()

    missing = [c for c in FEATURE_COLS if c not in payload]
    if missing:
        raise ValueError(f"Payload missing required fields: {missing}")

    X_raw = [[payload[c] for c in FEATURE_COLS]]
    X = scaler.transform(X_raw)
    score = float(model.decision_function(X)[0])
    pred = int(model.predict(X)[0])   # 1=normal, -1=anomaly
    is_anomaly = 1 if pred == -1 else 0

    result = {
        **payload,
        "anomaly_score": round(score, 6),
        "is_anomaly": is_anomaly,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    return result


# ---------------------------------------------------------------------------
# S3 write
# ---------------------------------------------------------------------------

def _write_to_s3(result: dict) -> str | None:
    if not S3_BUCKET:
        logger.warning("S3_BUCKET not set — skipping S3 write.")
        return None

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%f")
    key = f"{RESULTS_PREFIX}{ts}.json"

    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=json.dumps(result).encode("utf-8"),
        ContentType="application/json",
    )
    logger.info(f"Wrote result to s3://{S3_BUCKET}/{key}")
    return f"s3://{S3_BUCKET}/{key}"


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------

def lambda_handler(event: dict, context) -> dict:
    """
    AWS Lambda entry point.

    ``event`` is the IoT Rule SQL output — a dict matching the sensor payload
    schema plus optional ``_topic`` and ``_iot_ts`` injected by the rule.
    """
    logger.info(f"Received event: {json.dumps(event)}")

    try:
        result = run_inference(event)
        location = _write_to_s3(result)

        label = "ANOMALY" if result["is_anomaly"] else "NORMAL"
        logger.info(
            f"{label} | score={result['anomaly_score']:.4f} | "
            f"stored={location}"
        )

        return {
            "statusCode": 200,
            "body": json.dumps(result),
        }

    except (ValueError, KeyError) as exc:
        logger.error(f"Bad payload: {exc}")
        return {"statusCode": 400, "body": str(exc)}
    except Exception as exc:
        logger.exception(f"Unexpected error: {exc}")
        return {"statusCode": 500, "body": str(exc)}


# ---------------------------------------------------------------------------
# Local test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_event = {
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    print("Testing Lambda handler locally...")
    print("Input:", json.dumps(test_event, indent=2))
    response = lambda_handler(test_event, None)
    print("Output:", json.dumps(json.loads(response["body"]), indent=2))

    # Test an anomalous reading
    anomaly_event = {
        "Air temperature [K]": 304.5,
        "Process temperature [K]": 313.8,
        "Rotational speed [rpm]": 2940,
        "Torque [Nm]": 5.2,
        "Tool wear [min]": 290,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    print("\nTesting anomalous reading...")
    response2 = lambda_handler(anomaly_event, None)
    print("Output:", json.dumps(json.loads(response2["body"]), indent=2))
