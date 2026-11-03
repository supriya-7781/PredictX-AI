"""
sensor_simulator.py
===================
Simulates a machine sensor that reads rows from ai4i2020.csv and either
prints them to the console (mock mode) or publishes them to AWS IoT Core
via MQTT.

Usage
-----
    # Print to console (no AWS required):
    python simulator/sensor_simulator.py --mode mock --delay 1

    # Publish to AWS IoT Core (requires .env credentials + certs):
    python simulator/sensor_simulator.py --mode mqtt --delay 2

    # Loop forever (default is one-pass):
    python simulator/sensor_simulator.py --mode mock --loop

    # Publish only N rows:
    python simulator/sensor_simulator.py --mode mock --rows 50
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from datetime import datetime, timezone

import pandas as pd

# ---------------------------------------------------------------------------
# Make sure the src/ package is importable when running from repo root
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from predictx.iot.mqtt_client import MQTTClient  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = ROOT / "data" / "ai4i2020.csv"

FEATURE_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

OPTIONAL_COLS = ["UDI", "Product ID", "Type", "Machine failure"]


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class SensorSimulator:
    def __init__(
        self,
        data_path: pathlib.Path = DATA_PATH,
        mode: str = "mock",
        delay: float = 1.0,
        rows: int | None = None,
        loop: bool = False,
    ):
        self.data_path = data_path
        self.mode = mode            # "mock" or "mqtt"
        self.delay = delay
        self.max_rows = rows
        self.loop = loop

        self.df = self._load_data()
        self.mqtt_client = MQTTClient(mock=(mode == "mock"))

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        print(f"[sim]    Loaded {len(df):,} rows from {self.data_path.name}")
        return df

    def _row_to_payload(self, row: pd.Series) -> dict:
        payload: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        for col in FEATURE_COLS:
            if col in row.index:
                payload[col] = round(float(row[col]), 3)
        for col in OPTIONAL_COLS:
            if col in row.index:
                payload[col] = str(row[col]) if col in ("Product ID", "Type") else int(row[col])
        return payload

    def run(self) -> None:
        self.mqtt_client.connect()
        print(f"[sim]    Mode={self.mode}, delay={self.delay}s, loop={self.loop}")
        print("[sim]    Press Ctrl+C to stop.\n")

        total_sent = 0
        try:
            while True:
                df_iter = self.df.iterrows()
                for idx, row in df_iter:
                    if self.max_rows is not None and total_sent >= self.max_rows:
                        print(f"[sim]    Reached row limit ({self.max_rows}). Stopping.")
                        return

                    payload = self._row_to_payload(row)
                    self.mqtt_client.publish(payload)
                    total_sent += 1

                    if self.mode == "mock":
                        # Pretty-print for console mode
                        _print_payload(payload, total_sent)

                    time.sleep(self.delay)

                if not self.loop:
                    break

        except KeyboardInterrupt:
            print(f"\n[sim]    Stopped after {total_sent} readings.")
        finally:
            if self.mode == "mqtt":
                self.mqtt_client.disconnect()

        print(f"[sim]    Done. Total readings sent: {total_sent}")


def _print_payload(payload: dict, n: int) -> None:
    ts = payload.get("timestamp", "")
    air = payload.get("Air temperature [K]", "-")
    proc = payload.get("Process temperature [K]", "-")
    rpm = payload.get("Rotational speed [rpm]", "-")
    torque = payload.get("Torque [Nm]", "-")
    wear = payload.get("Tool wear [min]", "-")
    failure = payload.get("Machine failure", "-")
    print(
        f"[{n:05d}] {ts}  |  "
        f"AirT={air}K  ProcT={proc}K  RPM={rpm}  "
        f"Torque={torque}Nm  Wear={wear}min  Failure={failure}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PredictX sensor simulator — stream CSV rows as MQTT messages."
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "mqtt"],
        default="mock",
        help="mock = print to console; mqtt = publish to AWS IoT Core",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="Delay between readings (default: 1.0 s)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N rows (default: run through entire dataset)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Restart from the beginning when the dataset ends",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DATA_PATH),
        metavar="PATH",
        help=f"Path to sensor CSV (default: {DATA_PATH})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sim = SensorSimulator(
        data_path=pathlib.Path(args.data),
        mode=args.mode,
        delay=args.delay,
        rows=args.rows,
        loop=args.loop,
    )
    sim.run()
