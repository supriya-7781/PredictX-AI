"""
mqtt_client.py
==============
AWS IoT Core MQTT integration using paho-mqtt with TLS mutual auth.

Two modes controlled by STORAGE_MODE / explicit mock flag:
  - REAL mode  : connects to AWS IoT Core with x.509 certificates
  - MOCK mode  : publishes to an in-process queue (no AWS needed)

Usage (real):
    client = MQTTClient()
    client.connect()
    client.publish({"sensor": "reading"})
    client.disconnect()

Usage (mock):
    client = MQTTClient(mock=True)
    client.connect()
    client.publish({"sensor": "reading"})
    msg = client.mock_receive()   # pop message from internal queue
"""

from __future__ import annotations

import json
import os
import queue
import ssl
import threading
import time
from datetime import datetime, timezone
from typing import Callable

import paho.mqtt.client as paho
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
IOT_ENDPOINT = os.getenv("IOT_ENDPOINT", "")
IOT_CLIENT_ID = os.getenv("IOT_CLIENT_ID", "predictx-simulator")
IOT_CERT = os.getenv("IOT_CERT", "certs/device.pem.crt")
IOT_KEY = os.getenv("IOT_KEY", "certs/private.pem.key")
IOT_CA = os.getenv("IOT_CA", "certs/AmazonRootCA1.pem")
TOPIC_PREFIX = os.getenv("TOPIC_PREFIX", "predictx")
AWS_MQTT_PORT = 8883


def _sensor_topic(client_id: str = IOT_CLIENT_ID) -> str:
    return f"{TOPIC_PREFIX}/sensors/{client_id}"


# ---------------------------------------------------------------------------
# MQTT Client
# ---------------------------------------------------------------------------

class MQTTClient:
    """
    MQTT publisher / subscriber for PredictX sensor data.

    Parameters
    ----------
    mock : bool
        If True, skip real MQTT — use an in-process queue instead.
    on_message_callback : callable, optional
        Called with (topic, payload_dict) when a message arrives.
    """

    def __init__(
        self,
        mock: bool = False,
        on_message_callback: Callable[[str, dict], None] | None = None,
    ):
        self.mock = mock
        self.on_message_callback = on_message_callback
        self._connected = False
        self._client: paho.Client | None = None

        # In-process queue used in mock mode
        self._mock_queue: queue.Queue = queue.Queue()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        if self.mock:
            self._connected = True
            print("[mqtt]   Mock mode — no real connection.")
            return

        if not IOT_ENDPOINT:
            raise EnvironmentError(
                "IOT_ENDPOINT is not set. Add it to .env or use mock=True."
            )

        client = paho.Client(
            client_id=IOT_CLIENT_ID,
            protocol=paho.MQTTv311,
        )
        client.tls_set(
            ca_certs=IOT_CA,
            certfile=IOT_CERT,
            keyfile=IOT_KEY,
            tls_version=ssl.PROTOCOL_TLS_CLIENT,
        )

        client.on_connect = self._on_connect
        client.on_message = self._on_message
        client.on_disconnect = self._on_disconnect

        client.connect(IOT_ENDPOINT, port=AWS_MQTT_PORT, keepalive=60)
        client.loop_start()
        self._client = client

        # Wait up to 10 s for connection
        deadline = time.time() + 10
        while not self._connected and time.time() < deadline:
            time.sleep(0.1)

        if not self._connected:
            raise ConnectionError(
                f"Failed to connect to AWS IoT Core at {IOT_ENDPOINT}:{AWS_MQTT_PORT}"
            )

    def disconnect(self) -> None:
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None
        self._connected = False
        print("[mqtt]   Disconnected.")

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    def publish(
        self,
        payload: dict,
        topic: str | None = None,
        qos: int = 1,
    ) -> None:
        """
        Publish a sensor reading as a JSON payload.

        Adds ``timestamp`` automatically if not present.
        """
        if topic is None:
            topic = _sensor_topic()

        if "timestamp" not in payload:
            payload["timestamp"] = datetime.now(timezone.utc).isoformat()

        message = json.dumps(payload)

        if self.mock:
            self._mock_queue.put({"topic": topic, "payload": payload})
            print(f"[mqtt]   MOCK publish → {topic}: {message[:120]}")
            return

        if not self._connected or self._client is None:
            raise RuntimeError("Not connected. Call connect() first.")

        result = self._client.publish(topic, message, qos=qos)
        result.wait_for_publish(timeout=5)
        print(f"[mqtt]   Published → {topic} (mid={result.mid})")

    # ------------------------------------------------------------------
    # Subscribe
    # ------------------------------------------------------------------

    def subscribe(self, topic: str | None = None, qos: int = 1) -> None:
        if topic is None:
            topic = _sensor_topic("+")  # wildcard

        if self.mock:
            print(f"[mqtt]   MOCK subscribe to {topic}")
            return

        if not self._client:
            raise RuntimeError("Not connected.")
        self._client.subscribe(topic, qos=qos)
        print(f"[mqtt]   Subscribed to {topic}")

    # ------------------------------------------------------------------
    # Mock helper
    # ------------------------------------------------------------------

    def mock_receive(self, block: bool = False) -> dict | None:
        """
        Pop the next message from the mock queue.
        Returns None if the queue is empty and block=False.
        """
        try:
            return self._mock_queue.get(block=block, timeout=1.0)
        except queue.Empty:
            return None

    def mock_queue_size(self) -> int:
        return self._mock_queue.qsize()

    # ------------------------------------------------------------------
    # paho callbacks
    # ------------------------------------------------------------------

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            self._connected = True
            print(f"[mqtt]   Connected to AWS IoT Core ({IOT_ENDPOINT})")
        else:
            print(f"[mqtt]   Connection failed, rc={rc}")

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
        except json.JSONDecodeError:
            payload = {"raw": msg.payload.decode()}

        if self.on_message_callback:
            self.on_message_callback(msg.topic, payload)

    def _on_disconnect(self, client, userdata, rc, properties=None):
        self._connected = False
        if rc != 0:
            print(f"[mqtt]   Unexpected disconnect, rc={rc}")
