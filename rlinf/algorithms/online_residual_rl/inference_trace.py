# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Lossless request/reply traces written outside the robot control loop."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from queue import Full, Queue
from threading import Thread
from typing import Any, Iterator

import numpy as np


def _snapshot(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_snapshot(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


class InferenceTrace:
    """Record exchanges with a bounded queue and explicit loss counters.

    The owner must close this writer after the network worker has stopped.
    Images and inference tensors use NPZ without pickle or lossy compression.
    """

    def __init__(self, root: Path, session: str, metadata: dict | None = None):
        self.directory = Path(root) / f"session_{session}"
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "session": session,
                    "started_unix_ns": time.time_ns(),
                    "metadata": _snapshot(metadata),
                },
                indent=2,
            )
            + "\n"
        )
        self._queue: Queue = Queue(maxsize=128)
        self._closed = False
        self._counts = {"submitted": 0, "written": 0, "dropped": 0, "errors": 0}
        self._stream = (self.directory / "events.jsonl").open("a", buffering=1)
        self._thread = Thread(
            target=self._write_loop, name="inference-trace", daemon=True
        )
        self._thread.start()
        logging.info("Saving inference trace to %s", self.directory)

    def submit(self, event: dict) -> None:
        """Snapshot one exchange without waiting for filesystem writes."""
        if self._closed:
            raise RuntimeError("Inference trace is closed.")
        self._counts["submitted"] += 1
        try:
            self._queue.put_nowait(_snapshot(event))
        except Full:
            self._counts["dropped"] += 1
            logging.warning(
                "Inference trace queue full; dropped sequence=%s", event["sequence"]
            )

    def _write_loop(self) -> None:
        with self._stream as stream:
            while True:
                event = self._queue.get()
                if event is None:
                    return
                try:
                    arrays = {}
                    inference = event["request"]["type"] == "infer"

                    def encode(value):
                        if isinstance(value, np.ndarray):
                            if not inference:
                                return value.tolist()
                            key = f"array_{len(arrays):04d}"
                            arrays[key] = value
                            return {"__ndarray__": key}
                        if isinstance(value, dict):
                            return {key: encode(item) for key, item in value.items()}
                        if isinstance(value, list):
                            return [encode(item) for item in value]
                        return value

                    encoded = encode(event)
                    if arrays:
                        filename = f"exchange_{event['sequence']:08d}.npz"
                        np.savez(self.directory / filename, **arrays)
                        encoded["array_file"] = filename
                    stream.write(json.dumps(encoded) + "\n")
                    self._counts["written"] += 1
                except Exception:
                    self._counts["errors"] += 1
                    logging.exception("Failed to write inference trace exchange")

    def close(self) -> None:
        """Drain the queue and persist completeness counters; idempotent."""
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join()
        (self.directory / "summary.json").write_text(
            json.dumps(self._counts, indent=2) + "\n"
        )
        logging.info("Inference trace saved to %s: %s", self.directory, self._counts)


def iter_trace(directory: str | Path) -> Iterator[dict]:
    """Load completed exchanges, restoring exact array shapes and dtypes."""
    directory = Path(directory)
    with (directory / "events.jsonl").open() as stream:
        for line in stream:
            event = json.loads(line)
            filename = event.pop("array_file", None)
            if filename is None:
                yield event
                continue
            with np.load(directory / filename, allow_pickle=False) as arrays:

                def decode(value):
                    if isinstance(value, dict):
                        if set(value) == {"__ndarray__"}:
                            return arrays[value["__ndarray__"]].copy()
                        return {key: decode(item) for key, item in value.items()}
                    if isinstance(value, list):
                        return [decode(item) for item in value]
                    return value

                yield decode(event)
