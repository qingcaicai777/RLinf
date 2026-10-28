# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-light asynchronous client for split robot inference."""

from __future__ import annotations

import logging
import socket
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from queue import PriorityQueue
from typing import Any

if __package__:
    from .inference_trace import InferenceTrace
    from .remote_wire import PROTOCOL_VERSION, receive_message, send_message
else:
    from inference_trace import InferenceTrace
    from remote_wire import PROTOCOL_VERSION, receive_message, send_message


class RemotePolicy:
    """Serialize socket operations off the robot control thread."""

    def __init__(
        self,
        host: str,
        port: int,
        session: str,
        *,
        input_shapes: dict[str, tuple[int, ...]],
        backend: str = "torch",
        action_representation: str = "eef_delta",
        binary_gripper: bool = False,
        gripper_threshold: float = 0.5,
        diagnostic_dir: Path | None = None,
        diagnostic_metadata: dict | None = None,
    ):
        self.socket = socket.create_connection((host, port), timeout=10.0)
        self.socket.settimeout(30.0)
        self.session = session
        self.diagnostics_enabled = diagnostic_dir is not None
        try:
            self._trace = (
                InferenceTrace(diagnostic_dir, session, diagnostic_metadata)
                if diagnostic_dir is not None
                else None
            )
        except BaseException:
            self.socket.close()
            raise
        self._shared_clock = host in {"127.0.0.1", "::1", "localhost"}
        self._closed = False
        self._lock = threading.Lock()
        self._pending = 0
        self._sequence = 0
        self._failure: BaseException | None = None
        self._queue: PriorityQueue[
            tuple[int, int, dict[str, Any] | None, Future | None, float]
        ] = PriorityQueue()
        self._worker = threading.Thread(
            target=self._exchange_loop, name="policy-network", daemon=True
        )
        self._worker.start()
        try:
            started = self.call(
                "start",
                input_shapes=input_shapes,
                backend=backend,
                action_representation=action_representation,
                binary_gripper=binary_gripper,
                gripper_threshold=gripper_threshold,
            ).result(timeout=10.0)
            if self.diagnostics_enabled and not started.get(
                "supports_diagnostics", False
            ):
                raise ValueError(
                    "Restart the inference server to enable diagnostic traces."
                )
            if started.get("binary_gripper", False) != binary_gripper:
                raise ValueError(
                    "Robot and inference server binary gripper settings differ."
                )
            if (
                binary_gripper
                and started.get("gripper_threshold", 0.5) != gripper_threshold
            ):
                raise ValueError(
                    "Robot and learner gripper thresholds differ: "
                    f"robot={gripper_threshold!r}, "
                    f"learner={started.get('gripper_threshold', 0.5)!r}."
                )
            self.action_representation = str(
                started.get("action_representation", "eef_delta")
            )
            expected_width = 8 if action_representation == "joint_absolute" else 7
            if (
                action_representation not in {"eef_delta", "joint_absolute"}
                or self.action_representation != action_representation
                or int(started.get("action_dim", 7)) != expected_width
            ):
                raise ValueError(
                    "Robot and inference server action representations differ."
                )
            self.inference_horizon = int(started["inference_horizon"])
            if self.inference_horizon <= 0:
                raise ValueError("Server returned an invalid inference horizon.")
            self.base_horizon = int(started.get("base_horizon", self.inference_horizon))
            self.base_request_steps = int(
                started.get("base_request_steps", self.base_horizon)
            )
            self.coarse_origin_schedule = bool(
                started.get("coarse_origin_schedule", False)
            )
            self.fine_horizon = int(started.get("fine_horizon", self.inference_horizon))
            self.fine_available = bool(started.get("fine_available", False))
            self.continuous_episodes = bool(started.get("continuous_episodes", False))
            self.initial_phase = str(started.get("initial_phase", "fine"))
            self.rtc_enabled = bool(started.get("rtc_enabled", False))
            if self.initial_phase not in {"coarse", "fine"}:
                raise ValueError("Server returned an invalid inference phase.")
        except BaseException:
            self._closed = True
            self.socket.close()
            self._queue.put((3, self._sequence, None, None, time.monotonic()))
            self._worker.join(timeout=1.0)
            if self._trace is not None:
                self._trace.close()
            raise

    def completion_upper_bound(self, server_monotonic: float) -> float | None:
        """Use a server timestamp only when both processes share the host clock."""
        return server_monotonic if self._shared_clock else None

    def call(self, kind: str, **payload: Any) -> Future:
        """Queue robot events in call order on the shared socket."""
        if self._closed:
            raise RuntimeError("Remote policy is closed.")
        with self._lock:
            if self._failure is not None:
                raise RuntimeError("Inference connection failed.") from self._failure
            if self._pending >= 64:
                raise RuntimeError("Inference/receipt queue exceeded 64 messages.")
            self._pending += 1
            sequence = self._sequence
            self._sequence += 1
        future = Future()
        if kind == "infer" and self.diagnostics_enabled:
            payload["diagnostics"] = True
        priority = 2 if kind == "finish" else 0
        self._queue.put(
            (
                priority,
                sequence,
                {
                    "version": PROTOCOL_VERSION,
                    "type": kind,
                    "session": self.session,
                    **payload,
                },
                future,
                time.monotonic(),
            )
        )
        return future

    def _exchange_loop(self) -> None:
        """Keep socket exchanges serialized in request order."""
        while True:
            _priority, _sequence, request, future, enqueued_at = self._queue.get()
            if request is None:
                return
            assert future is not None
            response = None
            exchange_started = sent_at = received_at = None
            try:
                with self._lock:
                    failure = self._failure
                if failure is not None:
                    raise RuntimeError("Inference connection failed.") from failure
                exchange_started = time.monotonic()
                send_message(self.socket, request)
                sent_at = time.monotonic()
                response = receive_message(self.socket)
                received_at = time.monotonic()
                response["_client_received_at"] = received_at
                if request.get("type") == "infer":
                    logging.getLogger("flexiv.status").info(
                        "inference transport session=%s frame=%s queue_ms=%.2f send_ms=%.2f response_ms=%.2f",
                        self.session,
                        request.get("frame"),
                        (exchange_started - enqueued_at) * 1000.0,
                        (sent_at - exchange_started) * 1000.0,
                        (received_at - sent_at) * 1000.0,
                    )
                if response.get("type") == "error":
                    raise RuntimeError(str(response.get("detail")))
                if response.get("session") != self.session:
                    raise ValueError("Inference reply belongs to another session.")
            except BaseException as exc:
                if self._trace is not None:
                    self._trace.submit(
                        {
                            "sequence": _sequence,
                            "request": request,
                            "response": response,
                            "enqueued_monotonic": enqueued_at,
                            "exchange_started_monotonic": exchange_started,
                            "sent_monotonic": sent_at,
                            "received_monotonic": received_at,
                            "error": repr(exc),
                        }
                    )
                with self._lock:
                    self._failure = exc
                    self._pending -= 1
                future.set_exception(exc)
            else:
                if self._trace is not None:
                    self._trace.submit(
                        {
                            "sequence": _sequence,
                            "request": request,
                            "response": response,
                            "enqueued_monotonic": enqueued_at,
                            "exchange_started_monotonic": exchange_started,
                            "sent_monotonic": sent_at,
                            "received_monotonic": received_at,
                        }
                    )
                with self._lock:
                    self._pending -= 1
                future.set_result(response)

    def close(self, frame: int, success: bool | None, *, discard: bool = False) -> None:
        """Finalize only after robot motion has stopped; idempotent."""
        if self._closed:
            return
        try:
            with self._lock:
                failed = self._failure is not None
            if not failed:
                self.call(
                    "finish", frame=frame, success=success, discard=discard
                ).result(timeout=30.0)
        finally:
            self._closed = True
            self._queue.put((3, self._sequence, None, None, time.monotonic()))
            if self._failure is not None:
                try:
                    self.socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            self._worker.join(timeout=31.0)
            self.socket.close()
            if self._trace is not None:
                self._trace.close()
