# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Non-blocking latest-target gripper commands shared by local and split control."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class _MoveCommand:
    width_m: float
    force_n: float


class LatestGripperCommand:
    """Send physical-width commands off the arm deadline thread."""

    def __init__(self, gripper: Any, *, velocity_m_s: float, force_n: float) -> None:
        self._gripper = gripper
        self._velocity_m_s = float(velocity_m_s)
        self._force_n = float(force_n)
        self._queue: queue.Queue[
            _MoveCommand | tuple[Callable[[], float], Future] | None
        ] = queue.Queue(maxsize=1)
        self._error: BaseException | None = None
        self._closed = False
        self._initialization: Future | None = None
        self._thread = threading.Thread(
            target=self._run, name="flexiv-gripper", daemon=True
        )
        self._thread.start()

    def submit(self, width_m: float, *, force_n: float | None = None) -> None:
        """Queue a latest target, optionally overriding the configured force limit."""
        if self._closed:
            raise RuntimeError("Flexiv gripper command worker is closed.")
        if self._error is not None:
            raise RuntimeError("Flexiv gripper command failed.") from self._error
        if self._initialization is not None and not self._initialization.done():
            raise RuntimeError(
                "Wait for gripper initialization before submitting targets."
            )
        command = _MoveCommand(
            float(width_m), self._force_n if force_n is None else float(force_n)
        )
        try:
            self._queue.put_nowait(command)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(command)

    def initialize(self, callback: Callable[[], float]) -> Future:
        """Run calibration after the current Move, dropping obsolete queued targets."""
        if self._closed or self._error is not None:
            raise RuntimeError("Flexiv gripper worker is unavailable.") from self._error
        if self._initialization is not None and not self._initialization.done():
            raise RuntimeError("Gripper initialization is already pending.")
        future = Future()
        self._initialization = future
        try:
            stale = self._queue.get_nowait()
            if isinstance(stale, tuple):
                self._queue.put_nowait(stale)
                raise RuntimeError("Gripper initialization is already pending.")
        except queue.Empty:
            pass
        self._queue.put_nowait((callback, future))
        return future

    def _run(self) -> None:
        try:
            while True:
                width = self._queue.get()
                if width is None:
                    return
                if isinstance(width, tuple):
                    callback, future = width
                    if not future.set_running_or_notify_cancel():
                        continue
                    try:
                        future.set_result(callback())
                    except BaseException as exc:
                        future.set_exception(exc)
                        raise
                else:
                    self._gripper.Move(width.width_m, self._velocity_m_s, width.force_n)
        except BaseException as exc:
            self._error = exc

    def close(self) -> None:
        """Stop accepting targets and wait briefly for the current move."""
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            try:
                pending = self._queue.get_nowait()
                if isinstance(pending, tuple):
                    pending[1].cancel()
            except queue.Empty:
                pass
            self._queue.put_nowait(None)
        initializing = (
            self._initialization is not None and self._initialization.running()
        )
        self._thread.join(timeout=35.0 if initializing else 2.0)
        if initializing and self._thread.is_alive():
            raise RuntimeError("Gripper initialization did not finish during shutdown.")
