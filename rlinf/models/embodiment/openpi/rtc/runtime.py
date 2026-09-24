# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fixed-deadline control loop joining RTC scheduling to a robot adapter."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from rlinf.utils.logging import get_logger

from .scheduler import RTCChunkScheduler
from .types import PhysicalActionStep, PolicyObservation


class RobotAdapter(Protocol):
    """Lifecycle and non-blocking command surface required by the runtime."""

    def start(self) -> None: ...

    def read_observation(self, *, tick: int, prompt: str) -> PolicyObservation: ...

    def apply_policy_action(self, step: PhysicalActionStep) -> None: ...

    def hold_target(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


class LateChunkBehavior(str, Enum):
    """Safe behavior when every delta in the installed chunk was consumed."""

    HOLD = "hold"
    STOP = "stop"


@dataclass(frozen=True)
class RuntimeConfig:
    """Control-loop settings independent of the robot implementation."""

    prompt: str
    control_hz: float
    max_ticks: int | None = None
    late_chunk_behavior: LateChunkBehavior = LateChunkBehavior.HOLD
    deadline_warn_s: float = 0.005

    def validate(self) -> None:
        """Validate scheduling settings before robot startup."""
        if not self.prompt:
            raise ValueError("prompt must not be empty.")
        if self.control_hz <= 0:
            raise ValueError("control_hz must be positive.")
        if self.max_ticks is not None and self.max_ticks <= 0:
            raise ValueError("max_ticks must be positive when configured.")


class RTCRobotRuntime:
    """Run the single-owner control sequence at absolute deadlines."""

    def __init__(
        self,
        robot: RobotAdapter,
        scheduler: RTCChunkScheduler,
        config: RuntimeConfig,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        config.validate()
        self.robot = robot
        self.scheduler = scheduler
        self.config = config
        self._monotonic = monotonic
        self._sleep = sleep
        self._logger = get_logger()
        self._stop_event = threading.Event()
        self._closed = False

    def request_stop(self) -> None:
        """Ask the owner thread to stop after its current control tick."""
        self._stop_event.set()

    def run(self) -> int:
        """Bootstrap once, then run inference without blocking normal ticks."""
        tick = 0
        self.robot.start()
        try:
            observation = self.robot.read_observation(
                tick=tick, prompt=self.config.prompt
            )
            self.scheduler.bootstrap(observation, observation_tick=tick)
            period = 1.0 / self.config.control_hz
            next_deadline = self._monotonic()
            while not self._stop_event.is_set():
                if self.config.max_ticks is not None and tick >= self.config.max_ticks:
                    break
                next_deadline += period

                # Poll before the observation snapshot so a completed chunk does
                # not wait behind camera/state acquisition. Poll again afterward
                # to make completion during acquisition deterministic this tick.
                self.scheduler.accept_ready(accept_tick=tick)
                if tick > 0:
                    observation = self.robot.read_observation(
                        tick=tick, prompt=self.config.prompt
                    )
                self.scheduler.accept_ready(accept_tick=tick)
                self.scheduler.submit_if_due(observation, observation_tick=tick)
                self.scheduler.accept_ready(accept_tick=tick)

                step = self.scheduler.next_step()
                if step is None:
                    if self.config.late_chunk_behavior is LateChunkBehavior.STOP:
                        self._logger.error(
                            "RTC chunk exhausted at tick=%s; controlled stop.", tick
                        )
                        break
                    self.robot.hold_target()
                else:
                    self.robot.apply_policy_action(step)
                self.scheduler.commit_tick()
                tick += 1

                remaining = next_deadline - self._monotonic()
                if remaining > 0:
                    self._sleep(remaining)
                elif -remaining >= self.config.deadline_warn_s:
                    self._logger.warning(
                        "RTC deadline miss tick=%s miss_ms=%.3f",
                        tick,
                        -remaining * 1000.0,
                    )
            return 0
        finally:
            self.scheduler.stop_accepting()
            self.robot.stop()
            self.scheduler.close(wait=True)
            self.robot.close()
            self._closed = True

    def close(self) -> None:
        """Idempotently release resources after a partial-start failure."""
        if self._closed:
            return
        self.request_stop()
        try:
            self.scheduler.close(wait=True)
        finally:
            try:
                self.robot.stop()
            finally:
                self.robot.close()
        self._closed = True
