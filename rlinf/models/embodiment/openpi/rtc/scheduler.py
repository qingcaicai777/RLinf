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

"""Robot-independent asynchronous PI0.5 RTC chunk scheduler."""

from __future__ import annotations

import collections
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from rlinf.utils.logging import get_logger

from .types import (
    InferenceRequest,
    PhysicalActionStep,
    PolicyObservation,
    PolicyResult,
    RTCGuidance,
    SchedulerEvent,
)


def rtc_decay_endpoint(
    action_horizon: int, submitted_cursor: int, inference_delay: int
) -> int:
    """Return the ordinary speed-field guidance decay endpoint."""
    return min(
        action_horizon,
        max(inference_delay, action_horizon - submitted_cursor + inference_delay),
    )


class RTCChunkScheduler:
    """Own chunk cursors and exchange immutable work with one inference thread."""

    def __init__(
        self,
        infer: Callable[[InferenceRequest], PolicyResult],
        *,
        action_horizon: int = 12,
        action_dim: int = 7,
        execution_horizon: int = 6,
        initial_delay: int = 1,
        delay_history_length: int = 8,
        max_guidance_weight: float = 5.0,
        num_steps: int = 10,
    ) -> None:
        if not 0 < execution_horizon < action_horizon:
            raise ValueError("execution_horizon must be in [1, action_horizon).")
        if (
            initial_delay < 0
            or delay_history_length <= 0
            or max_guidance_weight <= 0
            or num_steps <= 0
        ):
            raise ValueError("RTC delay, history, weight, and steps must be valid.")
        self._infer = infer
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.execution_horizon = execution_horizon
        self.initial_delay = initial_delay
        self.max_guidance_weight = max_guidance_weight
        self.num_steps = num_steps
        self.delay_history: collections.deque[int] = collections.deque(
            maxlen=delay_history_length
        )
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="pi05-rtc-infer"
        )
        # Start the only worker before robot activation so the first steady-state
        # submit never pays thread creation latency on the control thread.
        self._executor.submit(lambda: None).result()
        self._future: Future[PolicyResult] | None = None
        self._current: PolicyResult | None = None
        self._cursor = 0
        self._committed_tick = 0
        self._generation_id = 0
        self._session_id = 0
        self._next_chunk_id = 0
        self._accepting = True
        self._closed = False
        self._owner_thread = threading.get_ident()
        self._logger = get_logger()

    @property
    def cursor(self) -> int:
        """Return the next index in the installed physical chunk."""
        return self._cursor

    @property
    def exhausted(self) -> bool:
        """Return whether no unconsumed physical delta remains."""
        return self._current is None or self._cursor >= len(
            self._current.physical_delta_chunk
        )

    def _assert_owner(self) -> None:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError(
                "RTC scheduler cursors may only be changed by the control thread."
            )

    def reset(self, *, session_id: int | None = None) -> None:
        """Invalidate late results and clear chunks without waiting for inference."""
        self._assert_owner()
        self._generation_id += 1
        if session_id is not None:
            self._session_id = session_id
        self._current = None
        self._cursor = 0
        self._committed_tick = 0
        self.delay_history.clear()
        if self._future is not None and self._future.cancel():
            self._future = None

    def bootstrap(
        self, observation: PolicyObservation, *, observation_tick: int = 0
    ) -> PolicyResult:
        """Run the first unguided inference before steady control starts."""
        self._assert_owner()
        if self._future is not None:
            raise RuntimeError("Cannot bootstrap while inference is in flight.")
        request = self._make_request(observation, observation_tick, guidance=None)
        result = self._infer(request)
        self._current = result
        self._cursor = 0
        return result

    def should_submit(self) -> bool:
        """Return whether the current cursor reached the prefetch boundary."""
        return (
            self._accepting
            and self._future is None
            and self._current is not None
            and self._cursor >= self.execution_horizon
        )

    def submit_if_due(
        self, observation: PolicyObservation, *, observation_tick: int
    ) -> bool:
        """Submit at most one inference from this control tick's snapshot."""
        self._assert_owner()
        if not self.should_submit():
            return False
        assert self._current is not None
        remaining = self._current.normalized_raw_chunk[self._cursor :]
        guidance = self._make_guidance(remaining, submitted_cursor=self._cursor)
        request = self._make_request(observation, observation_tick, guidance=guidance)
        self._future = self._executor.submit(self._infer, request)
        self._logger.info(
            "rtc submit session=%s generation=%s chunk=%s observation_tick=%s "
            "submit_tick=%s committed=%s cursor=%s prefix=%s",
            request.session_id,
            request.generation_id,
            request.chunk_id,
            request.observation_tick,
            request.submit_tick,
            request.submit_committed_tick,
            request.old_chunk_cursor,
            0 if guidance is None else guidance.prefix_len,
        )
        return True

    def accept_ready(self, *, accept_tick: int) -> SchedulerEvent | None:
        """Install a completed result without blocking the control thread."""
        self._assert_owner()
        if self._future is None or not self._future.done():
            return None
        future = self._future
        self._future = None
        result = future.result()
        request = result.request
        stale_prefix = self._committed_tick - request.submit_committed_tick
        reason = ""
        if (
            request.generation_id != self._generation_id
            or request.session_id != self._session_id
        ):
            reason = "stale generation"
        elif stale_prefix < 0 or stale_prefix >= self.action_horizon:
            reason = "entire result is stale"
        if reason:
            return SchedulerEvent(
                False,
                request.chunk_id,
                request.observation_tick,
                request.submit_tick,
                accept_tick,
                stale_prefix,
                result.inference_latency_s,
                reason,
            )

        self._current = result
        self._cursor = stale_prefix
        self.delay_history.append(stale_prefix)
        event = SchedulerEvent(
            True,
            request.chunk_id,
            request.observation_tick,
            request.submit_tick,
            accept_tick,
            stale_prefix,
            result.inference_latency_s,
        )
        self._logger.info(
            "rtc accept session=%s generation=%s chunk=%s observation_tick=%s "
            "submit_tick=%s accept_tick=%s prefix_skip=%s latency_ms=%.1f",
            request.session_id,
            request.generation_id,
            request.chunk_id,
            request.observation_tick,
            request.submit_tick,
            accept_tick,
            stale_prefix,
            result.inference_latency_s * 1000.0,
        )
        return event

    def next_step(self) -> PhysicalActionStep | None:
        """Consume one physical delta, or return None instead of replaying one."""
        self._assert_owner()
        if self.exhausted:
            return None
        assert self._current is not None
        index = self._cursor
        physical = self._current.physical_delta_chunk[index]
        self._cursor += 1
        return PhysicalActionStep(
            tcp_delta=np.array(physical[:6], dtype=np.float32, copy=True),
            gripper=float(physical[6]),
            chunk_id=self._current.request.chunk_id,
            chunk_index=index,
            tcp_reference_pose=self._current.request.observation.tcp_reference_pose,
        )

    def commit_tick(self) -> None:
        """Commit one control tick, including an explicit hold tick."""
        self._assert_owner()
        self._committed_tick += 1

    def stop_accepting(self) -> None:
        """Prevent submission and invalidate results from the current generation."""
        self._assert_owner()
        self._accepting = False
        self._generation_id += 1

    def close(self, *, wait: bool = True) -> None:
        """Idempotently invalidate work and close the inference executor."""
        if self._closed:
            return
        self._assert_owner()
        self.stop_accepting()
        if self._future is not None:
            self._future.cancel()
            self._future = None
        self._executor.shutdown(wait=wait, cancel_futures=True)
        self._closed = True

    def _make_guidance(
        self, remaining: np.ndarray, *, submitted_cursor: int
    ) -> RTCGuidance | None:
        if len(remaining) == 0:
            return None
        predicted_delay = max(max(self.delay_history, default=0), self.initial_delay)
        prefix_len = min(predicted_delay, len(remaining), self.action_horizon)
        previous = np.zeros((self.action_horizon, self.action_dim), dtype=np.float32)
        overlap = min(len(remaining), self.action_horizon)
        previous[:overlap] = remaining[:overlap, : self.action_dim]
        previous.setflags(write=False)
        return RTCGuidance(
            previous_normalized_actions=previous,
            prefix_len=prefix_len,
            guidance_weight=self.max_guidance_weight,
            decay_end=rtc_decay_endpoint(
                self.action_horizon, submitted_cursor, prefix_len
            ),
        )

    def _make_request(
        self,
        observation: PolicyObservation,
        observation_tick: int,
        *,
        guidance: RTCGuidance | None,
    ) -> InferenceRequest:
        request = InferenceRequest(
            session_id=self._session_id,
            generation_id=self._generation_id,
            chunk_id=self._next_chunk_id,
            observation_tick=observation_tick,
            submit_tick=observation_tick,
            submit_committed_tick=self._committed_tick,
            old_chunk_cursor=self._cursor,
            observation=observation,
            guidance=guidance,
            num_steps=self.num_steps,
        )
        self._next_chunk_id += 1
        return request
