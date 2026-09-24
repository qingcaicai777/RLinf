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

"""Typed messages exchanged by the RTC control and inference threads."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np


def _readonly_copy(
    value: np.ndarray, *, dtype: np.dtype | type | None = None
) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PolicyObservation:
    """Immutable policy input captured at one control tick."""

    state: np.ndarray
    images: Mapping[str, np.ndarray]
    prompt: str
    tcp_reference_pose: np.ndarray | None = None

    @classmethod
    def snapshot(
        cls,
        state: np.ndarray,
        images: Mapping[str, np.ndarray],
        prompt: str,
        tcp_reference_pose: np.ndarray | None = None,
    ) -> PolicyObservation:
        """Copy mutable robot data into an immutable inference message."""
        return cls(
            state=_readonly_copy(state, dtype=np.float32),
            images=MappingProxyType(
                {key: _readonly_copy(image) for key, image in images.items()}
            ),
            prompt=str(prompt),
            tcp_reference_pose=(
                None
                if tcp_reference_pose is None
                else _readonly_copy(tcp_reference_pose, dtype=np.float64)
            ),
        )


@dataclass(frozen=True)
class RTCGuidance:
    """Previous normalized model actions used by speed-field guidance."""

    previous_normalized_actions: np.ndarray
    prefix_len: int
    guidance_weight: float
    decay_end: int


@dataclass(frozen=True)
class InferenceRequest:
    """Immutable inference work item created by the control thread."""

    session_id: int
    generation_id: int
    chunk_id: int
    observation_tick: int
    submit_tick: int
    submit_committed_tick: int
    old_chunk_cursor: int
    observation: PolicyObservation
    guidance: RTCGuidance | None
    num_steps: int


@dataclass(frozen=True)
class PolicyResult:
    """One PI0.5 chunk in normalized and physical action domains."""

    normalized_raw_chunk: np.ndarray
    physical_delta_chunk: np.ndarray
    inference_latency_s: float
    request: InferenceRequest


@dataclass(frozen=True)
class PhysicalActionStep:
    """One physical TCP delta and independent gripper command."""

    tcp_delta: np.ndarray
    gripper: float
    chunk_id: int
    chunk_index: int
    tcp_reference_pose: np.ndarray | None = None


@dataclass(frozen=True)
class SchedulerEvent:
    """Metadata describing an accepted or rejected asynchronous result."""

    accepted: bool
    chunk_id: int
    observation_tick: int
    submit_tick: int
    accept_tick: int
    stale_prefix: int
    inference_latency_s: float
    reason: str = ""
