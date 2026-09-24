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

"""Standard OpenPI/JAX PI0.5 policy adapter for RTC inference."""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .types import InferenceRequest, PolicyObservation, PolicyResult


class InferencePolicy(Protocol):
    """OpenPI policy surface used by the RTC adapter."""

    def infer(
        self,
        obs: dict[str, Any],
        *,
        sample_kwargs: dict[str, Any] | None = None,
        return_raw_actions: bool = False,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class PI05PolicySpec:
    """Checkpoint and observation contract for standard PI0.5 inference."""

    config_name: str
    checkpoint_path: Path
    policy_camera_keys: tuple[str, str]
    state_key: str = "observation/state"
    state_dim: int = 8
    action_horizon: int = 12
    physical_action_dim: int = 7

    def validate(self) -> None:
        """Reject incompatible deployment semantics before hardware activation."""
        if not self.config_name:
            raise ValueError("config_name must not be empty.")
        if len(set(self.policy_camera_keys)) != 2:
            raise ValueError("policy_camera_keys must contain two distinct keys.")
        if self.state_dim != 8:
            raise ValueError(f"PI0.5 Flexiv state_dim must be 8, got {self.state_dim}.")
        if self.action_horizon <= 0 or self.physical_action_dim != 7:
            raise ValueError("Expected a positive horizon and 7D physical actions.")


class PI05PolicyAdapter:
    """Load PI0.5 and preserve raw and postprocessed chunks separately."""

    def __init__(self, policy: InferencePolicy, spec: PI05PolicySpec) -> None:
        spec.validate()
        self._policy = policy
        self.spec = spec

    @classmethod
    def load(cls, spec: PI05PolicySpec) -> PI05PolicyAdapter:
        """Load an RLinf OpenPI config and a standard JAX SFT checkpoint."""
        spec.validate()
        from openpi.models import pi0
        from openpi.policies import policy as openpi_policy
        from openpi.policies import policy_config

        from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

        infer_parameters = inspect.signature(openpi_policy.Policy.infer).parameters
        sample_parameters = inspect.signature(pi0.Pi0.sample_actions).parameters
        required_infer = {"sample_kwargs", "return_raw_actions"}
        required_sample = {
            "prev_actions",
            "prefix_len",
            "rtc_guidance_weight",
            "rtc_decay_end",
        }
        if not required_infer.issubset(
            infer_parameters
        ) or not required_sample.issubset(sample_parameters):
            raise RuntimeError(
                "Installed OpenPI does not provide ordinary RTC speed-field inference. "
                "Install the compatible checkout selected by OPENPI_RTC_PATH."
            )

        train_config = get_openpi_config(spec.config_name)
        model = train_config.model
        if not bool(getattr(model, "pi05", False)):
            raise ValueError(f"Config {spec.config_name!r} is not PI0.5.")
        if bool(getattr(model, "legato_enabled", False)):
            raise ValueError("Standard RTC does not support Legato guidance.")
        if bool(getattr(model, "use_rlt", False)):
            raise ValueError("Standard SFT RTC does not support RL Token inference.")
        if int(model.action_horizon) != spec.action_horizon:
            raise ValueError(
                "Config action horizon is "
                f"{model.action_horizon}, expected {spec.action_horizon}."
            )
        output_dim = int(
            getattr(train_config.data, "output_action_dim", model.action_dim)
        )
        if output_dim != spec.physical_action_dim:
            raise ValueError(
                f"Config physical action dimension is {output_dim}, "
                f"expected {spec.physical_action_dim}."
            )
        checkpoint_path = spec.checkpoint_path.expanduser()
        if (checkpoint_path / "model.safetensors").exists():
            raise ValueError(
                "Standard Flexiv RTC requires a JAX checkpoint, not model.safetensors."
            )
        if not (checkpoint_path / "params").exists():
            raise ValueError(
                f"JAX checkpoint params were not found under {checkpoint_path}."
            )
        policy = policy_config.create_trained_policy(
            train_config,
            checkpoint_path,
            sample_kwargs={},
        )
        return cls(policy, spec)

    def infer(self, request: InferenceRequest) -> PolicyResult:
        """Run one immutable request on the inference thread."""
        sample_kwargs: dict[str, Any] = {
            "num_steps": np.asarray(request.num_steps, dtype=np.int32)
        }
        if request.guidance is not None:
            guidance = request.guidance
            sample_kwargs.update(
                prev_actions=np.asarray(
                    guidance.previous_normalized_actions, dtype=np.float32
                ),
                prefix_len=np.asarray(guidance.prefix_len, dtype=np.int32),
                rtc_guidance_weight=np.asarray(
                    guidance.guidance_weight, dtype=np.float32
                ),
                rtc_decay_end=np.asarray(guidance.decay_end, dtype=np.int32),
            )
        started = time.monotonic()
        output = self._policy.infer(
            self.build_policy_input(request.observation),
            sample_kwargs=sample_kwargs,
            return_raw_actions=True,
        )
        latency = time.monotonic() - started
        raw = np.asarray(output["raw_actions"], dtype=np.float32)
        physical = np.asarray(output["actions"], dtype=np.float32)
        expected = (self.spec.action_horizon, self.spec.physical_action_dim)
        if raw.ndim != 2 or raw.shape[0] != expected[0] or raw.shape[1] < expected[1]:
            raise RuntimeError(
                "Normalized raw chunk must have shape "
                f"({expected[0]}, >= {expected[1]}), got {raw.shape}."
            )
        if physical.shape != expected:
            raise RuntimeError(
                f"Physical delta chunk must have shape {expected}, got {physical.shape}."
            )
        if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(physical)):
            raise RuntimeError("Policy output contains NaN or Inf.")
        return PolicyResult(
            normalized_raw_chunk=np.array(raw, copy=True),
            physical_delta_chunk=np.array(physical, copy=True),
            inference_latency_s=latency,
            request=request,
        )

    def build_policy_input(self, observation: PolicyObservation) -> dict[str, Any]:
        """Map explicit training keys without task-specific preprocessing."""
        state = np.asarray(observation.state, dtype=np.float32)
        if state.shape != (self.spec.state_dim,) or not np.all(np.isfinite(state)):
            raise ValueError(
                f"Policy state must be finite shape ({self.spec.state_dim},), "
                f"got {state.shape}."
            )
        result: dict[str, Any] = {
            self.spec.state_key: state,
            "prompt": observation.prompt,
        }
        for key in self.spec.policy_camera_keys:
            if key not in observation.images:
                raise KeyError(
                    f"Missing policy camera {key!r}; "
                    f"available={sorted(observation.images)}."
                )
            image = np.asarray(observation.images[key])
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(
                    f"Camera {key!r} must be HxWx3 RGB, got {image.shape}."
                )
            result[key] = image
        return result
