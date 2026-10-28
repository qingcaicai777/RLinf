# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Non-Module mixin: env worker dict ↔ Observation ↔ decoded env actions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Sequence

import numpy as np
import torch

from rlinf.models.embodiment.openpi_rlinf.modules.model import Observation
from rlinf.models.embodiment.openpi_rlinf.transforms.apply import (
    apply_input_transform,
    apply_output_transform,
)
from rlinf.models.embodiment.openpi_rlinf.transforms.env import repack_env_obs


class EnvIO:
    """Shared env I/O for task subclasses that inherit :class:`Pi0`.

    Not an ``nn.Module`` — mix in *before* ``Pi0`` so ``super().__init__`` still
    reaches the network constructor.
    """

    _input_transform_fn = None
    _output_transform_fn = None
    _action_norm_stats = None
    _action_use_quantiles = False

    def setup_transforms(
        self,
        transforms: Sequence,
        output_transforms: Sequence,
    ) -> None:
        """Install the openpi.transforms input/output pipelines."""
        from openpi.transforms import compose

        self._input_transform_fn = compose(transforms)
        self._output_transform_fn = compose(output_transforms)
        self._action_norm_stats = None
        self._action_use_quantiles = False
        for transform in transforms:
            if type(transform).__name__ != "Normalize":
                continue
            stats = getattr(transform, "norm_stats", None)
            if isinstance(stats, Mapping):
                self._action_norm_stats = stats.get("actions")
            self._action_use_quantiles = bool(
                getattr(transform, "use_quantiles", False)
            )
            break

    def _ensure_transforms(self) -> None:
        if self._input_transform_fn is None or self._output_transform_fn is None:
            raise RuntimeError(
                f"{type(self).__name__}.setup_transforms(...) must be called "
                "after construction (the factory does this); the openpi "
                "transforms pipeline is not yet installed."
            )

    def _select_configured_state(self, states):
        """Select a configured subset of the raw env state (openpi parity)."""
        indices = self.state_indices
        if not indices:
            return states

        if hasattr(states, "shape"):
            state_dim = states.shape[-1]
        else:
            state_dim = np.asarray(states).shape[-1]
        if state_dim == len(indices):
            return states
        if state_dim <= max(indices):
            raise ValueError(
                f"Cannot select state_indices={indices} from state dim {state_dim}."
            )

        if torch.is_tensor(states):
            index_tensor = torch.as_tensor(indices, device=states.device)
            return states.index_select(-1, index_tensor)
        return np.asarray(states)[..., indices]

    def input_transform(self, obs: dict, transpose: bool = False) -> dict:
        self._ensure_transforms()
        return apply_input_transform(self._input_transform_fn, obs, transpose=transpose)

    def output_transform(self, outputs: dict) -> dict:
        self._ensure_transforms()
        return apply_output_transform(
            self._output_transform_fn,
            outputs,
            action_chunk=self.action_chunk,
        )

    def decode_actions(
        self, model_actions: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        """Run the output transform and return env-space actions on this device."""
        env_outputs = self.output_transform({"actions": model_actions, "state": state})
        return env_outputs["actions"].to(device=self.device, dtype=torch.float32)

    def normalize_rlt_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Normalize physical dataset actions for Stage-2 residual learning."""
        stats = self._action_norm_stats
        if stats is None:
            raise ValueError(
                "The OpenPI checkpoint has no action normalization statistics."
            )
        source = actions.detach().to(torch.float32)
        width = source.shape[-1]

        def as_tensor(value):
            return torch.as_tensor(
                value[..., :width], device=source.device, dtype=source.dtype
            )

        if self._action_use_quantiles:
            q01 = getattr(stats, "q01", None)
            q99 = getattr(stats, "q99", None)
            if q01 is None or q99 is None:
                raise ValueError("Quantile action statistics require q01 and q99.")
            low, high = as_tensor(q01), as_tensor(q99)
            return (source - low) / (high - low + 1e-6) * 2.0 - 1.0
        mean, std = as_tensor(stats.mean), as_tensor(stats.std)
        return (source - mean) / (std + 1e-6)

    def denormalize_rlt_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Decode normalized dataset actions using the frozen Stage1 statistics."""
        stats = self._action_norm_stats
        if stats is None:
            raise ValueError(
                "The OpenPI checkpoint has no action normalization statistics."
            )
        source = actions.detach().to(torch.float32)
        width = source.shape[-1]

        def as_tensor(value):
            return torch.as_tensor(
                value[..., :width], device=source.device, dtype=source.dtype
            )

        if self._action_use_quantiles:
            q01 = getattr(stats, "q01", None)
            q99 = getattr(stats, "q99", None)
            if q01 is None or q99 is None:
                raise ValueError("Quantile action statistics require q01 and q99.")
            low, high = as_tensor(q01), as_tensor(q99)
            return (source + 1.0) * (high - low + 1e-6) / 2.0 + low
        mean, std = as_tensor(stats.mean), as_tensor(stats.std)
        return source * (std + 1e-6) + mean

    def decode_rlt_actions(
        self, normalized_actions: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        """Apply the checkpoint's deterministic output map to normalized actions."""
        model_width = int(self.config.action_dim)
        if normalized_actions.shape[-1] > model_width:
            raise ValueError(
                f"RLT action width {normalized_actions.shape[-1]} exceeds model width "
                f"{model_width}."
            )
        padded = torch.zeros(
            *normalized_actions.shape[:-1],
            model_width,
            device=normalized_actions.device,
            dtype=normalized_actions.dtype,
        )
        padded[..., : normalized_actions.shape[-1]] = normalized_actions
        return self.decode_actions(padded, state)

    def _observation_dict_to_device(self, processed: dict) -> Observation:
        """Convert a per-key dict into a device-resident Observation."""
        device = self.device
        obs = Observation.from_dict(processed)

        def _move(x):
            return x.to(device) if isinstance(x, torch.Tensor) else x

        def _move_state(x):
            return (
                x.to(device=device, dtype=torch.float32)
                if isinstance(x, torch.Tensor)
                else x
            )

        return Observation(
            images={k: _move(v) for k, v in obs.images.items()},
            image_masks={k: _move(v) for k, v in obs.image_masks.items()},
            state=_move_state(obs.state),
            tokenized_prompt=_move(obs.tokenized_prompt),
            tokenized_prompt_mask=_move(obs.tokenized_prompt_mask),
            token_ar_mask=_move(obs.token_ar_mask),
            token_loss_mask=_move(obs.token_loss_mask),
            pcd_xyz=_move(obs.pcd_xyz),
        )

    def env_obs_to_observation(self, env_obs: dict[str, Any]) -> Observation:
        """Repack env obs through the transform pipeline onto this device."""
        self._ensure_transforms()
        repacked = repack_env_obs(
            self.config_name,
            env_obs,
            select_state=self._select_configured_state,
        )
        processed = self.input_transform(repacked, transpose=False)
        return self._observation_dict_to_device(processed)

    def _predict_eval(
        self,
        observation: Observation,
        *,
        noise: torch.Tensor | None = None,
        rng: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Deterministic Euler ODE sampler shared by eval / RL eval / DAgger."""
        model_actions = self.sample_actions(
            observation, num_steps=self.num_steps, noise=noise, rng=rng
        )
        actions = self.decode_actions(model_actions, observation.state)
        B = actions.shape[0]
        result = {
            "prev_logprobs": None,
            "prev_values": None,
            "forward_inputs": {
                "action": actions.reshape(B, -1).contiguous(),
                "model_action": model_actions.reshape(B, -1).contiguous(),
            },
            "model_actions": model_actions,
        }
        return actions, result
