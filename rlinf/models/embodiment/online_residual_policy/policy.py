# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""BasePolicy adapter for bounded online residual inference."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from rlinf.algorithms.online_residual_rl import (
    ActionRepresentation,
    OnlineResidualRLConfig,
    ResidualActor,
    TwinQCritic,
)
from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType


class OnlineResidualPolicy(nn.Module, BasePolicy):
    """Generate only the new ``n`` actions while retaining each full candidate."""

    def __init__(self, config: OnlineResidualRLConfig) -> None:
        super().__init__()
        self.config = config
        self.actor = ResidualActor(config)
        self.q_head = TwinQCritic(config)
        self.action_dim = config.action_dim
        self.num_action_chunks = config.inference_horizon
        self.ref_num_action_chunks = config.actor_horizon
        self.torch_compile_enabled = False

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "OnlineResidualPolicy":
        """Translate static Hydra values into the validated shared contract."""
        def select(name: str, default: Any) -> Any:
            return OmegaConf.select(cfg, name, default=default)

        contract = OnlineResidualRLConfig(
            action_representation=ActionRepresentation(
                select("action_representation", "eef_delta")
            ),
            proprio_dim=int(select("proprio_dim", 8)),
            z_dim=int(select("z_dim", 2048)),
            inference_horizon=int(select("num_action_chunks", 6)),
            actor_horizon=int(select("ref_num_action_chunks", 12)),
            td_horizon=int(select("td_horizon", 12)),
            hidden_dim=int(select("mlp_hidden_dim", 512)),
            num_hidden_layers=int(select("mlp_num_hidden_layers", 3)),
            residual_scale=float(select("residual_scale", 0.05)),
            gamma=float(select("gamma", 0.99)),
            rank_margin=float(select("rank_margin", 0.05)),
            rank_weight=float(select("rank_weight", 50.0)),
            actor_margin=float(select("actor_margin", 0.0)),
            actor_kappa=float(select("actor_kappa", 0.05)),
            relative_q_weight=float(select("relative_q_weight", 1.0)),
            ordinary_bc_weight=float(select("ordinary_bc_weight", 5.0)),
            human_bc_multiplier=float(select("human_bc_multiplier", 5.0)),
            smooth_weight=float(select("smooth_weight", 0.0)),
            rollout_noise_std=float(select("rollout_noise_std", 0.04)),
            target_noise_std=float(select("target_noise_std", 0.04)),
            target_noise_clip=float(select("target_noise_clip", 0.10)),
            target_tau=float(select("target_tau", 0.005)),
            critic_updates_per_transition=int(
                select("critic_updates_per_transition", 5)
            ),
            critic_updates_per_actor_update=int(
                select("critic_updates_per_actor_update", 5)
            ),
        )
        configured_action_dim = int(select("action_dim", contract.action_dim))
        if configured_action_dim != contract.action_dim:
            raise ValueError(
                f"action_representation={contract.action_representation.value!r} "
                f"requires action_dim={contract.action_dim}, got {configured_action_dim}."
            )
        return cls(contract)

    def preprocess_env_obs(self, env_obs: dict[str, Any]) -> dict[str, Any]:
        """Move tensor observations to the policy device."""
        device = next(self.parameters()).device
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in env_obs.items()
        }

    def _features(
        self, obs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = obs["z_rl"].reshape(obs["z_rl"].shape[0], -1)
        proprio = obs["proprio"].reshape(obs["proprio"].shape[0], -1)
        reference = obs.get("base_ref", obs.get("ref_chunk"))
        if reference is None:
            raise KeyError("Online residual observations require base_ref or ref_chunk.")
        reference = reference.reshape(
            reference.shape[0], self.config.actor_horizon, self.config.action_dim
        )
        committed = obs.get("committed")
        if committed is None:
            raise KeyError("Online residual observations require committed actions.")
        committed = committed.reshape(
            committed.shape[0], self.config.inference_horizon, self.config.action_dim
        )
        return z, proprio, reference, committed

    def residual_forward(
        self,
        obs: dict[str, torch.Tensor],
        *,
        deterministic: bool,
        target_noise: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the full candidate and its executable ``[n,2n)`` half."""
        z, proprio, reference, committed = self._features(obs)
        noise = None
        std = 0.0
        if target_noise:
            std = self.config.target_noise_std
        elif not deterministic:
            std = self.config.rollout_noise_std
        if std > 0.0:
            noise = torch.randn_like(reference) * std
            if target_noise:
                noise = noise.clamp(
                    -self.config.target_noise_clip, self.config.target_noise_clip
                )
        full = self.actor(z, proprio, reference, pre_tanh_noise=noise)
        candidate = self.actor.candidate(committed, full)
        return full, candidate[:, self.config.inference_horizon :]

    def q_forward(
        self, obs: dict[str, torch.Tensor], candidate: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate a full action with the observation's RL state."""
        z, proprio, _, _ = self._features(obs)
        candidate = candidate.reshape(
            candidate.shape[0], self.config.actor_horizon, self.config.action_dim
        )
        return self.q_head(z, proprio, candidate)

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        """Support the off-policy forward types used by RLinf workers."""
        obs = self.preprocess_env_obs(kwargs["obs"])
        if forward_type == ForwardType.SAC:
            _, execution = self.residual_forward(
                obs,
                deterministic=bool(kwargs.get("deterministic", False)),
                target_noise=bool(kwargs.get("target_noise", False)),
            )
            flat = execution.reshape(execution.shape[0], -1)
            return flat, torch.zeros_like(flat), None
        if forward_type == ForwardType.SAC_Q:
            q1, q2 = self.q_forward(obs, kwargs["actions"])
            return torch.cat((q1, q2), dim=-1)
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(f"Unsupported forward_type: {forward_type}")

    def default_forward(self, **kwargs):
        raise NotImplementedError("OnlineResidualPolicy uses residual/off-policy forwards.")

    @torch.inference_mode()
    def predict_action_batch(
        self,
        env_obs,
        calculate_logprobs=True,
        calculate_values=True,
        return_obs=True,
        mode="train",
        **kwargs,
    ):
        """Return the new execution block and retain the full proposal for replay."""
        del calculate_logprobs, calculate_values, kwargs
        obs = self.preprocess_env_obs(env_obs)
        full, execution = self.residual_forward(obs, deterministic=(mode == "eval"))
        _, _, _, committed = self._features(obs)
        proposal = self.actor.candidate(committed, full)
        forward_inputs: dict[str, Any] = {
            "action": execution.reshape(execution.shape[0], -1),
            "model_action": full.reshape(full.shape[0], -1),
            "policy_proposal": proposal,
        }
        if return_obs:
            forward_inputs.update(obs)
        result = {
            "prev_logprobs": torch.zeros_like(execution[..., 0]),
            "prev_values": torch.zeros_like(execution[..., :1]),
            "forward_inputs": forward_inputs,
        }
        return execution, result
