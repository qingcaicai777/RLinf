# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Residual Actor and twin Critic networks from the online design."""

from __future__ import annotations

import torch
from torch import nn

from .config import OnlineResidualRLConfig


class ResidualActor(nn.Module):
    """Generate a bounded residual around a full base-reference chunk."""

    def __init__(self, config: OnlineResidualRLConfig) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        width = config.actor_input_dim
        for _ in range(config.num_hidden_layers):
            layers.extend((nn.Linear(width, config.hidden_dim), nn.Tanh()))
            width = config.hidden_dim
        output = nn.Linear(width, config.actor_output_dim)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.mlp = nn.Sequential(*layers)

    def raw_output(
        self,
        z: torch.Tensor,
        proprio: torch.Tensor,
        base_reference: torch.Tensor,
    ) -> torch.Tensor:
        """Return pre-tanh residual logits for the full ``2n`` horizon."""
        self._validate_inputs(z, proprio, base_reference)
        inputs = torch.cat(
            (z, proprio, base_reference.reshape(base_reference.shape[0], -1)), dim=-1
        )
        return self.mlp(inputs).reshape(
            -1, self.config.actor_horizon, self.config.action_dim
        )

    def forward(
        self,
        z: torch.Tensor,
        proprio: torch.Tensor,
        base_reference: torch.Tensor,
        *,
        pre_tanh_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``base_reference + scale * tanh(logits + noise)``."""
        logits = self.raw_output(z, proprio, base_reference)
        if pre_tanh_noise is not None:
            if pre_tanh_noise.shape != logits.shape:
                raise ValueError(
                    "pre_tanh_noise must match actor logits, got "
                    f"{tuple(pre_tanh_noise.shape)} and {tuple(logits.shape)}."
                )
            logits = logits + pre_tanh_noise
        residual = self.config.residual_scale * torch.tanh(logits)
        return base_reference + residual

    def execution_half(self, actions: torch.Tensor) -> torch.Tensor:
        """Select the ``[n, 2n)`` action block that a decision may execute."""
        n = self.config.inference_horizon
        return actions[:, n : 2 * n]

    def candidate(self, committed: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Combine an immutable committed prefix with the new execution half."""
        expected = (
            actions.shape[0],
            self.config.inference_horizon,
            self.config.action_dim,
        )
        if tuple(committed.shape) != expected:
            raise ValueError(f"committed must have shape {expected}.")
        return torch.cat((committed.detach(), self.execution_half(actions)), dim=1)

    def _validate_inputs(
        self,
        z: torch.Tensor,
        proprio: torch.Tensor,
        base_reference: torch.Tensor,
    ) -> None:
        expected_ref = (self.config.actor_horizon, self.config.action_dim)
        if z.ndim != 2 or z.shape[-1] != self.config.z_dim:
            raise ValueError(f"z must have shape [B,{self.config.z_dim}].")
        if proprio.ndim != 2 or proprio.shape[-1] != self.config.proprio_dim:
            raise ValueError(f"proprio must have shape [B,{self.config.proprio_dim}].")
        if base_reference.ndim != 3 or tuple(base_reference.shape[1:]) != expected_ref:
            raise ValueError(
                f"base_reference must have shape [B,{expected_ref[0]},{expected_ref[1]}]."
            )


class QNetwork(nn.Module):
    """Evaluate one full candidate with three LayerNorm-Tanh hidden layers."""

    def __init__(self, config: OnlineResidualRLConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = config.critic_input_dim
        for _ in range(config.num_hidden_layers):
            layers.extend(
                (
                    nn.Linear(width, config.hidden_dim),
                    nn.LayerNorm(config.hidden_dim),
                    nn.Tanh(),
                )
            )
            width = config.hidden_dim
        layers.append(nn.Linear(width, 1))
        self.config = config
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        z: torch.Tensor,
        proprio: torch.Tensor,
        candidate: torch.Tensor,
    ) -> torch.Tensor:
        expected = (self.config.actor_horizon, self.config.action_dim)
        if candidate.ndim != 3 or tuple(candidate.shape[1:]) != expected:
            raise ValueError(
                f"candidate must have shape [B,{expected[0]},{expected[1]}]."
            )
        inputs = torch.cat(
            (z, proprio, candidate.reshape(candidate.shape[0], -1)), dim=-1
        )
        return self.mlp(inputs)


class TwinQCritic(nn.Module):
    """Two independently initialized scalar Q networks."""

    def __init__(self, config: OnlineResidualRLConfig) -> None:
        super().__init__()
        self.q1 = QNetwork(config)
        self.q2 = QNetwork(config)

    def forward(
        self,
        z: torch.Tensor,
        proprio: torch.Tensor,
        candidate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(Q1, Q2)`` without output squashing."""
        return self.q1(z, proprio, candidate), self.q2(z, proprio, candidate)
