# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Validated configuration for the online residual RL contract."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ActionRepresentation(str, Enum):
    """Action layouts supported by the residual policy."""

    EEF_DELTA = "eef_delta"
    JOINT_ABSOLUTE = "joint_absolute"

    @property
    def action_dim(self) -> int:
        """Return the physical action width for this representation."""
        return 7 if self is ActionRepresentation.EEF_DELTA else 8

    @property
    def smooth_dims(self) -> int:
        """Return the non-gripper dimensions used by smoothness loss."""
        return 6 if self is ActionRepresentation.EEF_DELTA else 7


@dataclass(frozen=True)
class OnlineResidualRLConfig:
    """Shape and optimization constants shared by rollout and learner."""

    action_representation: ActionRepresentation = ActionRepresentation.EEF_DELTA
    proprio_dim: int = 8
    z_dim: int = 2048
    inference_horizon: int = 6
    actor_horizon: int = 12
    td_horizon: int = 12
    hidden_dim: int = 512
    num_hidden_layers: int = 3
    residual_scale: float = 0.05
    gamma: float = 0.99
    rank_margin: float = 0.05
    rank_weight: float = 50.0
    actor_margin: float = 0.0
    actor_kappa: float = 0.05
    relative_q_weight: float = 1.0
    ordinary_bc_weight: float = 5.0
    human_bc_multiplier: float = 5.0
    smooth_weight: float = 0.0
    rollout_noise_std: float = 0.04
    target_noise_std: float = 0.04
    target_noise_clip: float = 0.10
    target_tau: float = 0.005
    critic_updates_per_transition: int = 5
    critic_updates_per_actor_update: int = 5

    def __post_init__(self) -> None:
        representation = ActionRepresentation(self.action_representation)
        object.__setattr__(self, "action_representation", representation)
        if self.actor_horizon != 2 * self.inference_horizon:
            raise ValueError("actor_horizon must equal 2 * inference_horizon.")
        if self.td_horizon != self.actor_horizon:
            raise ValueError("td_horizon must equal actor_horizon.")
        if self.num_hidden_layers != 3:
            raise ValueError(
                "The online residual Actor/Critic contract uses 3 hidden layers."
            )
        if self.residual_scale <= 0.0:
            raise ValueError("residual_scale must be positive.")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1].")
        if self.actor_kappa <= 0.0:
            raise ValueError("actor_kappa must be positive.")
        if self.rank_weight < 0.0 or self.smooth_weight < 0.0:
            raise ValueError("Loss weights must be nonnegative.")
        if self.critic_updates_per_transition <= 0:
            raise ValueError("critic_updates_per_transition must be positive.")
        if self.critic_updates_per_actor_update <= 0:
            raise ValueError("critic_updates_per_actor_update must be positive.")

    @property
    def action_dim(self) -> int:
        """Return 7 for EEF delta and 8 for absolute joint control."""
        return self.action_representation.action_dim

    @property
    def actor_input_dim(self) -> int:
        """Return ``z + proprio + flattened base reference`` width."""
        return self.z_dim + self.proprio_dim + self.actor_horizon * self.action_dim

    @property
    def actor_output_dim(self) -> int:
        """Return the flattened full-horizon action width."""
        return self.actor_horizon * self.action_dim

    @property
    def critic_input_dim(self) -> int:
        """Return ``z + proprio + flattened candidate`` width."""
        return self.z_dim + self.proprio_dim + self.actor_horizon * self.action_dim

    @property
    def human_bc_weight(self) -> float:
        """Return the explicit five-times-human BC coefficient."""
        return self.ordinary_bc_weight * self.human_bc_multiplier

    @property
    def lambda_rank(self) -> float:
        """Expose the design-document name for the ranking coefficient."""
        return self.rank_weight
