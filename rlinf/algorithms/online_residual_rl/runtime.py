# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Inference-boundary state for overlapping residual action windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .config import OnlineResidualRLConfig
from .networks import ResidualActor

ActionDecoder = Callable[[torch.Tensor], torch.Tensor]
ActionEncoder = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class ResidualDecision:
    """Immutable tensors captured for one inference request and replay row."""

    z: torch.Tensor
    proprio: torch.Tensor
    base_reference: torch.Tensor
    committed: torch.Tensor | None
    actor_output: torch.Tensor
    proposal: torch.Tensor | None
    normalized_execution: torch.Tensor
    decoded_execution: torch.Tensor


class OnlineResidualController:
    """Run a PyTorch RLT/Actor sidecar around an accelerated PI0.5 base."""

    def __init__(
        self,
        config: OnlineResidualRLConfig,
        *,
        rlt_encoder: torch.nn.Module,
        actor: ResidualActor,
        action_decoder: ActionDecoder | None = None,
        action_encoder: ActionEncoder | None = None,
    ) -> None:
        self.config = config
        self.rlt_encoder = rlt_encoder
        self.actor = actor
        self.action_decoder = action_decoder or (lambda actions: actions)
        self.action_encoder = action_encoder or (lambda actions: actions)
        self._pending_proposal: torch.Tensor | None = None

    def reset(self) -> None:
        """Discard cross-episode committed actions and pending receipts."""
        self._pending_proposal = None

    @torch.inference_mode()
    def decide(
        self,
        *,
        prefix_hidden_states: torch.Tensor,
        proprio: torch.Tensor,
        base_reference: torch.Tensor,
        deterministic: bool,
        committed: torch.Tensor | None = None,
        prefix_mask: torch.Tensor | None = None,
    ) -> ResidualDecision:
        """Predict independently; optionally construct a full Critic candidate."""
        if base_reference.ndim != 3:
            raise ValueError(
                "base_reference must have shape [B, actor_horizon, action_dim]."
            )
        expected = (self.config.actor_horizon, self.config.action_dim)
        if tuple(base_reference.shape[1:]) != expected:
            raise ValueError(f"base_reference must have trailing shape {expected}.")
        if proprio.shape != (base_reference.shape[0], self.config.proprio_dim):
            raise ValueError(f"proprio must have shape [B,{self.config.proprio_dim}].")
        z = (
            self.rlt_encoder.encode_flat(prefix_hidden_states)
            if prefix_mask is None
            else self.rlt_encoder.encode_flat(prefix_hidden_states, prefix_mask)
        )
        if committed is not None:
            committed = committed.detach().clone().to(base_reference)
            if not torch.isfinite(committed).all():
                raise ValueError(
                    "Committed queue must contain finite normalized commands."
                )
        noise = None
        if not deterministic and self.config.rollout_noise_std > 0.0:
            noise = torch.randn_like(base_reference) * self.config.rollout_noise_std
            noise[:, : self.config.inference_horizon] = 0
        actor_output = self.actor(
            z,
            proprio,
            base_reference,
            pre_tanh_noise=noise,
        )
        proposal = (
            None if committed is None else self.actor.candidate(committed, actor_output)
        )
        normalized_execution = actor_output[:, self.config.inference_horizon :]
        decoded_execution = self.action_decoder(normalized_execution)
        self._pending_proposal = normalized_execution.detach().clone()
        return ResidualDecision(
            z=z,
            proprio=proprio,
            base_reference=base_reference,
            committed=committed,
            actor_output=actor_output,
            proposal=proposal,
            normalized_execution=normalized_execution,
            decoded_execution=decoded_execution,
        )

    @property
    def pending_proposal(self) -> torch.Tensor | None:
        """Return the pre-intervention block awaiting execution receipts."""
        return self._pending_proposal
