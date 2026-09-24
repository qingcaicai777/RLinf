# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Standalone optimizer for delayed online residual transitions."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Callable, Sequence

import torch
from torch import nn

from .config import OnlineResidualRLConfig
from .losses import ActorLoss, CriticLoss, actor_loss, critic_loss, td_target
from .networks import ResidualActor, TwinQCritic
from .transition import OnlineResidualTransition

ActionProjector = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class LearnerMetrics:
    """Detached scalar metrics produced by one optimizer update."""

    values: dict[str, float]


@dataclass(frozen=True)
class _Batch:
    z: torch.Tensor
    proprio: torch.Tensor
    base_reference: torch.Tensor
    committed: torch.Tensor
    executed: torch.Tensor
    discounted_return: torch.Tensor
    terminal: torch.Tensor
    valid_len: torch.Tensor
    valid_mask: torch.Tensor
    human_mask: torch.Tensor
    pair_mask: torch.Tensor
    proposal: torch.Tensor
    relative_q_eligible: torch.Tensor
    next_z: torch.Tensor
    next_proprio: torch.Tensor
    next_reference: torch.Tensor
    next_committed: torch.Tensor


class OnlineResidualLearner(nn.Module):
    """Own Actor/Critic targets and apply the design's TD3-style updates."""

    def __init__(
        self,
        config: OnlineResidualRLConfig,
        *,
        actor_lr: float = 1.0e-4,
        critic_lr: float = 3.0e-4,
        max_grad_norm: float = 1.0,
        action_projector: ActionProjector | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.actor = ResidualActor(config)
        self.critic = TwinQCritic(config)
        self.target_actor = copy.deepcopy(self.actor).requires_grad_(False)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(0.9, 0.999)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(0.9, 0.999)
        )
        self.max_grad_norm = float(max_grad_norm)
        if action_projector is None:
            raise ValueError(
                "Provide the fixed physical action mapping shared with rollout."
            )
        self.action_projector = action_projector

    def critic_step(
        self,
        transitions: Sequence[OnlineResidualTransition],
        *,
        corrections: Sequence[OnlineResidualTransition] | None = None,
    ) -> LearnerMetrics:
        """Update both Critics from TD rows and any trustworthy correction pairs."""
        batch = self._collate(transitions)
        with torch.no_grad():
            active = ~batch.terminal
            target = batch.discounted_return.reshape(-1, 1).clone()
            reference = batch.next_reference[active]
            if active.any():
                target[active] = self._bootstrap_target(batch, active, reference)

        q_exec = self.critic(batch.z, batch.proprio, batch.executed)
        q_proposal = self.critic(batch.z, batch.proprio, batch.proposal)
        loss = critic_loss(
            q_exec,
            target,
            q_proposal=q_proposal if corrections is None else None,
            pair_valid=batch.pair_mask.reshape(batch.pair_mask.shape[0], -1).any(-1),
            config=self.config,
        )
        if corrections:
            pairs = self._collate(corrections)
            pair_q = self.critic(pairs.z, pairs.proprio, pairs.executed)
            proposal_q = self.critic(pairs.z, pairs.proprio, pairs.proposal)
            rank = critic_loss(
                pair_q,
                pair_q[0].detach(),
                q_proposal=proposal_q,
                pair_valid=pairs.pair_mask.flatten(1).any(-1),
                config=self.config,
            ).rank
            loss = CriticLoss(loss.td + self.config.rank_weight * rank, loss.td, rank)
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.max_grad_norm
        )
        self.critic_optimizer.step()
        return self._critic_metrics(loss, grad_norm)

    def _bootstrap_target(self, batch, active, reference):
        """Evaluate successor networks only for non-terminal rows."""
        with torch.no_grad():
            next_raw = self.target_actor(
                batch.next_z[active],
                batch.next_proprio[active],
                reference,
                pre_tanh_noise=self._target_noise(reference),
            )
            next_execution = self.action_projector(
                self.target_actor.execution_half(next_raw)
            )
            next_candidate = torch.cat(
                (batch.next_committed[active], next_execution), dim=1
            )
            next_q = self.target_critic(
                batch.next_z[active], batch.next_proprio[active], next_candidate
            )
            return td_target(
                batch.discounted_return[active],
                batch.terminal[active],
                next_q,
                valid_len=batch.valid_len[active],
                config=self.config,
            )

    def actor_step(
        self,
        transitions: Sequence[OnlineResidualTransition],
        *,
        ordinary: Sequence[OnlineResidualTransition] | None = None,
        human: Sequence[OnlineResidualTransition] | None = None,
    ) -> LearnerMetrics:
        """Update Actor with relative Q, disjoint BC, and optional smoothing."""
        separate = ordinary is not None or human is not None
        relative_count = len(transitions)
        rows = list(transitions) + list(ordinary or []) + list(human or [])
        batch = self._collate(rows)
        if separate:
            bc_valid = batch.valid_mask.clone()
            bc_valid[:relative_count] = False
            relative_valid = batch.relative_q_eligible.clone()
            relative_valid[relative_count:] = False
            batch = replace(
                batch, valid_mask=bc_valid, relative_q_eligible=relative_valid
            )
        actor_actions = self.actor(batch.z, batch.proprio, batch.base_reference)
        actor_execution = self.action_projector(
            self.actor.execution_half(actor_actions)
        )
        actor_candidate = torch.cat((batch.committed, actor_execution), dim=1)
        reference_candidate = torch.cat(
            (
                batch.committed,
                self.action_projector(
                    batch.base_reference[:, self.config.inference_horizon :]
                ),
            ),
            dim=1,
        )

        self.critic.requires_grad_(False)
        try:
            q_actor = self.critic(batch.z, batch.proprio, actor_candidate)
            with torch.no_grad():
                q_reference = self.critic(batch.z, batch.proprio, reference_candidate)
                q_proposal = self.critic(batch.z, batch.proprio, batch.proposal)
            loss = actor_loss(
                actor_actions,
                batch.base_reference,
                batch.executed,
                batch.committed,
                q_actor=q_actor,
                q_reference=q_reference,
                q_proposal=q_proposal,
                valid_mask=batch.valid_mask,
                human_mask=batch.human_mask,
                corrected=batch.pair_mask.reshape(batch.pair_mask.shape[0], -1).any(-1),
                relative_q_eligible=batch.relative_q_eligible,
                config=self.config,
                projected_execution=actor_execution,
            )
            self.actor_optimizer.zero_grad(set_to_none=True)
            loss.total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.max_grad_norm
            )
            self.actor_optimizer.step()
        finally:
            self.critic.requires_grad_(True)
        self.soft_update_targets()
        return self._actor_metrics(loss, grad_norm)

    @torch.no_grad()
    def soft_update_targets(self) -> None:
        """Polyak-update Actor and both Critic targets."""
        tau = self.config.target_tau
        for target, source in (
            (self.target_actor, self.actor),
            (self.target_critic, self.critic),
        ):
            for target_param, source_param in zip(
                target.parameters(), source.parameters(), strict=True
            ):
                target_param.lerp_(source_param, tau)

    def _target_noise(self, reference: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(reference) * self.config.target_noise_std
        return noise.clamp(
            -self.config.target_noise_clip, self.config.target_noise_clip
        )

    def _collate(self, rows: Sequence[OnlineResidualTransition]) -> _Batch:
        if not rows:
            raise ValueError("At least one transition is required.")
        device = next(self.parameters()).device

        def stack(name: str) -> torch.Tensor:
            return torch.stack([getattr(row.feature, name) for row in rows]).to(
                device=device, dtype=torch.float32
            )

        next_features = [row.next_feature or row.feature for row in rows]
        next_z = torch.stack([value.z for value in next_features]).to(
            device=device, dtype=torch.float32
        )
        next_proprio = torch.stack([value.proprio for value in next_features]).to(
            device=device, dtype=torch.float32
        )
        next_reference = torch.stack(
            [value.base_reference for value in next_features]
        ).to(device=device, dtype=torch.float32)
        next_committed = torch.stack([value.committed for value in next_features]).to(
            device=device, dtype=torch.float32
        )
        executed = torch.stack([row.executed for row in rows]).to(
            device=device, dtype=torch.float32
        )
        valid_len = torch.tensor(
            [row.valid_len for row in rows], device=device, dtype=torch.long
        )
        frame_valid = (
            torch.arange(self.config.actor_horizon, device=device)[None, :]
            < valid_len[:, None]
        )
        valid_mask = frame_valid[..., None].expand_as(executed)
        human_masks = []
        pair_masks = []
        proposals = []
        for row, executed_row in zip(rows, executed, strict=True):
            if row.correction is None:
                human_masks.append(torch.zeros_like(executed_row, dtype=torch.bool))
                pair_masks.append(torch.zeros_like(executed_row, dtype=torch.bool))
                proposals.append(executed_row)
            else:
                human_masks.append(row.correction.human_mask.to(device=device))
                pair_masks.append(row.correction.pair_mask.to(device=device))
                proposals.append(row.correction.proposal(executed_row))
        return _Batch(
            z=stack("z"),
            proprio=stack("proprio"),
            base_reference=stack("base_reference"),
            committed=stack("committed"),
            executed=executed,
            discounted_return=torch.stack([row.discounted_return for row in rows]).to(
                device=device, dtype=torch.float32
            ),
            terminal=torch.tensor(
                [row.terminal for row in rows], device=device, dtype=torch.bool
            ),
            valid_len=valid_len,
            valid_mask=valid_mask,
            human_mask=torch.stack(human_masks),
            pair_mask=torch.stack(pair_masks),
            proposal=torch.stack(proposals),
            relative_q_eligible=torch.tensor(
                [row.relative_q_eligible for row in rows],
                device=device,
                dtype=torch.bool,
            ),
            next_z=next_z,
            next_proprio=next_proprio,
            next_reference=next_reference,
            next_committed=next_committed,
        )

    @staticmethod
    def _critic_metrics(loss: CriticLoss, grad_norm: torch.Tensor) -> LearnerMetrics:
        return LearnerMetrics(
            {
                "critic/total": loss.total.detach().item(),
                "critic/td": loss.td.detach().item(),
                "critic/rank": loss.rank.detach().item(),
                "critic/grad_norm": torch.as_tensor(grad_norm).detach().item(),
            }
        )

    @staticmethod
    def _actor_metrics(loss: ActorLoss, grad_norm: torch.Tensor) -> LearnerMetrics:
        return LearnerMetrics(
            {
                "actor/total": loss.total.detach().item(),
                "actor/relative_q": loss.relative_q.detach().item(),
                "actor/ordinary_bc": loss.ordinary_bc.detach().item(),
                "actor/human_bc": loss.human_bc.detach().item(),
                "actor/smooth": loss.smooth.detach().item(),
                "actor/grad_norm": torch.as_tensor(grad_norm).detach().item(),
            }
        )
