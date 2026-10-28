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

from .action_mapping import AffineActionMapping
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
class _StateActions:
    z: torch.Tensor
    proprio: torch.Tensor
    executed: torch.Tensor


@dataclass(frozen=True)
class _TDBatch(_StateActions):
    discounted_return: torch.Tensor
    terminal: torch.Tensor
    valid_len: torch.Tensor
    next_z: torch.Tensor | None
    next_proprio: torch.Tensor | None
    next_reference: torch.Tensor | None
    next_committed: torch.Tensor | None


@dataclass(frozen=True)
class _PairBatch(_StateActions):
    pair_valid: torch.Tensor
    proposal: torch.Tensor


@dataclass(frozen=True)
class _Batch(_StateActions):
    base_reference: torch.Tensor
    committed: torch.Tensor
    valid_mask: torch.Tensor
    human_mask: torch.Tensor
    pair_mask: torch.Tensor
    proposal: torch.Tensor
    relative_q_eligible: torch.Tensor


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

    def _project_execution(
        self, actions: torch.Tensor, committed: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(self.action_projector, AffineActionMapping):
            return self.action_projector(actions, committed=committed)
        return self.action_projector(actions)

    def critic_step(
        self,
        transitions: Sequence[OnlineResidualTransition],
        *,
        corrections: Sequence[OnlineResidualTransition] | None = None,
    ) -> LearnerMetrics:
        """Update both Critics from TD rows and any trustworthy correction pairs."""
        batch = self._collate_td(transitions)
        with torch.no_grad():
            active = ~batch.terminal
            target = batch.discounted_return.reshape(-1, 1).clone()
            if active.any():
                target[active] = self._bootstrap_target(
                    batch, active, batch.next_reference
                )

        q_exec = self.critic(batch.z, batch.proprio, batch.executed)
        loss = critic_loss(q_exec, target, config=self.config)
        pair_rows = transitions if corrections is None else corrections
        if pair_rows:
            pairs = self._collate_pairs(pair_rows)
            pair_q = self.critic(pairs.z, pairs.proprio, pairs.executed)
            proposal_q = self.critic(pairs.z, pairs.proprio, pairs.proposal)
            rank = critic_loss(
                pair_q,
                pair_q[0].detach(),
                q_proposal=proposal_q,
                pair_valid=pairs.pair_valid,
                config=self.config,
            ).rank
            loss = CriticLoss(loss.td + self.config.rank_weight * rank, loss.td, rank)
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.max_grad_norm
        )
        self.critic_optimizer.step()
        metrics = self._critic_metrics(loss, grad_norm).values
        for index, values in enumerate(q_exec, start=1):
            metrics.update(self._distribution(f"critic/q{index}", values))
        metrics.update(self._distribution("critic/target", target))
        if pair_rows:
            for index in range(2):
                if pairs.pair_valid.any():
                    metrics.update(
                        self._distribution(
                            f"critic/pair_delta_q{index + 1}",
                            (pair_q[index] - proposal_q[index])[pairs.pair_valid],
                        )
                    )
        return LearnerMetrics(metrics)

    def _bootstrap_target(self, batch, active, reference):
        """Evaluate successor networks only for non-terminal rows."""
        with torch.no_grad():
            next_raw = self.target_actor(
                batch.next_z,
                batch.next_proprio,
                reference,
                pre_tanh_noise=self._target_noise(reference),
            )
            next_execution = self._project_execution(
                self.target_actor.execution_half(next_raw), batch.next_committed
            )
            next_candidate = torch.cat((batch.next_committed, next_execution), dim=1)
            next_q = self.target_critic(
                batch.next_z, batch.next_proprio, next_candidate
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
        batch = self._collate_actor(rows)
        smooth_valid = batch.valid_mask if self.config.smooth_weight > 0 else None
        if separate:
            bc_valid = batch.valid_mask.clone()
            bc_valid[:relative_count] = False
            relative_valid = batch.relative_q_eligible.clone()
            relative_valid[relative_count:] = False
            batch = replace(
                batch, valid_mask=bc_valid, relative_q_eligible=relative_valid
            )
        actor_actions = self.actor(batch.z, batch.proprio, batch.base_reference)
        actor_execution = self._project_execution(
            self.actor.execution_half(actor_actions), batch.committed
        )
        actor_candidate = torch.cat((batch.committed, actor_execution), dim=1)
        reference_candidate = torch.cat(
            (
                batch.committed,
                self._project_execution(
                    batch.base_reference[:, self.config.inference_horizon :],
                    batch.committed,
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
                smooth_valid_mask=smooth_valid,
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
        metrics = self._actor_metrics(loss, grad_norm).values
        for index in range(2):
            for name, values in (
                ("candidate", q_actor),
                ("reference", q_reference),
                ("proposal", q_proposal),
            ):
                metrics.update(
                    self._distribution(f"actor/{name}_q{index + 1}", values[index])
                )
        human = batch.human_mask & batch.valid_mask
        unreachable = (
            batch.executed - batch.base_reference
        ).abs() > self.config.residual_scale + 1e-6
        metrics["actor/human_unreachable_fraction"] = float(
            (unreachable & human).sum() / human.sum().clamp_min(1)
        )
        metrics["actor/human_coordinates"] = int(human.sum())
        return LearnerMetrics(metrics)

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

    def _state_actions(self, rows: Sequence[OnlineResidualTransition]) -> dict:
        if not rows:
            raise ValueError("At least one transition is required.")
        device = next(self.parameters()).device
        return {
            "z": torch.stack([row.feature.z for row in rows]).to(
                device=device, dtype=torch.float32
            ),
            "proprio": torch.stack([row.feature.proprio for row in rows]).to(
                device=device, dtype=torch.float32
            ),
            "executed": torch.stack([row.executed for row in rows]).to(
                device=device, dtype=torch.float32
            ),
        }

    def _collate_td(self, rows: Sequence[OnlineResidualTransition]) -> _TDBatch:
        values = self._state_actions(rows)
        device = values["z"].device
        successors = [row.next_feature for row in rows if not row.terminal]
        if any(feature is None for feature in successors):
            raise ValueError("Non-terminal TD rows require successor features.")
        next_values = {
            "next_" + name: torch.stack(
                [getattr(feature, attribute) for feature in successors]
            ).to(device=device, dtype=torch.float32)
            if successors
            else None
            for name, attribute in (
                ("z", "z"),
                ("proprio", "proprio"),
                ("reference", "base_reference"),
                ("committed", "committed"),
            )
        }
        return _TDBatch(
            **values,
            **next_values,
            discounted_return=torch.stack([row.discounted_return for row in rows]).to(
                device=device, dtype=torch.float32
            ),
            terminal=torch.tensor(
                [row.terminal for row in rows], device=device, dtype=torch.bool
            ),
            valid_len=torch.tensor(
                [row.valid_len for row in rows], device=device, dtype=torch.long
            ),
        )

    def _collate_pairs(self, rows: Sequence[OnlineResidualTransition]) -> _PairBatch:
        values = self._state_actions(rows)
        device = values["z"].device
        return _PairBatch(
            **values,
            pair_valid=torch.tensor(
                [
                    row.correction is not None and bool(row.correction.pair_mask.any())
                    for row in rows
                ],
                device=device,
            ),
            proposal=torch.stack(
                [
                    row.correction.proposal(executed)
                    if row.correction is not None
                    else executed
                    for row, executed in zip(rows, values["executed"], strict=True)
                ]
            ),
        )

    def _collate_actor(self, rows: Sequence[OnlineResidualTransition]) -> _Batch:
        values = self._state_actions(rows)
        device = values["z"].device

        def stack(name: str) -> torch.Tensor:
            return torch.stack([getattr(row.feature, name) for row in rows]).to(
                device=device, dtype=torch.float32
            )

        executed = values["executed"]
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
            z=values["z"],
            proprio=values["proprio"],
            base_reference=stack("base_reference"),
            committed=stack("committed"),
            executed=executed,
            valid_mask=valid_mask,
            human_mask=torch.stack(human_masks),
            pair_mask=torch.stack(pair_masks),
            proposal=torch.stack(proposals),
            relative_q_eligible=torch.tensor(
                [row.relative_q_eligible for row in rows],
                device=device,
                dtype=torch.bool,
            ),
        )

    @staticmethod
    def _distribution(name: str, values: torch.Tensor) -> dict[str, float]:
        values = values.detach().float().flatten()
        return {
            f"{name}/{key}": value.item()
            for key, value in {
                "mean": values.mean(),
                "std": values.std(unbiased=False),
                "min": values.min(),
                "max": values.max(),
                "p05": torch.quantile(values, 0.05),
                "p50": torch.quantile(values, 0.50),
                "p95": torch.quantile(values, 0.95),
            }.items()
        }

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
