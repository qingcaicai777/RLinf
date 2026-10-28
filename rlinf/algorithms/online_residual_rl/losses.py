# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Losses for delayed, overlapping online residual transitions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import OnlineResidualRLConfig


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=values.dtype)
    return torch.sum(values * mask) / torch.clamp(torch.sum(mask), min=1.0)


@dataclass(frozen=True)
class CriticLoss:
    """Critic loss terms before the optimizer step."""

    total: torch.Tensor
    td: torch.Tensor
    rank: torch.Tensor


def critic_loss(
    q_exec: tuple[torch.Tensor, torch.Tensor],
    target: torch.Tensor,
    *,
    q_proposal: tuple[torch.Tensor, torch.Tensor] | None = None,
    pair_valid: torch.Tensor | None = None,
    config: OnlineResidualRLConfig,
) -> CriticLoss:
    """Combine twin-Q TD loss and correction-pair margin ranking."""
    target = target.detach()
    td = 0.5 * sum(F.mse_loss(q, target.expand_as(q)) for q in q_exec)
    rank = target.new_zeros(())
    if q_proposal is not None:
        if pair_valid is None:
            pair_valid = torch.ones_like(target, dtype=torch.bool)
        pair_valid = pair_valid.reshape(-1, 1)
        terms = []
        for exec_q, proposal_q in zip(q_exec, q_proposal):
            margin_error = torch.clamp(
                config.rank_margin - (exec_q - proposal_q), min=0.0
            ).square()
            terms.append(_masked_mean(margin_error, pair_valid))
        rank = 0.5 * sum(terms)
    return CriticLoss(total=td + config.rank_weight * rank, td=td, rank=rank)


def td_target(
    discounted_return: torch.Tensor,
    terminal: torch.Tensor,
    next_q: tuple[torch.Tensor, torch.Tensor] | None,
    *,
    valid_len: torch.Tensor,
    config: OnlineResidualRLConfig,
) -> torch.Tensor:
    """Build a variable-length TD target and skip bootstrap for terminals."""
    result = discounted_return.reshape(-1, 1)
    terminal = terminal.reshape(-1, 1).bool()
    if next_q is None:
        if (~terminal).any():
            raise ValueError("Non-terminal transitions require next_q.")
        return result
    q_next = torch.minimum(next_q[0], next_q[1]).detach()
    discount = torch.pow(
        torch.as_tensor(config.gamma, device=result.device, dtype=result.dtype),
        valid_len.reshape(-1, 1).to(device=result.device, dtype=result.dtype),
    )
    return result + (~terminal).to(result.dtype) * discount * q_next


def discounted_return(
    rewards: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    """Accumulate per-control-frame rewards over the valid TD window."""
    rewards = rewards.reshape(rewards.shape[0], -1)
    valid_mask = valid_mask.reshape(valid_mask.shape[0], -1).to(rewards.dtype)
    powers = torch.arange(rewards.shape[1], device=rewards.device, dtype=rewards.dtype)
    discounts = torch.pow(rewards.new_tensor(gamma), powers)
    return torch.sum(rewards * valid_mask * discounts, dim=-1, keepdim=True)


@dataclass(frozen=True)
class ActorLoss:
    """Actor loss terms before the optimizer step."""

    total: torch.Tensor
    relative_q: torch.Tensor
    ordinary_bc: torch.Tensor
    human_bc: torch.Tensor
    smooth: torch.Tensor


def relative_q_loss(
    q_actor: tuple[torch.Tensor, torch.Tensor],
    q_reference: tuple[torch.Tensor, torch.Tensor],
    *,
    q_proposal: tuple[torch.Tensor, torch.Tensor] | None,
    corrected: torch.Tensor,
    eligible: torch.Tensor,
    config: OnlineResidualRLConfig,
) -> torch.Tensor:
    """Compute per-Critic relative advantages before pessimistic aggregation."""
    corrected = corrected.reshape(-1, 1).bool()
    eligible = eligible.reshape(-1, 1).bool()
    deltas = []
    for index in range(2):
        baseline = q_reference[index]
        if q_proposal is not None:
            corrected_baseline = q_reference[index] + torch.clamp(
                q_proposal[index] - q_reference[index], min=0.0
            )
            baseline = torch.where(corrected, corrected_baseline, baseline)
        deltas.append(q_actor[index] - baseline.detach())
    pessimistic_delta = torch.minimum(deltas[0], deltas[1])
    loss = config.actor_kappa * F.softplus(
        (config.actor_margin - pessimistic_delta) / config.actor_kappa
    )
    return _masked_mean(loss, eligible)


def behavior_cloning_losses(
    actor_actions: torch.Tensor,
    base_reference: torch.Tensor,
    executed: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    human_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute separately normalized ordinary and human BC terms."""
    valid_mask = valid_mask.to(dtype=torch.bool)
    human_mask = human_mask.to(dtype=torch.bool) & valid_mask
    human_sample = human_mask.reshape(human_mask.shape[0], -1).any(dim=-1)
    ordinary_sample = ~human_sample & valid_mask.flatten(1).any(-1)

    ref_error = (actor_actions - base_reference).square()
    human_error = (actor_actions - executed).square()
    ordinary_mask = valid_mask & ordinary_sample[:, None, None]
    human_only_mask = human_mask & human_sample[:, None, None]

    def sample_mean(error: torch.Tensor, mask: torch.Tensor, selected: torch.Tensor):
        numerator = (error * mask).flatten(1).sum(-1)
        denominator = mask.flatten(1).sum(-1).clamp_min(1)
        return _masked_mean(numerator / denominator, selected)

    return (
        sample_mean(ref_error, ordinary_mask, ordinary_sample),
        sample_mean(human_error, human_only_mask, human_sample),
    )


def direct_delta_smoothness_loss(
    committed: torch.Tensor,
    actor_execution: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    smooth_dims: int = 6,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Average squared joint-vector differences over valid times per sample.

    Each order excludes differences touching invalid frames, then averages
    samples with at least one valid difference. The committed prefix is frozen.
    """
    values = torch.cat((committed.detach(), actor_execution), dim=1)[..., :smooth_dims]
    mask = valid_mask[..., :smooth_dims].bool().all(dim=-1)
    total = values.new_zeros(())
    current = values
    current_mask = mask
    for order, weight in enumerate(weights, start=1):
        current = torch.diff(current, n=1, dim=1)
        current_mask = current_mask[:, 1:] & current_mask[:, :-1]
        if current.shape[1] == 0:
            break
        squared_norm = current.square().sum(dim=-1)
        counts = current_mask.sum(dim=-1)
        per_sample = (squared_norm * current_mask).sum(dim=-1) / counts.clamp_min(1)
        total = total + float(weight) * _masked_mean(per_sample, counts > 0)
    return total


def actor_loss(
    actor_actions: torch.Tensor,
    base_reference: torch.Tensor,
    executed: torch.Tensor,
    committed: torch.Tensor,
    *,
    q_actor: tuple[torch.Tensor, torch.Tensor],
    q_reference: tuple[torch.Tensor, torch.Tensor],
    q_proposal: tuple[torch.Tensor, torch.Tensor] | None,
    valid_mask: torch.Tensor,
    human_mask: torch.Tensor,
    corrected: torch.Tensor,
    relative_q_eligible: torch.Tensor,
    config: OnlineResidualRLConfig,
    projected_execution: torch.Tensor | None = None,
    smooth_valid_mask: torch.Tensor | None = None,
) -> ActorLoss:
    """Combine pessimistic relative Q, disjoint BC terms, and optional smoothing."""
    relative = relative_q_loss(
        q_actor,
        q_reference,
        q_proposal=q_proposal,
        corrected=corrected,
        eligible=relative_q_eligible,
        config=config,
    )
    ordinary_bc, human_bc = behavior_cloning_losses(
        actor_actions,
        base_reference,
        executed,
        valid_mask=valid_mask,
        human_mask=human_mask,
    )
    smooth = actor_actions.new_zeros(())
    if config.smooth_weight > 0.0:
        if projected_execution is None:
            raise ValueError("Smoothness requires the physically projected execution.")
        smooth = direct_delta_smoothness_loss(
            committed,
            projected_execution,
            valid_mask=valid_mask if smooth_valid_mask is None else smooth_valid_mask,
            smooth_dims=config.action_representation.smooth_dims,
            weights=config.smooth_weights,
        )
    total = (
        config.relative_q_weight * relative
        + config.ordinary_bc_weight * ordinary_bc
        + config.human_bc_weight * human_bc
        + config.smooth_weight * smooth
    )
    return ActorLoss(total, relative, ordinary_bc, human_bc, smooth)
