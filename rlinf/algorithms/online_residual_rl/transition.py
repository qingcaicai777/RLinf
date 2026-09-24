# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Replay records for a decision that matures after a full ``2n`` window."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

from .config import OnlineResidualRLConfig
from .losses import discounted_return


@dataclass(frozen=True)
class FeatureRecord:
    """Features captured once at a decision boundary."""

    obs_id: str
    z: torch.Tensor
    proprio: torch.Tensor
    base_reference: torch.Tensor
    committed: torch.Tensor
    episode_id: str
    frame_index: int
    model_version: int
    config_version: str


@dataclass(frozen=True)
class SparseCorrection:
    """Human supervision attached only when correction data is useful."""

    human_mask: torch.Tensor
    pair_mask: torch.Tensor
    proposal_values: torch.Tensor

    def proposal(self, executed: torch.Tensor) -> torch.Tensor:
        """Reconstruct the pre-correction proposal from sparse values."""
        if self.proposal_values.numel() != int(self.pair_mask.sum().item()):
            raise ValueError(
                "proposal_values must contain one value per pair_mask entry."
            )
        result = executed.clone()
        pair_mask = self.pair_mask.to(device=result.device)
        result[pair_mask] = self.proposal_values.to(
            device=result.device, dtype=result.dtype
        )
        return result


@dataclass(frozen=True)
class OnlineResidualTransition:
    """A validated replay row spanning the committed and new execution blocks."""

    feature: FeatureRecord
    next_feature: FeatureRecord | None
    executed: torch.Tensor
    rewards: torch.Tensor
    discounted_return: torch.Tensor
    terminal: bool
    valid_len: int
    correction: SparseCorrection | None = None
    schedule_valid: bool = True
    receipts_valid: bool = True

    @property
    def relative_q_eligible(self) -> bool:
        """Return whether the transition may train the relative-Q actor term."""
        return (
            not self.terminal
            and self.valid_len == self.executed.shape[0]
            and self.schedule_valid
            and self.receipts_valid
        )


@dataclass(frozen=True)
class _PendingDecision:
    feature: FeatureRecord
    policy_proposal: torch.Tensor


class TransitionAssembler:
    """Hold decisions until execution receipts, labels, and successors are final."""

    def __init__(self, config: OnlineResidualRLConfig) -> None:
        self.config = config
        self._pending: dict[str, _PendingDecision] = {}

    def begin(self, feature: FeatureRecord, policy_proposal: torch.Tensor) -> None:
        """Register an immutable request snapshot at its inference boundary."""
        if feature.obs_id in self._pending:
            raise ValueError(f"Decision {feature.obs_id!r} is already pending.")
        expected = (self.config.actor_horizon, self.config.action_dim)
        if tuple(policy_proposal.shape) != expected:
            raise ValueError(f"policy_proposal must have shape {expected}.")
        if not torch.allclose(
            policy_proposal[: self.config.inference_horizon],
            feature.committed,
            rtol=0.0,
            atol=1.0e-5,
        ):
            raise ValueError("Policy proposal must retain the real committed prefix.")
        self._pending[feature.obs_id] = _PendingDecision(
            feature=replace(
                feature,
                z=feature.z.detach().clone(),
                proprio=feature.proprio.detach().clone(),
                base_reference=feature.base_reference.detach().clone(),
                committed=feature.committed.detach().clone(),
            ),
            policy_proposal=policy_proposal.detach().clone(),
        )

    def discard(self, obs_id: str) -> None:
        """Drop a late or corrupted request without creating replay data."""
        self._pending.pop(obs_id, None)

    def complete(
        self,
        obs_id: str,
        *,
        executed: torch.Tensor,
        rewards: torch.Tensor,
        terminal: bool,
        valid_len: int,
        next_feature: FeatureRecord | None,
        human_mask: torch.Tensor | None = None,
        pair_mask: torch.Tensor | None = None,
        schedule_valid: bool = True,
        receipts_valid: bool = True,
        label_confirmed: bool = True,
    ) -> OnlineResidualTransition:
        """Finalize one row after its ``2n`` execution window matures."""
        if not label_confirmed:
            raise ValueError("An unconfirmed terminal label cannot enter replay.")
        try:
            pending = self._pending[obs_id]
        except KeyError as exc:
            raise KeyError(f"Unknown pending decision {obs_id!r}.") from exc

        if human_mask is not None or pair_mask is not None:
            human_mask = torch.as_tensor(
                human_mask if human_mask is not None else pair_mask, dtype=torch.bool
            ).clone()
            pair_mask = torch.as_tensor(
                pair_mask if pair_mask is not None else human_mask, dtype=torch.bool
            ).clone()
            human_mask[valid_len:] = False
            pair_mask[: self.config.inference_horizon] = False
            pair_mask[valid_len:] = False
        correction = self._build_correction(
            pending.policy_proposal,
            executed,
            human_mask=human_mask,
            pair_mask=pair_mask,
        )
        valid = torch.arange(self.config.td_horizon, device=rewards.device) < int(
            valid_len
        )
        value = discounted_return(
            rewards.unsqueeze(0), valid.unsqueeze(0), gamma=self.config.gamma
        ).squeeze(0)
        transition = OnlineResidualTransition(
            feature=pending.feature,
            next_feature=next_feature,
            executed=executed.detach().clone(),
            rewards=rewards.detach().clone(),
            discounted_return=value.detach().clone(),
            terminal=bool(terminal),
            valid_len=int(valid_len),
            correction=correction,
            schedule_valid=bool(schedule_valid),
            receipts_valid=bool(receipts_valid),
        )
        validate_transition(transition, self.config)
        del self._pending[obs_id]
        return transition

    def complete_from_receipts(
        self,
        obs_id: str,
        *,
        step_infos: Sequence[Mapping[str, Any]],
        encode_actions: Callable[[torch.Tensor], torch.Tensor],
        terminal: bool,
        next_feature: FeatureRecord | None,
        label_confirmed: bool = True,
    ) -> OnlineResidualTransition:
        """Build a transition from consecutive final robot command receipts.

        ``step_infos`` contains the 2n control frames starting at the pending
        decision's frame index, or the valid prefix of a terminal window. Each
        item must carry the final ``action_receipt`` and ``frame_index``.
        ``action_receipt_physical`` takes precedence when the environment's
        command representation differs from the SFT action domain (Flexiv's
        normalized gripper command versus width in metres).
        ``encode_actions`` uses the frozen SFT normalization, not measured state.
        """
        try:
            feature = self._pending[obs_id].feature
        except KeyError as exc:
            raise KeyError(f"Unknown pending decision {obs_id!r}.") from exc
        valid_len = len(step_infos)
        if not self.config.inference_horizon < valid_len <= self.config.td_horizon:
            raise ValueError(
                "Receipt window must contain the committed and executed frames."
            )

        physical = []
        human = torch.zeros(
            self.config.td_horizon, self.config.action_dim, dtype=torch.bool
        )
        pair = human.clone()
        for offset, info in enumerate(step_infos):
            if info.get("frame_index") != feature.frame_index + offset:
                raise ValueError(
                    "Receipt frame indices must be consecutive from the decision."
                )
            if not bool(info.get("receipt_valid", True)):
                raise ValueError(
                    "A missing or unknown final command receipt is invalid."
                )
            receipt = torch.as_tensor(
                info.get("action_receipt_physical", info["action_receipt"]),
                dtype=torch.float32,
            )
            if (
                receipt.shape != (self.config.action_dim,)
                or not torch.isfinite(receipt).all()
            ):
                raise ValueError("Final action receipt has invalid shape or values.")
            physical.append(receipt)
            if "intervene_action" in info and "human_mask" not in info:
                raise ValueError("Intervention receipts require a human_mask.")
            for name, target in (("human_mask", human), ("pair_mask", pair)):
                mask = info.get(name, info.get("human_mask", None))
                if mask is None:
                    continue
                mask = torch.as_tensor(mask, dtype=torch.bool)
                if mask.shape != (self.config.action_dim,):
                    raise ValueError(
                        f"{name} must have shape ({self.config.action_dim},)."
                    )
                target[offset] = mask

        normalized = (
            torch.as_tensor(encode_actions(torch.stack(physical)), dtype=torch.float32)
            .detach()
            .cpu()
        )
        if normalized.shape != (valid_len, self.config.action_dim):
            raise ValueError("Encoded receipts must retain [valid_len, action_dim].")
        executed = torch.zeros(
            self.config.td_horizon, self.config.action_dim, dtype=torch.float32
        )
        executed[:valid_len] = normalized
        return self.complete(
            obs_id,
            executed=executed,
            rewards=torch.zeros(self.config.td_horizon),
            terminal=terminal,
            valid_len=valid_len,
            next_feature=next_feature,
            human_mask=human,
            pair_mask=pair,
            label_confirmed=label_confirmed,
        )

    def _build_correction(
        self,
        proposal: torch.Tensor,
        executed: torch.Tensor,
        *,
        human_mask: torch.Tensor | None,
        pair_mask: torch.Tensor | None,
    ) -> SparseCorrection | None:
        shape = (self.config.actor_horizon, self.config.action_dim)
        if human_mask is None and pair_mask is None:
            return None
        if human_mask is None:
            human_mask = pair_mask
        if pair_mask is None:
            pair_mask = human_mask
        human_mask = torch.as_tensor(human_mask, dtype=torch.bool)
        pair_mask = torch.as_tensor(pair_mask, dtype=torch.bool)
        if tuple(human_mask.shape) != shape or tuple(pair_mask.shape) != shape:
            raise ValueError(f"Correction masks must have shape {shape}.")
        # A comparison is trustworthy only where an executed value was also
        # marked as a human intervention. This implements the requested default
        # while allowing callers to remove uncertain preference entries.
        pair_mask = pair_mask & human_mask
        values = proposal.to(device=pair_mask.device)[pair_mask]
        return SparseCorrection(human_mask, pair_mask, values.detach().clone())


def _validate_feature(feature: FeatureRecord, config: OnlineResidualRLConfig) -> None:
    for tensor in (
        feature.z,
        feature.proprio,
        feature.base_reference,
        feature.committed,
    ):
        if not torch.isfinite(tensor).all():
            raise ValueError("Feature tensors must be finite.")
    expected_ref = (config.actor_horizon, config.action_dim)
    expected_committed = (config.inference_horizon, config.action_dim)
    if tuple(feature.z.shape) != (config.z_dim,):
        raise ValueError(f"Feature z must have shape ({config.z_dim},).")
    if tuple(feature.proprio.shape) != (config.proprio_dim,):
        raise ValueError(f"Feature proprio must have shape ({config.proprio_dim},).")
    if tuple(feature.base_reference.shape) != expected_ref:
        raise ValueError(f"Feature base_reference must have shape {expected_ref}.")
    if tuple(feature.committed.shape) != expected_committed:
        raise ValueError(f"Feature committed must have shape {expected_committed}.")


def validate_transition(
    transition: OnlineResidualTransition,
    config: OnlineResidualRLConfig,
) -> None:
    """Reject rows that violate the replay timing and shape contract."""
    _validate_feature(transition.feature, config)
    expected = (config.td_horizon, config.action_dim)
    if tuple(transition.executed.shape) != expected:
        raise ValueError(f"executed must have shape {expected}.")
    if not torch.allclose(
        transition.executed[: config.inference_horizon],
        transition.feature.committed,
        rtol=0.0,
        atol=1.0e-5,
    ):
        raise ValueError("Executed actions must retain the real committed prefix.")
    if transition.rewards.numel() != config.td_horizon:
        raise ValueError(f"rewards must contain {config.td_horizon} frames.")
    if not 0 < transition.valid_len <= config.td_horizon:
        raise ValueError("valid_len must be in [1, td_horizon].")
    if not transition.schedule_valid:
        raise ValueError("Scheduling-invalid windows cannot enter replay.")
    if transition.valid_len <= config.inference_horizon:
        raise ValueError("The current execution must have started before termination.")
    if not transition.terminal and transition.valid_len != config.td_horizon:
        raise ValueError("Non-terminal windows must contain the full TD horizon.")
    for value in (
        transition.executed,
        transition.rewards,
        transition.discounted_return,
    ):
        if not torch.isfinite(value).all():
            raise ValueError("Replay tensors must be finite.")
    if transition.terminal:
        if transition.next_feature is not None:
            raise ValueError("Terminal transitions cannot link a reset observation.")
    elif transition.next_feature is None:
        raise ValueError("Non-terminal transitions require the t+2n feature.")
    else:
        _validate_feature(transition.next_feature, config)
        expected_frame = transition.feature.frame_index + config.td_horizon
        if transition.next_feature.frame_index != expected_frame:
            raise ValueError(
                "next_feature must be exactly t+2n: expected frame "
                f"{expected_frame}, got {transition.next_feature.frame_index}."
            )
        if transition.next_feature.episode_id != transition.feature.episode_id:
            raise ValueError("A transition cannot bootstrap across episode reset.")
    if not transition.receipts_valid:
        raise ValueError(
            "Transitions with missing or unknown execution receipts are invalid."
        )
    correction = transition.correction
    if correction is not None:
        if tuple(correction.human_mask.shape) != expected:
            raise ValueError(f"human_mask must have shape {expected}.")
        if tuple(correction.pair_mask.shape) != expected:
            raise ValueError(f"pair_mask must have shape {expected}.")
        if torch.any(correction.pair_mask & ~correction.human_mask):
            raise ValueError("pair_mask must be a subset of human_mask.")
        if correction.pair_mask[: config.inference_horizon].any():
            raise ValueError(
                "Historical committed corrections cannot create rank pairs."
            )
        if correction.human_mask[transition.valid_len :].any():
            raise ValueError("Padding cannot carry human supervision.")
        if not torch.isfinite(correction.proposal_values).all():
            raise ValueError("Proposal values must be finite.")
        if correction.proposal_values.numel() != correction.pair_mask.sum().item():
            raise ValueError("Sparse proposal length must match pair_mask.")
