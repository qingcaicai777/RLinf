# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Finite online training and complete learner checkpoints."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import torch

from .learner import OnlineResidualLearner
from .replay import EpisodeReplay
from .update_budget import UpdateBudget


class OnlineTraining:
    """Own replay admission, update credit, warmup and Actor publication.

    Invoke methods from one learner thread. Robot collection submits only fully
    labelled episodes at this boundary. ``publish`` receives a detached CPU Actor
    snapshot; rollout installs it only at a subsequent request boundary.
    """

    def __init__(
        self,
        learner: OnlineResidualLearner,
        replay: EpisodeReplay,
        *,
        initial_episodes: int = 50,
        warmup_updates: int = 100,
        td_batch_size: int = 256,
        supervision_batch_size: int = 64,
        publish_interval: int = 10,
        publish: Callable[[int, dict], None] | None = None,
    ) -> None:
        if (
            min(initial_episodes, warmup_updates) < 0
            or min(td_batch_size, supervision_batch_size, publish_interval) <= 0
        ):
            raise ValueError("Invalid online training counts.")
        self.learner, self.replay = learner, replay
        cfg = learner.config
        self.budget = UpdateBudget(
            cfg.critic_updates_per_transition, cfg.critic_updates_per_actor_update
        )
        self.initial_episodes = initial_episodes
        self.warmup_remaining = warmup_updates
        self.td_batch_size, self.supervision_batch_size = (
            td_batch_size,
            supervision_batch_size,
        )
        self.publish_interval, self.publish = publish_interval, publish
        self.episodes = 0
        self.actor_updates = 0

    def commit_episode(self, rows, *, success: bool, terminal_frame: int) -> int:
        """Credit each newly admitted transition exactly once."""
        count = self.replay.commit_episode(
            rows, success=success, terminal_frame=terminal_frame
        )
        if count:
            self.episodes += 1
            self.budget.add_transitions(count)
        return count

    def step(self) -> dict[str, float]:
        """Perform at most one Critic update and its due Actor update."""
        if self.episodes < self.initial_episodes or not len(self.replay):
            return {}
        if not self.budget.take_critic():
            return {}
        metrics = self.learner.critic_step(
            self.replay.sample(self.td_batch_size),
            corrections=self.replay.sample(self.supervision_batch_size, kind="rank"),
        ).values
        if self.warmup_remaining:
            self.warmup_remaining -= 1
            self.budget.critic_since_actor = 0
            return metrics
        if self.budget.take_actor():
            metrics.update(
                self.learner.actor_step(
                    self.replay.sample(self.td_batch_size, kind="relative"),
                    ordinary=self.replay.sample(
                        self.supervision_batch_size, kind="ordinary"
                    ),
                    human=self.replay.sample(self.supervision_batch_size, kind="human"),
                ).values
            )
            self.actor_updates += 1
            if self.publish and self.actor_updates % self.publish_interval == 0:
                snapshot = {
                    key: value.detach().cpu().clone()
                    for key, value in self.learner.actor.state_dict().items()
                }
                self.publish(self.actor_updates, snapshot)
        return metrics

    def save(self, destination: str | Path) -> None:
        """Atomically save networks, optimizers, replay, credit and RNG states."""
        import tempfile

        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "config": asdict(self.learner.config),
            "learner": self.learner.state_dict(),
            "actor_optimizer": self.learner.actor_optimizer.state_dict(),
            "critic_optimizer": self.learner.critic_optimizer.state_dict(),
            "replay": self.replay.state_dict(),
            "budget": asdict(self.budget),
            "episodes": self.episodes,
            "actor_updates": self.actor_updates,
            "warmup_remaining": self.warmup_remaining,
            "settings": {
                name: getattr(self, name)
                for name in (
                    "initial_episodes",
                    "td_batch_size",
                    "supervision_batch_size",
                    "publish_interval",
                )
            },
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        }
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, delete=False
        ) as temporary:
            path = Path(temporary.name)
        try:
            torch.save(state, path)
            os.replace(path, destination)
        finally:
            path.unlink(missing_ok=True)

    def load(self, source: str | Path) -> None:
        """Resume a trusted local training checkpoint with matching model contract."""
        state = torch.load(source, map_location="cpu", weights_only=False)
        if state["config"] != asdict(self.learner.config):
            raise ValueError("Checkpoint model contract does not match this learner.")
        self.learner.load_state_dict(state["learner"], strict=True)
        self.learner.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.learner.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.replay.load_state_dict(state["replay"])
        self.budget = UpdateBudget(**state["budget"])
        self.episodes, self.actor_updates = state["episodes"], state["actor_updates"]
        self.warmup_remaining = state["warmup_remaining"]
        for name, value in state["settings"].items():
            setattr(self, name, value)
        torch.set_rng_state(state["rng"])
        if state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
