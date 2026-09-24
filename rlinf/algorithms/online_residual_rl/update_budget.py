# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Finite learner-update accounting driven by newly committed transitions."""

from dataclasses import dataclass


@dataclass
class UpdateBudget:
    """Track G Critic updates per unique transition and one Actor per D Critic."""

    critic_per_transition: int = 5
    critic_per_actor: int = 5
    pending_critic: int = 0
    critic_since_actor: int = 0

    def __post_init__(self) -> None:
        if self.critic_per_transition <= 0 or self.critic_per_actor <= 0:
            raise ValueError("Update ratios must be positive.")

    def add_transitions(self, count: int) -> None:
        """Add update credit for previously unseen replay rows."""
        if count < 0:
            raise ValueError("Transition count cannot be negative.")
        self.pending_critic += count * self.critic_per_transition

    def take_critic(self) -> bool:
        """Consume one Critic update when credit remains."""
        if self.pending_critic == 0:
            return False
        self.pending_critic -= 1
        self.critic_since_actor += 1
        return True

    def take_actor(self) -> bool:
        """Consume one Actor update after the configured Critic interval."""
        if self.critic_since_actor < self.critic_per_actor:
            return False
        self.critic_since_actor -= self.critic_per_actor
        return True
