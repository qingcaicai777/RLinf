# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Online residual RL contracts, losses, and transition assembly."""

from .config import ActionRepresentation, OnlineResidualRLConfig
from .learner import LearnerMetrics, OnlineResidualLearner
from .losses import (
    ActorLoss,
    CriticLoss,
    actor_loss,
    critic_loss,
    direct_delta_smoothness_loss,
)
from .networks import ResidualActor, TwinQCritic
from .runtime import OnlineResidualController, ResidualDecision
from .storage import TransitionStore
from .transition import (
    FeatureRecord,
    OnlineResidualTransition,
    SparseCorrection,
    TransitionAssembler,
    validate_transition,
)
from .update_budget import UpdateBudget

__all__ = [
    "ActionRepresentation",
    "ActorLoss",
    "CriticLoss",
    "FeatureRecord",
    "LearnerMetrics",
    "OnlineResidualRLConfig",
    "OnlineResidualLearner",
    "OnlineResidualController",
    "OnlineResidualTransition",
    "ResidualActor",
    "ResidualDecision",
    "SparseCorrection",
    "TransitionAssembler",
    "TransitionStore",
    "TwinQCritic",
    "UpdateBudget",
    "actor_loss",
    "critic_loss",
    "direct_delta_smoothness_loss",
    "validate_transition",
]
