# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Load a published Actor or the Actor inside a local Stage2 training save."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch

from .config import OnlineResidualRLConfig


def load_actor_state(
    path: str | Path, config: OnlineResidualRLConfig | None = None
) -> dict[str, torch.Tensor]:
    """Return bare ``ResidualActor`` weights from either supported save layout."""
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, Mapping):
        raise ValueError("Actor checkpoint must contain a state dict.")
    if (
        config is not None
        and "config" in loaded
        and loaded["config"].get("action_representation")
        != config.action_representation.value
    ):
        raise ValueError(
            "Actor checkpoint action representation differs from deployment."
        )
    if config is not None and config.proprio_normalization == "stage1":
        saved_config = loaded.get("config", {})
        if any(
            saved_config.get(key) != getattr(config, key)
            for key in ("proprio_normalization", "normalization_id")
        ):
            raise ValueError(
                "Actor checkpoint proprio normalization differs from deployment."
            )
    from_learner = "learner" in loaded
    state = loaded.get("learner", loaded.get("actor", loaded.get("state_dict", loaded)))
    if not isinstance(state, Mapping):
        raise ValueError("Actor checkpoint does not contain network weights.")
    selected = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        bare = key
        for prefix in ("module.actor.", "actor.", "module."):
            if bare.startswith(prefix):
                bare = bare[len(prefix) :]
                break
        if not from_learner or key.startswith(("actor.", "module.actor.")):
            selected[bare] = value
    if not selected:
        raise ValueError("Actor checkpoint contains no Actor tensors.")
    if config is not None:
        output = selected.get(f"mlp.{2 * config.num_hidden_layers}.weight")
        inputs = selected.get("mlp.0.weight")
        if (
            output is None
            or inputs is None
            or output.ndim != 2
            or inputs.ndim != 2
            or output.shape[0] != config.actor_output_dim
            or inputs.shape[1] != config.actor_input_dim
        ):
            raise ValueError(
                "Actor checkpoint shape differs from deployment action contract."
            )
    return selected
