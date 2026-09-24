# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Fixed training normalization and deterministic physical command projection."""

from __future__ import annotations

import torch
from torch import nn


class ActionMapping(nn.Module):
    """Share decode, physical limits and receipt encoding across learner/rollout.

    ``offset`` and ``scale`` describe the affine inverse of the dataset's action
    normalization, including its epsilon. Bounds are in physical command units.
    Gripper bounds belong in the last dimension, just like arm bounds.
    """

    def __init__(self, offset, scale, lower, upper) -> None:
        super().__init__()
        values = [
            torch.as_tensor(value, dtype=torch.float32)
            for value in (offset, scale, lower, upper)
        ]
        if any(value.ndim != 1 or value.shape != values[0].shape for value in values):
            raise ValueError("Action mapping parameters must be equally sized vectors.")
        if not all(torch.isfinite(value).all() for value in values):
            raise ValueError("Action mapping parameters must be finite.")
        if (values[1] <= 0).any() or (values[2] >= values[3]).any():
            raise ValueError(
                "Scale must be positive and lower bounds below upper bounds."
            )
        for name, value in zip(
            ("offset", "scale", "lower", "upper"), values, strict=True
        ):
            self.register_buffer(name, value.clone())

    def decode(self, normalized: torch.Tensor) -> torch.Tensor:
        """Convert model actions to physical commands before controller limits."""
        return normalized * self.scale + self.offset

    def encode(self, physical: torch.Tensor) -> torch.Tensor:
        """Encode final accepted commands using the fixed training statistics."""
        return (physical - self.offset) / self.scale

    def project_physical(self, physical: torch.Tensor) -> torch.Tensor:
        """Apply the same deterministic physical limits as the controller."""
        return torch.maximum(torch.minimum(physical, self.upper), self.lower)

    def forward(self, normalized: torch.Tensor) -> torch.Tensor:
        """Return P(a): decode, apply command bounds, and encode again."""
        return self.encode(self.project_physical(self.decode(normalized)))
