# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Fixed training normalization and deterministic physical command projection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from torch import nn


class AffineActionMapping(nn.Module, ABC):
    """Encode/decode physical actions with frozen normalization buffers."""

    offset: torch.Tensor
    scale: torch.Tensor

    @abstractmethod
    def project_physical(self, physical: torch.Tensor) -> torch.Tensor:
        """Apply controller projection in physical action units."""
        raise NotImplementedError

    def decode(self, normalized: torch.Tensor) -> torch.Tensor:
        """Convert model actions to physical commands before controller limits."""
        return normalized * self.scale + self.offset

    def encode(self, physical: torch.Tensor) -> torch.Tensor:
        """Encode final accepted commands using the fixed training statistics."""
        return (physical - self.offset) / self.scale

    def forward(
        self, normalized: torch.Tensor, *, committed: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return P(a): decode, apply command bounds, and encode again."""
        return self.encode(self.project_physical(self.decode(normalized)))

    def _project_gripper(self, normalized, physical, committed):
        if committed is None:
            if bool(self.binary_gripper):
                raise ValueError(
                    "Binary projection requires a matching committed prefix."
                )
            return physical
        if (
            normalized.ndim != 3
            or committed.ndim != 3
            or committed.shape[0] != normalized.shape[0]
            or committed.shape[1] == 0
            or committed.shape[2] != normalized.shape[2]
        ):
            raise ValueError("Gripper projection requires a matching committed prefix.")
        previous = self.decode(committed[:, -1])[..., -1]
        widths = []
        for raw, width in zip(
            normalized[..., -1].unbind(1), physical[..., -1].unbind(1), strict=True
        ):
            if bool(self.binary_gripper):
                width = torch.where(
                    raw < self.threshold,
                    self.gripper_bounds[0],
                    torch.where(raw > self.threshold, self.gripper_bounds[1], previous),
                )
            # Match the controller's 1 mm command deadband, including fp32 roundoff.
            previous = torch.where(
                (width - previous).abs() > 0.001 + 1e-8, width, previous
            )
            widths.append(previous)
        return torch.cat(
            (physical[..., :-1], torch.stack(widths, dim=1)[..., None]), dim=-1
        )


class ActionMapping(AffineActionMapping):
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

    def project_physical(self, physical: torch.Tensor) -> torch.Tensor:
        """Apply the same deterministic physical limits as the controller."""
        return torch.maximum(torch.minimum(physical, self.upper), self.lower)


class FlexivDeltaActionMapping(ActionMapping):
    """Match the Flexiv adapter's norm-limited 6D delta projection."""

    def __init__(
        self,
        offset,
        scale,
        *,
        translation_limit: float,
        rotation_limit: float,
        gripper_range: tuple[float, float],
        limit_tcp_delta: bool = True,
        binary_gripper: bool = False,
        threshold: float = 0.5,
    ) -> None:
        if translation_limit <= 0 or rotation_limit <= 0:
            raise ValueError("Flexiv delta limits must be positive.")
        low, high = gripper_range
        super().__init__(
            offset,
            scale,
            [-translation_limit] * 3 + [-rotation_limit] * 3 + [low],
            [translation_limit] * 3 + [rotation_limit] * 3 + [high],
        )
        self.register_buffer("binary_gripper", torch.tensor(binary_gripper))
        self.register_buffer("threshold", torch.tensor(threshold))
        self.register_buffer("gripper_bounds", torch.tensor(gripper_range))
        self.translation_limit = float(translation_limit)
        self.rotation_limit = float(rotation_limit)
        self.limit_tcp_delta = bool(limit_tcp_delta)

    def forward(
        self, normalized: torch.Tensor, *, committed: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Project arm bounds and accept gripper commands sequentially."""
        physical = self.project_physical(self.decode(normalized))
        return self.encode(self._project_gripper(normalized, physical, committed))

    def project_physical(self, physical: torch.Tensor) -> torch.Tensor:
        """Limit translation and rotation norms; clamp gripper separately."""
        if physical.shape[-1] != 7:
            raise ValueError("Flexiv physical delta must have seven dimensions.")

        def limited(values: torch.Tensor, maximum: float) -> torch.Tensor:
            norm = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
            return values * (maximum / norm.clamp_min(maximum)).clamp(max=1.0)

        if not self.limit_tcp_delta:
            limited_translation = physical[..., :3]
            limited_rotation = physical[..., 3:6]
        else:
            limited_translation = limited(physical[..., :3], self.translation_limit)
            limited_rotation = limited(physical[..., 3:6], self.rotation_limit)
        return torch.cat(
            (
                limited_translation,
                limited_rotation,
                physical[..., 6:7].clamp(self.lower[6], self.upper[6]),
            ),
            dim=-1,
        )


class FlexivJointActionMapping(AffineActionMapping):
    """Preserve absolute joint angles and project the physical gripper command.

    Joint limits are validated by the connected robot, which rejects invalid
    targets rather than clipping angles. Binary gripper projection starts from
    the final command in the committed prefix and holds it at threshold ties.
    """

    def __init__(
        self,
        offset: torch.Tensor | Sequence[float],
        scale: torch.Tensor | Sequence[float],
        *,
        gripper_range: tuple[float, float],
        binary_gripper: bool = False,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        offset = torch.as_tensor(offset, dtype=torch.float32)
        scale = torch.as_tensor(scale, dtype=torch.float32)
        bounds = torch.as_tensor(gripper_range, dtype=torch.float32)
        if (
            offset.shape != (8,)
            or scale.shape != (8,)
            or not torch.isfinite(offset).all()
            or not torch.isfinite(scale).all()
            or (scale <= 0).any()
        ):
            raise ValueError(
                "Joint normalization needs eight finite offsets and positive scales."
            )
        if (
            bounds.shape != (2,)
            or not torch.isfinite(bounds).all()
            or bounds[0] >= bounds[1]
            or not torch.isfinite(torch.tensor(threshold))
        ):
            raise ValueError(
                "Gripper bounds must increase and threshold must be finite."
            )
        for name, value in (
            ("offset", offset),
            ("scale", scale),
            ("gripper_bounds", bounds),
            ("binary_gripper", torch.tensor(binary_gripper)),
            ("threshold", torch.tensor(threshold)),
        ):
            self.register_buffer(name, value.clone())

    def project_physical(self, physical: torch.Tensor) -> torch.Tensor:
        """Keep seven absolute radians and clamp only physical gripper width."""
        if physical.shape[-1] != 8:
            raise ValueError(
                "Flexiv absolute joint actions must have eight dimensions."
            )
        return torch.cat(
            (
                physical[..., :7],
                physical[..., 7:8].clamp(
                    self.gripper_bounds[0], self.gripper_bounds[1]
                ),
            ),
            dim=-1,
        )

    def forward(
        self, normalized: torch.Tensor, *, committed: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Project execution using the prior accepted gripper target for ties."""
        physical = self.project_physical(self.decode(normalized))
        return self.encode(self._project_gripper(normalized, physical, committed))
