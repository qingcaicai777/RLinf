# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Local copy of the myq intervention command contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class InterventionCommand:
    """One time-limited Cartesian command in the robot world frame."""

    timestamp_ns: int
    valid_until_ns: int
    mode: str
    arm: str
    eef_delta: np.ndarray
    gripper: np.ndarray | None = None
    source: str = "leader_drag"


class InterventionSource(Protocol):
    """Return the active command without importing the myq example package."""

    def poll(self, now_ns: int) -> InterventionCommand | None:
        """Return a current command, or None while policy owns the follower."""
