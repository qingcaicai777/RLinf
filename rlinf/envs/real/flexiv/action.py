# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Flexiv action conversion shared by hardware execution and replay checks."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.spatial.transform import Rotation


def flexiv_rotation(pose: Sequence[float]) -> Rotation:
    """Return SciPy's xyzw rotation from Flexiv's xyz+wxyz pose."""
    values = np.asarray(pose, dtype=np.float64)
    if values.shape != (7,):
        raise ValueError(f"Flexiv pose must have shape (7,), got {values.shape}.")
    return Rotation.from_quat([values[4], values[5], values[6], values[3]])


def apply_world_delta(pose: Sequence[float], delta: Sequence[float]) -> np.ndarray:
    """Apply a world-frame xyz(m)+rotvec(rad) delta to a measured TCP pose."""
    pose_values = np.asarray(pose, dtype=np.float64)
    delta_values = np.asarray(delta, dtype=np.float64)
    if delta_values.shape != (6,):
        raise ValueError(f"EEF delta must have shape (6,), got {delta_values.shape}.")
    result = pose_values.copy()
    result[:3] += delta_values[:3]
    # World-frame rotation is left-multiplied, matching the data collector.
    quaternion = (
        Rotation.from_rotvec(delta_values[3:]) * flexiv_rotation(pose_values)
    ).as_quat()
    result[3:] = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]
    return result


def normalize_gripper(width_m: float, *, closed_m: float, open_m: float) -> float:
    """Map GN01 width to ``[0,1]`` where zero is closed and one is open."""
    if open_m <= closed_m:
        raise ValueError("gripper_open_width_m must exceed gripper_close_width_m.")
    return float(np.clip((width_m - closed_m) / (open_m - closed_m), 0.0, 1.0))


def denormalize_gripper(value: float, *, closed_m: float, open_m: float) -> float:
    """Map a normalized gripper action to GN01 opening width in metres."""
    normalized = float(np.clip(value, 0.0, 1.0))
    return closed_m + normalized * (open_m - closed_m)
