# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Adapt myq's time-limited intervention commands to Flexiv env actions."""

from __future__ import annotations

import time
from typing import Any, Literal

import gymnasium as gym
import numpy as np

from rlinf.envs.real.wrappers.teleop.intervention import TeleopDevice, TeleopSample

from .action import normalize_gripper


class FlexivInterventionDevice(TeleopDevice):
    """Read a ``myq.intervention.InterventionSource`` for one Flexiv arm.

    The source is structural: it only needs ``poll(now_ns)`` returning the
    external ``InterventionCommand`` shape. Commands are physical TCP deltas;
    gripper commands, when present, default to the myq collector's width in
    metres and are converted to the Flexiv env's continuous 0–1 code.
    Wrap ``FlexivEnv`` with ``TeleopIntervention`` to get final action receipts
    and per-dimension human/pair masks from each ``step``.
    """

    timeout = 0.0  # The external command's valid_until_ns owns its lifetime.

    def __init__(
        self,
        source: Any,
        *,
        arm: str = "follower",
        gripper_unit: Literal["width_m", "normalized"] = "width_m",
    ) -> None:
        if arm not in {"leader", "follower"}:
            raise ValueError("arm must be 'leader' or 'follower'.")
        if gripper_unit not in {"width_m", "normalized"}:
            raise ValueError("gripper_unit must be 'width_m' or 'normalized'.")
        self.source = source
        self.arm = arm
        self.gripper_unit = gripper_unit

    def read(self, env: gym.Env, policy_action: np.ndarray) -> TeleopSample:
        """Return the accepted physical override and its controlled dimensions."""
        # Match the clock used by myq.drag_runtime for valid_until_ns.
        now_ns = time.perf_counter_ns()
        command = self.source.poll(now_ns)
        if command is None:
            return TeleopSample(action=None, active=False)
        if not int(command.timestamp_ns) <= now_ns <= int(command.valid_until_ns):
            return TeleopSample(action=None, active=False)
        if command.mode == "emergency_stop":
            stop = getattr(env.unwrapped, "stop_realtime", None)
            if callable(stop):
                stop()
            raise RuntimeError("Emergency stop requested by intervention source.")
        if command.mode not in {"override", "residual"}:
            raise ValueError(f"Unknown intervention mode: {command.mode!r}.")
        if command.arm not in {self.arm, "both"}:
            if command.arm in {"leader", "follower"}:
                return TeleopSample(action=None, active=False)
            raise ValueError(f"Unknown intervention arm selector: {command.arm!r}.")

        action = np.asarray(policy_action, dtype=np.float64).copy()
        if action.shape != (7,):
            raise ValueError("Flexiv Cartesian intervention requires a 7D action.")
        delta = np.asarray(command.eef_delta, dtype=np.float64).reshape(-1)
        expected = 12 if command.arm == "both" else 6
        if delta.shape != (expected,) or not np.isfinite(delta).all():
            raise ValueError(f"Intervention eef_delta must be finite {expected}D.")
        arm_index = int(self.arm == "follower") if command.arm == "both" else 0
        selected = delta[6 * arm_index : 6 * (arm_index + 1)]
        if command.mode == "override":
            action[:6] = selected
        else:
            action[:6] += selected

        controlled = np.zeros(7, dtype=bool)
        controlled[:6] = True
        if command.gripper is not None:
            gripper = np.asarray(command.gripper, dtype=np.float64).reshape(-1)
            expected_grippers = 2 if command.arm == "both" else 1
            if gripper.shape != (expected_grippers,) or not np.isfinite(gripper).all():
                raise ValueError(
                    f"Intervention gripper must be finite {expected_grippers}D."
                )
            gripper_value = float(gripper[arm_index])
            if self.gripper_unit == "width_m":
                config = env.unwrapped.config
                closed = float(config.gripper_close_width_m)
                opened = float(config.gripper_open_width_m)
                if not closed <= gripper_value <= opened:
                    raise ValueError(
                        "Gripper width is outside configured metre limits."
                    )
                gripper_value = normalize_gripper(
                    gripper_value, closed_m=closed, open_m=opened
                )
            action[6] = gripper_value
            controlled[6] = True

        # FlexivEnv.step rejects out-of-space commands before its own controller
        # limits. Bound the accepted command here, then log its final receipt.
        if not isinstance(env.action_space, gym.spaces.Box):
            raise TypeError("Flexiv intervention requires a Box action space.")
        action = np.clip(action, env.action_space.low, env.action_space.high)
        return TeleopSample(
            action=action.astype(np.float32),
            active=True,
            info={
                "teleop_action_mask": controlled,
                "intervention_source": str(command.source),
                "intervention_mode": str(command.mode),
                "intervention_timestamp_ns": int(command.timestamp_ns),
            },
        )
