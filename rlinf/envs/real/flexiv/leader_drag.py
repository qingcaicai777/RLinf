# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Main-arm drag source adapted from myq's admittance and axis alignment."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .action import flexiv_rotation
from .intervention_command import InterventionCommand


@dataclass(frozen=True)
class LeaderDragConfig:
    """Explicit-switch leader drag limits, in SI units."""

    control_hz: float = 30.0
    linear_gain: float = 0.0020
    angular_gain: float = 0.0080
    force_deadband_n: float = 1.5
    torque_deadband_nm: float = 0.15
    velocity_filter_hz: float = 6.0
    max_linear_speed_m_s: float = 0.028
    max_angular_speed_rad_s: float = 0.09
    max_linear_acceleration_m_s2: float = 0.12
    max_angular_acceleration_rad_s2: float = 0.40
    target_leash_m: float = 0.006
    target_leash_rad: float = np.deg2rad(3.0)
    impedance_scale: float = 0.30
    command_lifetime_s: float = 0.1
    force_bias_seconds: float = 1.0

    def __post_init__(self) -> None:
        positive = (
            self.control_hz,
            self.velocity_filter_hz,
            self.max_linear_speed_m_s,
            self.max_angular_speed_rad_s,
            self.max_linear_acceleration_m_s2,
            self.max_angular_acceleration_rad_s2,
            self.target_leash_m,
            self.target_leash_rad,
            self.impedance_scale,
            self.command_lifetime_s,
        )
        if any(not np.isfinite(value) or value <= 0 for value in positive):
            raise ValueError(
                "Leader drag frequency, limits and scale must be positive."
            )
        if not np.isfinite(self.force_bias_seconds) or self.force_bias_seconds < 0:
            raise ValueError("force_bias_seconds must be nonnegative.")


def _limit(vector: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm <= maximum else vector * (maximum / norm)


def _pose(values: Any) -> np.ndarray:
    pose = np.asarray(values, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("Leader/follower TCP pose must be finite xyz+wxyz.")
    return pose


class LeaderDragInterventionSource:
    """Drive a hand-guided leader; publish follower override only when enabled.

    The caller owns the follower environment. This source owns the leader robot
    only when created by ``connect``. ``set_enabled`` is the explicit button or
    switch edge; no force threshold can silently take over the policy.
    """

    def __init__(
        self,
        leader_robot: Any,
        follower_pose: Callable[[], np.ndarray],
        config: LeaderDragConfig = LeaderDragConfig(),
        *,
        owns_robot: bool = False,
    ) -> None:
        self.robot = leader_robot
        self.follower_pose = follower_pose
        self.config = config
        self.owns_robot = owns_robot
        self.enabled = False
        self._started = False
        self._closed = False
        self._rebase = True
        self._leader_anchor: np.ndarray | None = None
        self._follower_anchor: np.ndarray | None = None
        self._target: np.ndarray | None = None
        self._linear_velocity = np.zeros(3)
        self._angular_velocity = np.zeros(3)
        self._wrench_bias = np.zeros(6)
        self._axis = Rotation.from_matrix(np.diag([-1.0, -1.0, 1.0]))

    @classmethod
    def connect(
        cls,
        env: Any,
        leader_serial: str,
        config: LeaderDragConfig = LeaderDragConfig(),
        *,
        follower_pose: Callable[[], np.ndarray] | None = None,
    ) -> LeaderDragInterventionSource:
        """Connect a distinct leader arm; activation still requires ``start``."""
        if not leader_serial or leader_serial == env.unwrapped.config.serial_number:
            raise ValueError("Leader serial must be nonempty and differ from follower.")
        import flexivrdk

        robot = flexivrdk.Robot(leader_serial)
        if not robot.connected() or robot.fault() or not robot.operational():
            robot.Stop()
            raise RuntimeError("Leader robot is not ready for drag control.")
        return cls(
            robot,
            follower_pose or env.unwrapped.read_tcp_pose,
            config,
            owns_robot=True,
        )

    def start(self) -> None:
        """Set NRT Cartesian impedance and anchor the stationary leader."""
        if self._closed:
            raise RuntimeError("A closed leader drag source cannot restart.")
        if self._started:
            return
        import flexivrdk

        try:
            samples = []
            deadline = time.perf_counter() + self.config.force_bias_seconds
            while True:
                state = self.robot.states()
                samples.append(np.asarray(state.ext_wrench_in_world, dtype=np.float64))
                if time.perf_counter() >= deadline:
                    break
                time.sleep(0.01)
            pose = _pose(state.tcp_pose)
            bias_samples = np.stack(samples)
            if bias_samples.shape[1:] != (6,) or not np.isfinite(bias_samples).all():
                raise ValueError("Leader wrench bias must be finite 6D.")
            bias = bias_samples.mean(axis=0)
            self.robot.SwitchMode(flexivrdk.Mode.NRT_CARTESIAN_MOTION_FORCE)
            self.robot.SetForceControlAxis([False] * 6)
            self.robot.SetCartesianImpedance(
                (
                    np.asarray(self.robot.info().K_x_nom) * self.config.impedance_scale
                ).tolist()
            )
            self._target = pose.copy()
            self._wrench_bias = bias.copy()
            self._started = True
            self._rebase = True
        except BaseException:
            self.close()
            raise

    def set_enabled(self, enabled: bool) -> None:
        """Apply the external takeover switch; re-anchor on each rising edge."""
        if self._closed:
            raise RuntimeError("A closed leader drag source cannot take over.")
        enabled = bool(enabled)
        if enabled and not self.enabled:
            self._rebase = True
        self.enabled = enabled

    def poll(self, now_ns: int) -> InterventionCommand | None:
        """Advance leader admittance and optionally map drag to follower delta."""
        if not self._started:
            raise RuntimeError("Start the leader drag source before polling.")
        try:
            state = self.robot.states()
            leader = _pose(state.tcp_pose)
            wrench = np.asarray(state.ext_wrench_in_world, dtype=np.float64)
            if wrench.shape != (6,) or not np.isfinite(wrench).all():
                raise ValueError("Leader wrench must be finite 6D.")
            self._target = self._admittance_step(self._target, leader, wrench)
            self.robot.SendCartesianMotionForce(
                self._target.tolist(),
                [0.0] * 6,
                [0.0] * 6,
                self.config.max_linear_speed_m_s,
                self.config.max_angular_speed_rad_s,
                self.config.max_linear_acceleration_m_s2,
                self.config.max_angular_acceleration_rad_s2,
            )
            if not self.enabled:
                return None
            follower = _pose(self.follower_pose())
            if self._rebase:
                self._leader_anchor = leader.copy()
                self._follower_anchor = follower.copy()
                self._rebase = False
            delta_position = np.diag([-1.0, -1.0, 1.0]) @ (
                leader[:3] - self._leader_anchor[:3]
            )
            leader_rotation_delta = (
                flexiv_rotation(leader) * flexiv_rotation(self._leader_anchor).inv()
            )
            target_rotation = (
                self._axis
                * leader_rotation_delta
                * self._axis.inv()
                * flexiv_rotation(self._follower_anchor)
            )
            follower_delta = np.concatenate(
                (
                    self._follower_anchor[:3] + delta_position - follower[:3],
                    (target_rotation * flexiv_rotation(follower).inv()).as_rotvec(),
                )
            )
            return InterventionCommand(
                timestamp_ns=now_ns,
                valid_until_ns=now_ns + int(self.config.command_lifetime_s * 1e9),
                mode="override",
                arm="follower",
                eef_delta=follower_delta,
                source="leader_drag",
            )
        except BaseException:
            self.close()
            raise

    def _admittance_step(
        self, target: np.ndarray, actual: np.ndarray, measured_wrench: np.ndarray
    ) -> np.ndarray:
        cfg = self.config
        period = 1.0 / cfg.control_hz
        alpha = 1.0 - np.exp(-2.0 * np.pi * cfg.velocity_filter_hz * period)
        wrench = measured_wrench - self._wrench_bias
        force, torque = wrench[:3].copy(), wrench[3:].copy()
        force[np.abs(force) < cfg.force_deadband_n] = 0.0
        torque[np.abs(torque) < cfg.torque_deadband_nm] = 0.0
        desired_linear = _limit(cfg.linear_gain * force, cfg.max_linear_speed_m_s)
        desired_angular = _limit(cfg.angular_gain * torque, cfg.max_angular_speed_rad_s)
        linear_change = alpha * (desired_linear - self._linear_velocity)
        angular_change = alpha * (desired_angular - self._angular_velocity)
        self._linear_velocity += _limit(
            linear_change, cfg.max_linear_acceleration_m_s2 * period
        )
        self._angular_velocity += _limit(
            angular_change, cfg.max_angular_acceleration_rad_s2 * period
        )
        result = target.copy()
        result[:3] += self._linear_velocity * period
        quaternion = (
            Rotation.from_rotvec(self._angular_velocity * period)
            * flexiv_rotation(result)
        ).as_quat()
        result[3:] = [quaternion[3], *quaternion[:3]]
        result[:3] = actual[:3] + _limit(result[:3] - actual[:3], cfg.target_leash_m)
        error = flexiv_rotation(result) * flexiv_rotation(actual).inv()
        quaternion = (
            Rotation.from_rotvec(_limit(error.as_rotvec(), cfg.target_leash_rad))
            * flexiv_rotation(actual)
        ).as_quat()
        result[3:] = [quaternion[3], *quaternion[:3]]
        return result

    def close(self) -> None:
        """Stop the leader once; the caller separately closes the follower."""
        if self._closed:
            return
        self._closed = True
        if self._started or self.owns_robot:
            self.robot.Stop()
        self._started = False
        self.enabled = False


class ThreadedLeaderDrag:
    """Poll and command the leader off the follower's 30 Hz control thread."""

    def __init__(self, source: LeaderDragInterventionSource) -> None:
        self.source = source
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._enabled = False
        self._latest: InterventionCommand | None = None
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Activate the leader during bootstrap, then poll it in the background."""
        if self._thread is not None:
            return
        self.source.start()
        self._thread = threading.Thread(target=self._run, name="flexiv-leader-drag")
        self._thread.start()

    def set_enabled(self, enabled: bool) -> None:
        """Queue an explicit takeover switch without waiting for RDK calls."""
        with self._lock:
            self._enabled = bool(enabled)

    def _run(self) -> None:
        period = 1.0 / self.source.config.control_hz
        deadline = time.perf_counter()
        try:
            while not self._stop.is_set():
                with self._lock:
                    enabled = self._enabled
                self.source.set_enabled(enabled)
                command = self.source.poll(time.perf_counter_ns())
                with self._lock:
                    self._latest = command
                deadline += period
                self._stop.wait(max(0.0, deadline - time.perf_counter()))
        except BaseException as exc:
            with self._lock:
                self._error = exc

    def poll(self, now_ns: int) -> InterventionCommand | None:
        """Return the most recent unexpired command without calling the RDK."""
        with self._lock:
            if self._error is not None:
                raise RuntimeError("Leader drag worker failed.") from self._error
            command = self._latest
        if command is None or now_ns > command.valid_until_ns:
            return None
        return command

    def close(self) -> None:
        """Stop polling before closing the leader exactly once."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self.source.close()
