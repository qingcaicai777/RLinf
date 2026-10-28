# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Flexiv adapter for standard PI0.5 RTC execution."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from rlinf.models.embodiment.openpi.rtc.types import (
    PhysicalActionStep,
    PolicyObservation,
)

from .base import FlexivEnv, FlexivEnvConfig


@dataclass(frozen=True)
class FlexivRTCConfig:
    """Explicit policy-to-Flexiv observation and action semantics."""

    env: dict[str, Any]
    camera_key_map: dict[str, str]
    translation_unit: str
    rotation_representation: str
    rotation_unit: str
    tcp_delta_frame: str
    translation_delta_limit_m: float
    rotation_delta_limit_rad: float
    gripper_state_range: tuple[float, float]
    gripper_command_range: tuple[float, float]
    gripper_state_open_at_max: bool
    gripper_command_open_at_max: bool
    leader_serial: str | None = None
    enable_tcp_delta_limits: bool = True

    def validate(self) -> None:
        """Validate physical semantics before constructing the hardware env."""
        if set(self.camera_key_map) != {"main", "wrist"}:
            raise ValueError("camera_key_map must map the main and wrist cameras.")
        if len(set(self.camera_key_map.values())) != 2:
            raise ValueError("Policy camera keys must be distinct.")
        if self.translation_unit not in {"m", "mm"}:
            raise ValueError("translation_unit must be m or mm.")
        if self.rotation_representation != "rotvec":
            raise ValueError("Only rotation_representation=rotvec is supported.")
        if self.rotation_unit not in {"rad", "deg"}:
            raise ValueError("rotation_unit must be rad or deg.")
        if self.tcp_delta_frame not in {"world", "tcp"}:
            raise ValueError("tcp_delta_frame must be world or tcp.")
        if self.translation_delta_limit_m <= 0 or self.rotation_delta_limit_rad <= 0:
            raise ValueError("TCP delta limits must be positive.")
        state_min, state_max = self.gripper_state_range
        command_min, command_max = self.gripper_command_range
        if state_min >= state_max or command_min >= command_max:
            raise ValueError("Gripper state and command ranges must increase.")


def _limit_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= maximum:
        return vector
    return vector * (maximum / norm)


def compose_tcp_target(
    reference_pose: np.ndarray,
    policy_delta: np.ndarray,
    config: FlexivRTCConfig,
) -> np.ndarray:
    """Convert one limited delta into a final target from its observation pose."""
    pose = np.asarray(reference_pose, dtype=np.float64)
    delta = np.asarray(policy_delta, dtype=np.float64)
    if pose.shape != (7,) or delta.shape != (6,):
        raise ValueError(
            f"Expected pose (7,) and delta (6,), got {pose.shape}/{delta.shape}."
        )
    if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(delta)):
        raise ValueError("TCP pose or delta contains NaN or Inf.")
    translation = delta[:3] * (0.001 if config.translation_unit == "mm" else 1.0)
    rotvec = delta[3:] * (np.pi / 180.0 if config.rotation_unit == "deg" else 1.0)
    if config.enable_tcp_delta_limits:
        translation = _limit_norm(translation, config.translation_delta_limit_m)
        rotvec = _limit_norm(rotvec, config.rotation_delta_limit_rad)
    current = Rotation.from_quat([pose[4], pose[5], pose[6], pose[3]])
    result = pose.copy()
    if config.tcp_delta_frame == "tcp":
        result[:3] += current.apply(translation)
        target_rotation = current * Rotation.from_rotvec(rotvec)
    else:
        result[:3] += translation
        target_rotation = Rotation.from_rotvec(rotvec) * current
    quaternion = target_rotation.as_quat()
    result[3:] = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]
    return result


def tcp_delta_from_target(
    reference_pose: np.ndarray,
    target_pose: np.ndarray,
    config: FlexivRTCConfig,
) -> np.ndarray:
    """Express a final TCP target in the policy observation's delta frame."""
    reference = np.asarray(reference_pose, dtype=np.float64)
    target = np.asarray(target_pose, dtype=np.float64)
    if reference.shape != (7,) or target.shape != (7,):
        raise ValueError("Reference and target TCP poses must have shape (7,).")
    if not np.isfinite(reference).all() or not np.isfinite(target).all():
        raise ValueError("Reference and target TCP poses must be finite.")
    reference_rotation = Rotation.from_quat(
        [reference[4], reference[5], reference[6], reference[3]]
    )
    target_rotation = Rotation.from_quat([target[4], target[5], target[6], target[3]])
    translation = target[:3] - reference[:3]
    if config.tcp_delta_frame == "tcp":
        translation = reference_rotation.inv().apply(translation)
        rotation = (reference_rotation.inv() * target_rotation).as_rotvec()
    else:
        rotation = (target_rotation * reference_rotation.inv()).as_rotvec()
    translation /= 0.001 if config.translation_unit == "mm" else 1.0
    rotation /= np.pi / 180.0 if config.rotation_unit == "deg" else 1.0
    return np.concatenate((translation, rotation)).astype(np.float32)


def project_flexiv_actions(actions: np.ndarray, config: FlexivRTCConfig) -> np.ndarray:
    """Apply Flexiv command limits without accessing any robot resource."""
    source = np.asarray(actions, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 7 or not np.isfinite(source).all():
        raise ValueError("Physical chunk must be finite shape [steps, 7].")
    result = source.copy()
    translation_scale = 0.001 if config.translation_unit == "mm" else 1.0
    rotation_scale = np.pi / 180.0 if config.rotation_unit == "deg" else 1.0
    for row in result:
        if config.enable_tcp_delta_limits:
            row[:3] = (
                _limit_norm(
                    row[:3] * translation_scale, config.translation_delta_limit_m
                )
                / translation_scale
            )
            row[3:6] = (
                _limit_norm(row[3:6] * rotation_scale, config.rotation_delta_limit_rad)
                / rotation_scale
            )
    low, high = config.gripper_command_range
    result[:, 6] = np.clip(result[:, 6], low, high)
    return result.astype(np.float32)


class FlexivRTCAdapter:
    """Expose low-latency Flexiv operations to the generic RTC runtime."""

    def __init__(self, config: FlexivRTCConfig) -> None:
        config.validate()
        env_config = FlexivEnvConfig(**config.env)
        if env_config.action_representation != "eef_delta":
            raise ValueError("Flexiv RTC requires action_representation=eef_delta.")
        if not env_config.is_dummy:
            if not env_config.serial_number:
                raise ValueError("flexiv.serial_number is required before activation.")
            missing = [
                name
                for name, camera in env_config.cameras.items()
                if not camera.get("serial_number")
            ]
            if missing:
                raise ValueError(
                    f"Flexiv camera serial numbers are missing for {missing}."
                )
        self.config = config
        self._env: FlexivEnv | None = None
        self._target_pose: np.ndarray | None = None
        self._last_gripper = 1.0
        self._stopped = False
        self._intervention = None
        self._leader = None
        self._pose_lock = threading.Lock()
        self._pose_snapshot: np.ndarray | None = None
        if config.leader_serial and (
            config.translation_unit != "m"
            or config.rotation_unit != "rad"
            or config.tcp_delta_frame != "world"
        ):
            raise ValueError("Leader drag requires world-frame metre/radian deltas.")

    def _cached_pose(self) -> np.ndarray:
        with self._pose_lock:
            if self._pose_snapshot is None:
                raise RuntimeError("Follower pose is not available.")
            return self._pose_snapshot.copy()

    def _remember_pose(self, pose: np.ndarray) -> None:
        with self._pose_lock:
            self._pose_snapshot = np.asarray(pose, dtype=np.float64).copy()

    def start(self) -> None:
        """Activate cameras, robot, and gripper after policy validation."""
        if self._env is not None:
            return
        self._env = FlexivEnv(self.config.env)
        _, pose, gripper, _ = self._env.read_control_snapshot()
        self._target_pose = np.asarray(pose, dtype=np.float64).copy()
        self._remember_pose(pose)
        self._last_gripper = float(gripper)
        self._stopped = False
        if self.config.leader_serial:
            from .intervention import FlexivInterventionDevice
            from .leader_drag import LeaderDragInterventionSource, ThreadedLeaderDrag

            try:
                source = LeaderDragInterventionSource.connect(
                    self._env,
                    self.config.leader_serial,
                    follower_pose=self._cached_pose,
                )
                self._leader = ThreadedLeaderDrag(source)
                self._leader.start()
                self._intervention = FlexivInterventionDevice(
                    self._leader,
                    clip_to_action_space=self.config.enable_tcp_delta_limits,
                )
            except BaseException:
                self.close()
                raise

    def set_intervention_enabled(self, enabled: bool) -> None:
        """Apply an operator switch edge on the robot-owning control thread."""
        if self._leader is None:
            if enabled:
                raise RuntimeError("Leader drag is not configured.")
            return
        self._leader.set_enabled(enabled)

    def _map_gripper_state(self, open_fraction: float) -> float:
        low, high = self.config.gripper_state_range
        alpha = (
            open_fraction
            if self.config.gripper_state_open_at_max
            else 1.0 - open_fraction
        )
        return float(low + np.clip(alpha, 0.0, 1.0) * (high - low))

    def _map_gripper_command(self, command: float) -> float:
        low, high = self.config.gripper_command_range
        alpha = np.clip((command - low) / (high - low), 0.0, 1.0)
        return float(alpha if self.config.gripper_command_open_at_max else 1.0 - alpha)

    def project_policy_actions(self, actions: np.ndarray) -> np.ndarray:
        """Project a future physical chunk to the commands this adapter accepts."""
        return project_flexiv_actions(actions, self.config)

    def read_observation(self, *, tick: int, prompt: str) -> PolicyObservation:
        """Snapshot 7D joints, gripper state, and the latest two RGB frames."""
        del tick
        if self._env is None:
            raise RuntimeError("Flexiv RTC adapter is not started.")
        joints, _pose, gripper, frames = self._env.read_control_snapshot()
        self._remember_pose(_pose)
        state = np.concatenate(
            [
                np.asarray(joints, dtype=np.float32),
                np.asarray([self._map_gripper_state(gripper)], dtype=np.float32),
            ]
        )
        images = {
            policy_key: frames[robot_key]
            for robot_key, policy_key in self.config.camera_key_map.items()
        }
        return PolicyObservation.snapshot(state, images, prompt, _pose)

    def read_tcp_pose(self) -> np.ndarray:
        """Read the current TCP reference without waiting for camera frames."""
        if self._env is None:
            raise RuntimeError("Flexiv RTC adapter is not started.")
        pose = self._env.read_tcp_pose()
        self._remember_pose(pose)
        return pose

    def apply_policy_action(self, step: PhysicalActionStep) -> dict[str, Any]:
        """Send one accepted command and return its physical execution receipt."""
        if self._env is None or self._target_pose is None:
            raise RuntimeError("Flexiv RTC adapter is not started.")
        if step.tcp_reference_pose is None:
            raise ValueError(
                "Flexiv actions require an observation TCP reference pose."
            )
        physical = project_flexiv_actions(
            np.concatenate((step.tcp_delta, [step.gripper]))[None], self.config
        )[0]
        policy_target = compose_tcp_target(
            step.tcp_reference_pose, physical[:6], self.config
        )
        proposal = physical.copy()
        current_pose = self._cached_pose()
        current_delta = tcp_delta_from_target(current_pose, policy_target, self.config)
        command = np.concatenate(
            (current_delta, [self._map_gripper_command(float(physical[6]))])
        )
        mask = np.zeros(7, dtype=bool)
        if self._intervention is not None:
            sample = self._intervention.read(self._env, command)
            if sample.active and sample.action is not None:
                command = np.asarray(sample.action, dtype=np.float64)
                mask = np.asarray(sample.info["teleop_action_mask"], dtype=bool)
        low, high = self.config.gripper_command_range
        fraction = float(np.clip(command[6], 0.0, 1.0))
        if not self.config.gripper_command_open_at_max:
            fraction = 1.0 - fraction
        physical[6] = low + fraction * (high - low)
        if mask[:6].any():
            if self.config.tcp_delta_frame != "world":
                raise ValueError("Leader drag requires world-frame TCP deltas.")
            self._target_pose = compose_tcp_target(
                current_pose, command[:6], self.config
            )
            physical[:6] = tcp_delta_from_target(
                step.tcp_reference_pose, self._target_pose, self.config
            )
        else:
            self._target_pose = policy_target
        self._last_gripper = self._map_gripper_command(float(physical[6]))
        self._env.send_realtime_target(self._target_pose, self._last_gripper)
        tolerance = np.asarray([1e-4] * 3 + [1e-3] * 3 + [0.001])
        mask &= np.abs(physical - proposal) > tolerance
        return {
            "policy_proposal_physical": proposal,
            "policy_proposal_valid": True,
            "policy_proposal_tcp_target_pose": policy_target.copy(),
            "policy_proposal_gripper_width_m": float(proposal[6]),
            "action_receipt_physical": physical.copy(),
            "tcp_target_pose": self._target_pose.copy(),
            "human_mask": mask.copy(),
            "pair_mask": mask.copy(),
            "receipt_valid": True,
        }

    def hold_target(self) -> None:
        """Resend the final target without applying an expired delta."""
        if self._env is None or self._target_pose is None:
            raise RuntimeError("Flexiv RTC adapter is not started.")
        self._env.send_realtime_target(self._target_pose, self._last_gripper)

    def stop(self) -> None:
        """Stop robot motion; repeated calls are safe."""
        if self._stopped:
            return
        try:
            if self._env is not None:
                self._env.stop_realtime()
        finally:
            try:
                if self._intervention is not None:
                    self._intervention.close()
                    self._intervention = None
                    self._leader = None
                elif self._leader is not None:
                    self._leader.close()
                    self._leader = None
            finally:
                self._stopped = True

    def close(self) -> None:
        """Close the gripper worker, camera workers, and RDK connection."""
        if self._intervention is not None:
            self._intervention.close()
            self._intervention = None
            self._leader = None
        elif self._leader is not None:
            self._leader.close()
            self._leader = None
        if self._env is None:
            return
        self._env.close()
        self._env = None
        self._target_pose = None
