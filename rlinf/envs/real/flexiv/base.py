# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Flexiv Rizon environment with two RGB views and configurable action space."""

from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import gymnasium as gym
import numpy as np

from rlinf.algorithms.online_residual_rl import ActionRepresentation
from rlinf.envs.real.utils.seeding import seed_sampled_spaces
from rlinf.robotics.actions import ActionKind, ActionPart
from rlinf.scheduler import WorkerInfo
from rlinf.utils.logging import get_logger

from .action import apply_world_delta, denormalize_gripper, normalize_gripper
from .gripper_command import LatestGripperCommand


@dataclass
class FlexivEnvConfig:
    """Robot, camera, and command limits for one Flexiv Rizon arm."""

    is_dummy: bool = False
    task_description: str = "complete the manipulation task"
    serial_number: str = ""
    gripper_name: str = "Flexiv-GN01"
    action_representation: str = "eef_delta"
    control_hz: float = 30.0
    max_num_steps: int = 300
    gripper_open_width_m: float = 0.060
    gripper_close_width_m: float = 0.003
    gripper_velocity_m_s: float = 0.05
    gripper_force_n: float = 20.0
    gripper_command_threshold: float = 0.01
    max_linear_velocity_m_s: float | None = None
    max_angular_velocity_rad_s: float | None = None
    max_linear_acceleration_m_s2: float | None = None
    max_angular_acceleration_rad_s2: float | None = None
    max_joint_velocity_rad_s: list[float] = field(default_factory=lambda: [0.2] * 7)
    max_joint_acceleration_rad_s2: list[float] = field(
        default_factory=lambda: [0.5] * 7
    )
    joint_limit_low_rad: list[float] = field(default_factory=lambda: [-6.283] * 7)
    joint_limit_high_rad: list[float] = field(default_factory=lambda: [6.283] * 7)
    max_translation_delta_m: float = 0.02
    max_rotation_delta_rad: float = 0.10
    camera_timeout_ms: int = 2000
    cameras: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            "main": {
                "serial_number": "",
                "width": 640,
                "height": 480,
                "fps": 30,
                "crop_width": 224,
                "crop_height": 224,
                "crop_offset_x": -50,
                "crop_offset_y": 40,
            },
            "wrist": {
                "serial_number": "",
                "width": 640,
                "height": 480,
                "fps": 30,
                "crop_width": 480,
                "crop_height": 480,
                "crop_offset_x": 20,
                "crop_offset_y": 0,
                "resize_width": 224,
                "resize_height": 224,
            },
        }
    )

    def __post_init__(self) -> None:
        self.action_representation = ActionRepresentation(
            self.action_representation
        ).value
        if self.control_hz <= 0.0:
            raise ValueError("control_hz must be positive.")
        cartesian_limits = (
            self.max_linear_velocity_m_s,
            self.max_angular_velocity_rad_s,
            self.max_linear_acceleration_m_s2,
            self.max_angular_acceleration_rad_s2,
        )
        if any(value is not None and value <= 0.0 for value in cartesian_limits):
            raise ValueError(
                "Configured Cartesian velocity and acceleration limits must be positive."
            )
        if tuple(self.cameras) != ("main", "wrist"):
            raise ValueError(
                "Flexiv online residual control requires main then wrist cameras."
            )
        if len(self.max_joint_velocity_rad_s) != 7:
            raise ValueError("max_joint_velocity_rad_s must have seven values.")
        if len(self.max_joint_acceleration_rad_s2) != 7:
            raise ValueError("max_joint_acceleration_rad_s2 must have seven values.")
        if len(self.joint_limit_low_rad) != 7 or len(self.joint_limit_high_rad) != 7:
            raise ValueError("Joint limits must each contain seven values.")


class _RealSensePair:
    """Capture two RGB streams without blocking the robot control thread."""

    def __init__(self, configs: dict[str, dict[str, Any]], timeout_ms: int) -> None:
        import pyrealsense2 as rs

        self._rs = rs
        self._timeout_ms = timeout_ms
        self._configs = configs
        self._pipelines: dict[str, Any] = {}
        self._latest: dict[str, np.ndarray] = {}
        self._error: BaseException | None = None
        self._stopping = False
        self._condition = threading.Condition()
        self._threads: list[threading.Thread] = []
        try:
            for name, values in configs.items():
                pipeline = rs.pipeline()
                config = rs.config()
                serial = str(values["serial_number"])
                if not serial:
                    raise ValueError(f"cameras.{name}.serial_number is required.")
                config.enable_device(serial)
                config.enable_stream(
                    rs.stream.color,
                    int(values["width"]),
                    int(values["height"]),
                    rs.format.rgb8,
                    int(values["fps"]),
                )
                pipeline.start(config)
                self._pipelines[name] = pipeline
        except BaseException:
            for pipeline in self._pipelines.values():
                pipeline.stop()
            self._pipelines.clear()
            raise
        for name in ("main", "wrist"):
            thread = threading.Thread(
                target=self._capture,
                args=(name,),
                name=f"flexiv-camera-{name}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    @staticmethod
    def _crop_resize(image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
        crop_width = int(config.get("crop_width", image.shape[1]))
        crop_height = int(config.get("crop_height", image.shape[0]))
        x = (image.shape[1] - crop_width) // 2 + int(config.get("crop_offset_x", 0))
        y = (image.shape[0] - crop_height) // 2 + int(config.get("crop_offset_y", 0))
        x = max(0, min(x, image.shape[1] - crop_width))
        y = max(0, min(y, image.shape[0] - crop_height))
        image = image[y : y + crop_height, x : x + crop_width]
        width = int(config.get("resize_width", crop_width))
        height = int(config.get("resize_height", crop_height))
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(image, dtype=np.uint8)

    def _capture(self, name: str) -> None:
        try:
            while True:
                with self._condition:
                    if self._stopping:
                        return
                frames = self._pipelines[name].wait_for_frames(
                    timeout_ms=self._timeout_ms
                )
                color = frames.get_color_frame()
                if not color:
                    raise RuntimeError(f"Flexiv camera {name!r} returned no RGB frame.")
                image = self._crop_resize(
                    np.asanyarray(color.get_data()), self._configs[name]
                )
                with self._condition:
                    self._latest[name] = image
                    self._condition.notify_all()
        except BaseException as exc:
            with self._condition:
                if not self._stopping:
                    self._error = exc
                self._condition.notify_all()

    def read(self) -> dict[str, np.ndarray]:
        """Return latest frames, waiting only for the initial pair."""
        deadline = time.monotonic() + self._timeout_ms / 1000.0
        with self._condition:
            while len(self._latest) < 2 and self._error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "Timed out waiting for initial Flexiv RGB frames."
                    )
                self._condition.wait(remaining)
            if self._error is not None:
                raise RuntimeError("Flexiv RGB capture failed.") from self._error
            return {
                name: np.array(self._latest[name], copy=True)
                for name in ("main", "wrist")
            }

    def close(self) -> None:
        """Stop both pipelines; repeated calls are safe."""
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        pipelines, self._pipelines = self._pipelines, {}
        for pipeline in pipelines.values():
            pipeline.stop()
        for thread in self._threads:
            thread.join(timeout=2.0)
        self._threads.clear()


class FlexivEnv(gym.Env):
    """Control one Rizon with EEF deltas or absolute joint targets."""

    TELEOP = ("spacemouse",)
    TELEOP_DEFAULT = "none"
    TELEOP_MARK_FLAG = True
    ACTION_WRAPPERS = ()
    TRANSFORMS = ()

    def __init__(
        self,
        override_cfg: dict[str, Any],
        worker_info: Optional[WorkerInfo] = None,
        robot_info: Any = None,
        env_idx: int = 0,
    ) -> None:
        del worker_info, robot_info, env_idx
        self._logger = get_logger()
        self.config = FlexivEnvConfig(**override_cfg)
        self.task_description = self.config.task_description
        self._steps = 0
        self._robot = None
        self._gripper = None
        self._gripper_executor = None
        self._cameras = None
        self._last_gripper = 1.0
        self._dummy_q = np.zeros(7, dtype=np.float64)
        self._dummy_pose = np.array([0.5, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0])
        self._init_spaces()
        if not self.config.is_dummy:
            self._connect()

    def _init_spaces(self) -> None:
        representation = ActionRepresentation(self.config.action_representation)
        if representation is ActionRepresentation.EEF_DELTA:
            low = np.asarray(
                [-self.config.max_translation_delta_m] * 3
                + [-self.config.max_rotation_delta_rad] * 3
                + [0.0],
                dtype=np.float32,
            )
            high = np.asarray(
                [self.config.max_translation_delta_m] * 3
                + [self.config.max_rotation_delta_rad] * 3
                + [1.0],
                dtype=np.float32,
            )
        else:
            low = np.asarray(self.config.joint_limit_low_rad + [0.0], dtype=np.float32)
            high = np.asarray(
                self.config.joint_limit_high_rad + [1.0], dtype=np.float32
            )
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)
        frame_spaces = {
            name: gym.spaces.Box(
                0,
                255,
                shape=(
                    int(
                        camera.get(
                            "resize_height", camera.get("crop_height", camera["height"])
                        )
                    ),
                    int(
                        camera.get(
                            "resize_width", camera.get("crop_width", camera["width"])
                        )
                    ),
                    3,
                ),
                dtype=np.uint8,
            )
            for name, camera in self.config.cameras.items()
        }
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        # RealWorldEnv flattens state fields in sorted key order.
                        # Keep the seven arm joints before the scalar gripper.
                        "arm_joint_position": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,), dtype=np.float64
                        ),
                        "gripper_position": gym.spaces.Box(
                            0.0, 1.0, shape=(1,), dtype=np.float64
                        ),
                    }
                ),
                "frames": gym.spaces.Dict(frame_spaces),
            }
        )
        self._base_observation_space = copy.deepcopy(self.observation_space)

    def _connect(self) -> None:
        try:
            import flexivrdk
        except ImportError as exc:
            raise RuntimeError(
                "Install Flexiv RDK Python bindings on the robot node."
            ) from exc
        if not self.config.serial_number:
            raise ValueError("serial_number is required for a real Flexiv run.")
        robot = None
        gripper_executor = None
        cameras = None
        try:
            robot = flexivrdk.Robot(self.config.serial_number)
            if not robot.connected() or robot.fault() or not robot.operational():
                raise RuntimeError(
                    "Flexiv robot is not connected, operational, and fault-free."
                )
            gripper = flexivrdk.Gripper(robot)
            tool = flexivrdk.Tool(robot)
            gripper.Enable(self.config.gripper_name)
            tool.Switch(self.config.gripper_name)
            representation = ActionRepresentation(self.config.action_representation)
            if representation is ActionRepresentation.EEF_DELTA:
                robot.SwitchMode(flexivrdk.Mode.NRT_CARTESIAN_MOTION_FORCE)
                robot.SetForceControlAxis([False] * 6)
            else:
                robot.SwitchMode(flexivrdk.Mode.NRT_JOINT_POSITION)
            gripper_executor = LatestGripperCommand(
                gripper,
                velocity_m_s=self.config.gripper_velocity_m_s,
                force_n=self.config.gripper_force_n,
            )
            cameras = _RealSensePair(self.config.cameras, self.config.camera_timeout_ms)
            self._robot = robot
            self._gripper = gripper
            self._gripper_executor = gripper_executor
            self._cameras = cameras
        except BaseException:
            if gripper_executor is not None:
                gripper_executor.close()
            if cameras is not None:
                cameras.close()
            if robot is not None:
                robot.Stop()
            raise

    def action_parts(self) -> tuple[ActionPart, ...]:
        """Describe the configured arm command plus independent scalar gripper."""
        representation = ActionRepresentation(self.config.action_representation)
        kind = (
            ActionKind.CARTESIAN_DELTA
            if representation is ActionRepresentation.EEF_DELTA
            else ActionKind.JOINT_POSITION
        )
        arm_dim = 6 if representation is ActionRepresentation.EEF_DELTA else 7
        return (
            ActionPart("arm", arm_dim, kind),
            ActionPart("end_effector", 1, ActionKind.GRIPPER),
        )

    def _read_state(self) -> tuple[np.ndarray, np.ndarray, float]:
        if self.config.is_dummy:
            return self._dummy_q.copy(), self._dummy_pose.copy(), self._last_gripper
        state = self._robot.states()
        grip = self._gripper.states()
        value = normalize_gripper(
            float(grip.width),
            closed_m=self.config.gripper_close_width_m,
            open_m=self.config.gripper_open_width_m,
        )
        return np.asarray(state.q), np.asarray(state.tcp_pose), value

    def _observation(self) -> dict[str, Any]:
        q, _, gripper = self._read_state()
        if self.config.is_dummy:
            frames = {
                name: np.zeros(
                    self.observation_space["frames"][name].shape, dtype=np.uint8
                )
                for name in ("main", "wrist")
            }
        else:
            frames = self._cameras.read()
        return {
            "state": {
                "arm_joint_position": q.astype(np.float64),
                "gripper_position": np.asarray([gripper], dtype=np.float64),
            },
            "frames": frames,
        }

    def read_control_snapshot(
        self,
    ) -> tuple[np.ndarray, np.ndarray, float, dict[str, np.ndarray]]:
        """Read state and latest frames, waiting only for the initial pair."""
        q, pose, gripper = self._read_state()
        if self.config.is_dummy:
            frames = {
                name: np.zeros(
                    self.observation_space["frames"][name].shape, dtype=np.uint8
                )
                for name in ("main", "wrist")
            }
        else:
            frames = self._cameras.read()
        return q, pose, gripper, frames

    def read_tcp_pose(self) -> np.ndarray:
        """Read the measured TCP pose without capturing either camera."""
        return self._read_state()[1]

    def send_realtime_target(self, target: np.ndarray, gripper: float) -> None:
        """Send one final TCP target without sleeping or reading observations."""
        if self.config.action_representation != ActionRepresentation.EEF_DELTA.value:
            raise RuntimeError("Realtime RTC control requires eef_delta actions.")
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (7,) or not np.all(np.isfinite(target)):
            raise ValueError(
                f"TCP target must be finite shape (7,), got {target.shape}."
            )
        if self.config.is_dummy:
            self._dummy_pose = target.copy()
        else:
            self._send_cartesian_target(target)
        self._submit_gripper(gripper)
        self._steps += 1

    def stop_realtime(self) -> None:
        """Stop robot motion without releasing cameras or the gripper worker."""
        if self._robot is not None:
            self._robot.Stop()

    def _submit_gripper(self, gripper: float) -> None:
        gripper = float(np.clip(gripper, 0.0, 1.0))
        if abs(gripper - self._last_gripper) < self.config.gripper_command_threshold:
            return
        if self._gripper_executor is not None:
            self._gripper_executor.submit(
                denormalize_gripper(
                    gripper,
                    closed_m=self.config.gripper_close_width_m,
                    open_m=self.config.gripper_open_width_m,
                )
            )
        self._last_gripper = gripper

    def _send_cartesian_target(self, target: np.ndarray) -> None:
        """Send a TCP target, leaving omitted limits at the RDK API defaults."""
        configured = (
            self.config.max_linear_velocity_m_s,
            self.config.max_angular_velocity_rad_s,
            self.config.max_linear_acceleration_m_s2,
            self.config.max_angular_acceleration_rad_s2,
        )
        arguments: list[Any] = [target.tolist(), [0.0] * 6, [0.0] * 6]
        defaults = (0.5, 1.0, 2.0, 5.0)
        last_configured = max(
            (index for index, value in enumerate(configured) if value is not None),
            default=-1,
        )
        if last_configured >= 0:
            arguments.extend(
                defaults[index] if value is None else value
                for index, value in enumerate(configured[: last_configured + 1])
            )
        self._robot.SendCartesianMotionForce(*arguments)

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Send one accepted high-level command and return its execution receipt."""
        started = time.monotonic()
        action = np.asarray(action, dtype=np.float64)
        if not self.action_space.contains(action.astype(np.float32)):
            raise ValueError(f"Action is outside {self.action_space}: {action}.")
        q, pose, _ = self._read_state()
        representation = ActionRepresentation(self.config.action_representation)
        if representation is ActionRepresentation.EEF_DELTA:
            delta = action[:6].copy()
            delta[:3] = np.clip(
                delta[:3],
                -self.config.max_translation_delta_m,
                self.config.max_translation_delta_m,
            )
            delta[3:] = np.clip(
                delta[3:],
                -self.config.max_rotation_delta_rad,
                self.config.max_rotation_delta_rad,
            )
            target = apply_world_delta(pose, delta)
            final_action = np.concatenate((delta, action[6:7]))
            if self.config.is_dummy:
                self._dummy_pose = target
            else:
                self._send_cartesian_target(target)
        else:
            target = action[:7]
            final_action = action.copy()
            if self.config.is_dummy:
                self._dummy_q = target.copy()
            else:
                self._robot.SendJointPosition(
                    target.tolist(),
                    [0.0] * 7,
                    self.config.max_joint_velocity_rad_s,
                    self.config.max_joint_acceleration_rad_s2,
                )

        self._submit_gripper(float(action[-1]))

        self._steps += 1
        elapsed = time.monotonic() - started
        remaining = 1.0 / self.config.control_hz - elapsed
        if remaining > 0.0:
            time.sleep(remaining)
        observation = self._observation()
        truncated = self._steps >= self.config.max_num_steps
        info = {
            "frame_index": self._steps - 1,
            "action_receipt": final_action.astype(np.float32),
            "action_receipt_physical": np.concatenate(
                (
                    final_action[:-1],
                    [
                        denormalize_gripper(
                            float(final_action[-1]),
                            closed_m=self.config.gripper_close_width_m,
                            open_m=self.config.gripper_open_width_m,
                        )
                    ],
                )
            ).astype(np.float32),
            "action_modified": not np.array_equal(final_action, action),
        }
        return observation, 0.0, False, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start a manually labelled episode without moving to an unknown pose."""
        del options
        seed_sampled_spaces(seed, self._base_observation_space)
        self._steps = 0
        return self._observation(), {}

    def close(self) -> None:
        """Release the gripper worker, cameras, and robot idempotently."""
        if self._gripper_executor is not None:
            self._gripper_executor.close()
            self._gripper_executor = None
        if self._cameras is not None:
            self._cameras.close()
            self._cameras = None
        if self._robot is not None:
            self._robot.Stop()
            self._robot = None
