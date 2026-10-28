# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Flexiv-side 30 Hz controller; importable without PyTorch or RLinf imports."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from queue import Full, Queue
from threading import Thread
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

_ROOT = Path(__file__).resolve().parents[2]
_WIRE = _ROOT / "rlinf" / "algorithms" / "online_residual_rl"
_FLEXIV = _ROOT / "rlinf" / "envs" / "real" / "flexiv"
if str(_WIRE) not in sys.path:
    sys.path.insert(0, str(_WIRE))
if str(_FLEXIV) not in sys.path:
    sys.path.insert(0, str(_FLEXIV))
from action import binarize_gripper  # noqa: E402
from dataset_home import read_dataset_joint_home  # noqa: E402
from gripper_command import LatestGripperCommand  # noqa: E402
from remote_client import RemotePolicy  # noqa: E402
from robot_runtime import (  # noqa: E402
    RemoteRobotRuntime,
    RobotControlResult,
    RobotObservation,
    RobotTickRequest,
)


class ProcessedImageRecorder:
    """Write immutable RGB observation pairs outside the control thread."""

    def __init__(self, output_dir: Path) -> None:
        import cv2

        self.output_dir = output_dir / f"session_{time.time_ns()}_{os.getpid()}"
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self._queue: Queue[
            tuple[int, np.ndarray, np.ndarray, dict[str, Any] | None] | None
        ] = Queue(maxsize=4)
        self._closed = False
        self._cv2 = cv2
        self._thread = Thread(
            target=self._write_loop, name="vla-rgb-capture", daemon=True
        )
        self._thread.start()
        logging.info("Saving processed VLA RGB observations to %s", self.output_dir)

    def submit(
        self,
        tick: int,
        main_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        *,
        observation: dict[str, Any] | None = None,
    ) -> bool:
        """Queue RGB and optional observation metadata without waiting for disk I/O."""
        try:
            self._queue.put_nowait((tick, main_rgb, wrist_rgb, observation))
        except Full:
            logging.warning(
                "Dropping RGB capture at observation tick %d: queue full", tick
            )
            return False
        return True

    def _write_loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                tick, main_rgb, wrist_rgb, observation = item
                for name, image_rgb in (("main", main_rgb), ("wrist", wrist_rgb)):
                    output_path = self.output_dir / f"tick_{tick:06d}_{name}.png"
                    image_bgr = self._cv2.cvtColor(image_rgb, self._cv2.COLOR_RGB2BGR)
                    if not self._cv2.imwrite(str(output_path), image_bgr):
                        raise OSError(f"OpenCV could not write {output_path}")
                if observation is not None:
                    output_path = self.output_dir / f"tick_{tick:06d}_observation.json"
                    output_path.write_text(
                        json.dumps(observation, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8",
                    )
            except Exception:
                logging.exception("Failed to save processed RGB capture")
            finally:
                self._queue.task_done()

    def close(self) -> None:
        """Drain queued images and stop the writer; safe to call repeatedly."""
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join()


def load_myq_config(path: Path) -> dict[str, Any]:
    """Validate the fixed dual-arm Home, camera and timing contract."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        config["control"]["leader"] != "right"
        or int(config["control"]["control_hz"]) != 30
    ):
        raise ValueError("This deployment requires myq's right leader at 30 Hz.")
    control = config["control"]
    if control.get("xy_motion_direction", "opposite") not in {"same", "opposite"}:
        raise ValueError("control.xy_motion_direction must be same or opposite.")
    scale = np.asarray(control.get("follower_position_scale", [1, 1, 1]), dtype=float)
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError(
            "control.follower_position_scale requires three positive finite values."
        )
    limits = control.get("follower_orientation_max_deviation_deg")
    if limits is not None:
        angles = np.asarray(limits, dtype=float)
        if (
            angles.shape != (3,)
            or not np.isfinite(angles).all()
            or np.any(angles < 0)
            or np.any(angles >= 90)
        ):
            raise ValueError(
                "Orientation limits require three finite angles in [0, 90)."
            )
    for key in ("force_bias_before_manual_seconds", "force_bias_after_home_seconds"):
        value = float(config.get("recording", {}).get(key, 0.5))
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"recording.{key} must be finite and nonnegative.")
    home_config = config.get("home", {})
    joint_home = [
        home_config.get(f"{name}_joint_positions_rad") for name in ("left", "right")
    ]
    if any(value is not None for value in joint_home):
        for name, value in zip(("left", "right"), joint_home):
            joints = np.asarray(value, dtype=float)
            if joints.shape != (7,) or not np.isfinite(joints).all():
                raise ValueError(
                    f"myq home.{name}_joint_positions_rad must be finite [7]."
                )
    else:
        home = np.asarray(home_config.get("left_tcp_pose_wxyz"), dtype=float)
        if home.shape != (7,) or not np.isfinite(home).all():
            raise ValueError(
                "myq requires both joint Homes or a finite left TCP Home [7]."
            )
    if set(config["cameras"]) != {"main", "wrist"}:
        raise ValueError("myq must configure main and wrist cameras.")
    for name, camera in config["cameras"].items():
        if not camera.get("serial_number"):
            raise ValueError(f"myq cameras.{name}.serial_number is required.")
        if int(camera.get("crop_width", camera["width"])) > int(camera["width"]) or int(
            camera.get("crop_height", camera["height"])
        ) > int(camera["height"]):
            raise ValueError(f"myq cameras.{name} crop exceeds the stream size.")
        width = int(
            camera.get("resize_width", camera.get("crop_width", camera["width"]))
        )
        height = int(
            camera.get("resize_height", camera.get("crop_height", camera["height"]))
        )
        if width <= 0 or height <= 0:
            raise ValueError(f"myq cameras.{name} has an invalid image size.")
    return config


def compose_world_delta(reference: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Apply one Stage1 metre/radian world-frame delta to its observation pose."""
    pose = np.asarray(reference, dtype=np.float64)
    step = np.asarray(delta, dtype=np.float64)
    if (
        pose.shape != (7,)
        or step.shape != (6,)
        or not np.isfinite(pose).all()
        or not np.isfinite(step).all()
    ):
        raise ValueError("TCP reference must be finite [7] and delta finite [6].")
    rotation = Rotation.from_quat([pose[4], pose[5], pose[6], pose[3]])
    result = pose.copy()
    result[:3] += step[:3]
    quat = (Rotation.from_rotvec(step[3:]) * rotation).as_quat()
    result[3:] = [quat[3], quat[0], quat[1], quat[2]]
    return result


def world_delta_from_target(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Record a final accepted target as a Stage1-frame physical delta."""
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    if (
        a.shape != (7,)
        or b.shape != (7,)
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("TCP poses must be finite [7].")
    ar = Rotation.from_quat([a[4], a[5], a[6], a[3]])
    br = Rotation.from_quat([b[4], b[5], b[6], b[3]])
    return np.concatenate((b[:3] - a[:3], (br * ar.inv()).as_rotvec())).astype(
        np.float32
    )


class FlexivRobotAdapter:
    """Own myq cameras, both Flexiv arms and the follower gripper."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        myq_path: Path,
        capture_dir: Path | None = None,
        capture_count: int = 3,
        capture_interval_seconds: float = 5.0,
        initialize_gripper: bool | None = None,
        action_representation: str = "eef_delta",
        joint_control: Mapping[str, Any] | None = None,
        home_dataset_path: str | Path | None = None,
        gripper_policy: Mapping[str, Any] | None = None,
        auto_fine: Mapping[str, Any] | None = None,
        record_diagnostics: bool = False,
    ):
        self._command_times = {}
        self.record_diagnostics = record_diagnostics
        self._observation_timing: dict[str, Any] = {}
        if action_representation not in {"eef_delta", "joint_absolute"}:
            raise ValueError("Unsupported Flexiv action representation.")
        self.action_representation = action_representation
        self._dataset_home: np.ndarray | None = None
        if home_dataset_path is not None:
            if action_representation != "joint_absolute":
                raise ValueError("Dataset Home requires joint_absolute actions.")
            right_home = np.asarray(
                config.get("home", {}).get("right_joint_positions_rad"), dtype=float
            )
            if right_home.shape != (7,) or not np.isfinite(right_home).all():
                raise ValueError("Dataset Home requires a finite right joint Home.")
            episode, self._dataset_home = read_dataset_joint_home(home_dataset_path)
            logging.info(
                "Dataset Home episode=%s frame=0 left_joint_positions_rad=%s",
                episode,
                self._dataset_home.tolist(),
            )
        self.joint_control = dict(joint_control or {})
        self._control_phase = "coarse"
        self.auto_fine = dict(auto_fine) if auto_fine is not None else None
        self._auto_fine_ticks = 0
        self._fine_requested = False
        if self.auto_fine is not None:
            unknown = set(self.auto_fine) - {
                "xy_center_mm",
                "xy_radius_mm",
                "z_below_mm",
                "consecutive_ticks",
            }
            center = np.asarray(self.auto_fine.get("xy_center_mm"), dtype=float)
            radius = float(self.auto_fine.get("xy_radius_mm", float("nan")))
            height = float(self.auto_fine.get("z_below_mm", float("nan")))
            ticks = self.auto_fine.get("consecutive_ticks", 3)
            if (
                unknown
                or center.shape != (2,)
                or not np.isfinite(center).all()
                or not np.isfinite([radius, height]).all()
                or radius <= 0
                or isinstance(ticks, bool)
                or not isinstance(ticks, int)
                or ticks <= 0
            ):
                raise ValueError(
                    "auto_fine requires finite xy_center_mm [2], positive xy_radius_mm, "
                    "finite z_below_mm and positive integer consecutive_ticks."
                )
            self.auto_fine = {
                "xy_center_mm": center,
                "xy_radius_mm": radius,
                "z_below_mm": height,
                "consecutive_ticks": ticks,
            }
        self._fine_motion_limit_scale = 0.5
        self._joint_max_velocity: np.ndarray | None = None
        if action_representation == "joint_absolute":
            scale = self.joint_control.get("fine_motion_limit_scale", 0.5)
            if (
                isinstance(scale, bool)
                or not isinstance(scale, (int, float))
                or not np.isfinite(scale)
                or not 0 < scale <= 1
            ):
                raise ValueError(
                    "joint_control.fine_motion_limit_scale must be finite and in (0, 1]."
                )
            self._fine_motion_limit_scale = float(scale)
            for key in ("max_joint_velocity_rad_s", "max_joint_acceleration_rad_s2"):
                if (
                    key == "max_joint_velocity_rad_s"
                    and self.joint_control.get(key) is None
                ):
                    continue
                values = np.asarray(self.joint_control.get(key), dtype=float)
                if (
                    values.shape != (7,)
                    or not np.isfinite(values).all()
                    or np.any(values <= 0)
                ):
                    raise ValueError(
                        f"joint_control.{key} must contain seven positive finite values."
                    )
                if key == "max_joint_velocity_rad_s":
                    self._joint_max_velocity = values.copy()
        self.gripper_policy = dict(gripper_policy or {})
        unknown = set(self.gripper_policy) - {
            "binary",
            "threshold",
            "hold_arm_on_close",
        }
        if unknown:
            raise ValueError(f"Unknown gripper_policy fields: {sorted(unknown)}")
        self.binary_gripper = self.gripper_policy.get("binary", False)
        if not isinstance(self.binary_gripper, bool):
            raise ValueError("gripper_policy.binary must be a boolean.")
        self.hold_arm_on_close = self.gripper_policy.get("hold_arm_on_close", True)
        if not isinstance(self.hold_arm_on_close, bool):
            raise ValueError("gripper_policy.hold_arm_on_close must be a boolean.")
        self.gripper_threshold = float(self.gripper_policy.get("threshold", 0.5))
        if not np.isfinite(self.gripper_threshold):
            raise ValueError("gripper_policy.threshold must be finite.")
        if self.binary_gripper:
            follower = config["robots"]["left"]
            binarize_gripper(
                self.gripper_threshold,
                current_width_m=float(follower["gripper_open_width_m"]),
                closed_m=float(follower["gripper_close_width_m"]),
                open_m=float(follower["gripper_open_width_m"]),
                threshold=self.gripper_threshold,
            )
        self.joint_target: np.ndarray | None = None
        self.joint_limits: tuple[np.ndarray, np.ndarray] | None = None
        self.config = config
        self.myq_path = myq_path
        self.robots: dict[str, Any] = {}
        self.gripper: Any = None
        self._gripper_worker: LatestGripperCommand | None = None
        self.cameras: Any = None
        self.console: Any = None
        self.target: dict[str, np.ndarray] = {}
        self.gripper_target = float(config["robots"]["left"]["gripper_open_width_m"])
        self._initialize_gripper = initialize_gripper
        self._closed = False
        self._stopped = False
        self._paused = False
        self._maintenance: str | None = None
        self._maintenance_started = 0.0
        self._pause_future: Future | None = None
        self._pause_thread: Thread | None = None
        self._bias_samples: dict[str, dict[str, list]] = {}
        self._bias_next = "paused"
        self._home_joints: dict[str, np.ndarray] = {}
        self._home_gripper_move = None
        self._binary_close_move = None
        self._binary_close_ack_at: float | None = None
        self._resume_after_gripper_close = False
        self._resume_coarse_after_home = False
        self._orientation_limit = None
        self._manual = False
        self._manual_gripper = False
        self._joint_drag_controllers: dict[str, Any] = {}
        self.fine_available = False
        self.continuous_episodes = False
        self._episode_ended = False
        self._requested_gripper: float | None = None
        self._capture_dir = capture_dir
        self._captures_remaining = capture_count
        if capture_count <= 0 or capture_interval_seconds <= 0:
            raise ValueError(
                "capture_count and capture_interval_seconds must be positive."
            )
        self._capture_interval_seconds = capture_interval_seconds
        self._next_capture_at: float | None = None
        self.image_recorder: ProcessedImageRecorder | None = None

    def _command(self, name: str, method: str, *args) -> None:
        before = time.perf_counter_ns()
        getattr(self.robots[name], method)(*args)
        self._command_times[name] = (before, time.perf_counter_ns())

    def _send(self, targets: dict[str, np.ndarray]) -> None:
        control = self.config["control"]
        args = (
            [0.0] * 6,
            [0.0] * 6,
            float(control["max_linear_velocity_m_s"]),
            float(control["max_angular_velocity_rad_s"]),
            float(control["max_linear_acceleration_m_s2"]),
            float(control["max_angular_acceleration_rad_s2"]),
        )
        for name in ("right", "left"):
            self._command(
                name, "SendCartesianMotionForce", targets[name].tolist(), *args
            )
        self.target = {name: target.copy() for name, target in targets.items()}

    def _connect_gripper(self, flexivrdk: Any, control_cls: Any) -> Any:
        """Connect and select the gripper tool before resolving TCP Home."""
        follower = self.config["robots"]["left"]
        gripper_config = self.config["gripper_control"]
        self.gripper = flexivrdk.Gripper(self.robots["left"])
        self.gripper.Enable(str(follower["gripper_name"]))
        flexivrdk.Tool(self.robots["left"]).Switch(str(follower["gripper_name"]))
        grip_control = control_cls(follower, gripper_config)
        grip_control.validate_hardware_params(self.gripper.params())
        return grip_control

    def _initialize_gripper_after_home(
        self, grip_control: Any, initialize: Any
    ) -> None:
        """Initialize the selected gripper after both arms reach Home."""
        gripper_config = self.config["gripper_control"]
        if self._initialize_gripper is None:
            should_initialize = bool(gripper_config.get("initialize_on_start", True))
        else:
            should_initialize = self._initialize_gripper
        if should_initialize:
            initialize(
                self.gripper,
                grip_control,
                {**gripper_config, "initialize_on_start": True},
                "left",
            )
            if bool(gripper_config.get("open_on_start", True)):
                target = float(self.config["robots"]["left"]["gripper_open_width_m"])
                timeout = float(gripper_config.get("initialization_timeout_s", 15.0))
                if not np.isfinite(timeout) or timeout <= 0:
                    raise ValueError(
                        "Gripper opening timeout must be positive and finite."
                    )
                deadline = time.perf_counter() + timeout
                while True:
                    state = self.gripper.states()
                    width = float(state.width)
                    if (
                        np.isfinite(width)
                        and abs(width - target) <= 0.001
                        and not getattr(state, "is_moving", False)
                    ):
                        break
                    if time.perf_counter() >= deadline:
                        raise RuntimeError(
                            "Startup gripper did not reach its open target; inference has not started."
                        )
                    time.sleep(0.02)
        self.gripper_target = self.measured_gripper_width()
        self._gripper_worker = LatestGripperCommand(
            self.gripper,
            velocity_m_s=float(gripper_config["velocity_m_s"]),
            force_n=float(gripper_config["force_limit_n"]),
        )

    def _start_gripper(self, flexivrdk: Any, control_cls: Any, initialize: Any) -> None:
        """Connect and initialize the gripper for isolated lifecycle callers."""
        grip_control = self._connect_gripper(flexivrdk, control_cls)
        self._initialize_gripper_after_home(grip_control, initialize)

    def _move_to_joint_home(
        self, flexivrdk: Any, joints: dict[str, np.ndarray]
    ) -> None:
        """Execute myq's bounded trajectory and wait for both arms to arrive."""
        from myq.home_control import JointHomeReturn

        control = self.config["control"]
        infos = {name: robot.info() for name, robot in self.robots.items()}
        start = {
            name: np.asarray(robot.states().q) for name, robot in self.robots.items()
        }
        trajectory = JointHomeReturn(start, joints, infos, control)
        tolerance = np.deg2rad(float(control.get("home_joint_tolerance_deg", 0.5)))
        timeout = float(control["home_settle_timeout_s"])
        if (
            not np.isfinite(tolerance)
            or tolerance <= 0
            or not np.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError(
                "Joint Home tolerance and settle timeout must be positive and finite."
            )
        for robot in self.robots.values():
            robot.SwitchMode(flexivrdk.Mode.NRT_JOINT_POSITION)
        begin = time.perf_counter()
        deadline = begin
        period = 1.0 / float(control["control_hz"])
        while True:
            elapsed = time.perf_counter() - begin
            for name, target in trajectory.targets(elapsed).items():
                self.robots[name].SendJointPosition(
                    target.tolist(),
                    [0.0] * 7,
                    np.minimum(
                        float(control["joint_max_velocity_rad_s"]), infos[name].dq_max
                    ).tolist(),
                    [float(control["joint_max_acceleration_rad_s2"])] * 7,
                )
            if elapsed >= trajectory.duration_s:
                measured = {
                    name: robot.states().q for name, robot in self.robots.items()
                }
                if trajectory.reached(measured, tolerance):
                    break
                if elapsed > trajectory.duration_s + timeout:
                    raise RuntimeError("Both Flexiv arms did not reach myq joint Home.")
            deadline += period
            time.sleep(max(0.0, deadline - time.perf_counter()))

    def _return_joint_home(self, flexivrdk: Any) -> dict[str, np.ndarray]:
        """Return to saved Home and apply fixed world-frame arm offsets."""
        from myq.home_control import (
            JointHomeReturn,
            home_joints_from_config,
            home_offset_from_config,
            offset_joint_home,
        )

        control = self.config["control"]
        joints = home_joints_from_config(self.config)
        if self._dataset_home is not None:
            if joints is None:
                raise ValueError("Dataset Home requires a configured right joint Home.")
            joints["left"] = self._dataset_home.copy()
        infos = {name: robot.info() for name, robot in self.robots.items()}
        start = {
            name: np.asarray(robot.states().q) for name, robot in self.robots.items()
        }
        JointHomeReturn(start, joints, infos, control)
        deferred_left = False
        for name in ("left", "right"):
            if name == "left" and self._dataset_home is not None:
                continue
            if not np.any(home_offset_from_config(control, name)):
                continue
            if name == "left" and self.config["home"].get("left_tcp_pose_wxyz") is None:
                deferred_left = True
                continue
            joints = offset_joint_home(
                self.config,
                joints,
                flexivrdk.Model(self.robots[name]),
                infos[name],
                name,
            )
            logging.info(
                "Home offset arm=%s world_translation_m=%s",
                name,
                home_offset_from_config(control, name).tolist(),
            )
        self._move_to_joint_home(flexivrdk, joints)
        if deferred_left:
            # Capture q and TCP together at the saved left Home; never use the
            # arbitrary startup pose as an offset reference.
            state = self.robots["left"].states()
            baseline = {name: q.copy() for name, q in joints.items()}
            baseline["left"] = np.asarray(state.q, dtype=float).copy()
            offset_config = {
                **self.config,
                "home": {
                    **self.config["home"],
                    "left_tcp_pose_wxyz": list(state.tcp_pose),
                },
            }
            joints = offset_joint_home(
                offset_config,
                baseline,
                flexivrdk.Model(self.robots["left"]),
                infos["left"],
                "left",
            )
            logging.info(
                "Home offset arm=left world_translation_m=%s",
                home_offset_from_config(control, "left").tolist(),
            )
            self._move_to_joint_home(flexivrdk, joints)
        self._home_joints = {name: q.copy() for name, q in joints.items()}
        home = {
            name: np.asarray(robot.states().tcp_pose, dtype=float)
            for name, robot in self.robots.items()
        }
        for robot in self.robots.values():
            robot.SwitchMode(flexivrdk.Mode.NRT_CARTESIAN_MOTION_FORCE)
            robot.SetForceControlAxis([False] * 6)
        return home

    def start(self) -> None:
        """Use myq's exact camera, gripper, force-mode and Home lifecycle."""
        if self._capture_dir is not None:
            self.image_recorder = ProcessedImageRecorder(self._capture_dir)
        import flexivrdk
        from myq.camera_capture import CameraCapture
        from myq.control_alignment import (
            PoseLowPass,
            WorldDeltaAligner,
            interpolate_pose,
            pose_rotation,
        )
        from myq.drag_runtime import (
            ConsoleCommands,
            SmoothedAdmittance,
            _force_bias,
            _initialize_and_open_follower_gripper,
        )
        from myq.follower_gripper import FollowerGripperControl

        self.cameras = CameraCapture(self.config["cameras"])
        self.cameras.start()
        self.console = ConsoleCommands(self.config)
        self.console.start()
        for name in ("left", "right"):
            robot = flexivrdk.Robot(str(self.config["robots"][name]["serial_number"]))
            if not robot.connected() or robot.fault() or not robot.operational():
                raise RuntimeError(f"Flexiv {name} is not ready.")
            self.robots[name] = robot
        grip_control = self._connect_gripper(flexivrdk, FollowerGripperControl)
        self.force_bias = _force_bias(
            self.robots, float(self.config["recording"].get("force_bias_seconds", 1.0))
        )
        for robot in self.robots.values():
            robot.SwitchMode(flexivrdk.Mode.NRT_CARTESIAN_MOTION_FORCE)
            robot.SetForceControlAxis([False] * 6)
        right = self.robots["right"]
        right.SetCartesianImpedance(
            (
                np.asarray(right.info().K_x_nom)
                * float(self.config["control"]["leader_impedance_scale"])
            ).tolist()
        )
        control = self.config["control"]
        period = 1.0 / float(control["control_hz"])
        if (
            self._dataset_home is not None
            or self.config["home"].get("left_joint_positions_rad") is not None
        ):
            home = self._return_joint_home(flexivrdk)
            right.SetCartesianImpedance(
                (
                    np.asarray(right.info().K_x_nom)
                    * float(control["leader_impedance_scale"])
                ).tolist()
            )
        else:
            from myq.drag_runtime import _home_poses_from_left, _pose_reached

            home = _home_poses_from_left(
                np.asarray(self.config["home"]["left_tcp_pose_wxyz"], dtype=float),
                float(control["home_right_x_offset_m"]),
                float(control["home_right_y_offset_m"]),
                float(control["home_right_z_offset_m"]),
                float(control["home_right_yaw_deg"]),
            )
            start = {
                name: np.asarray(robot.states().tcp_pose, dtype=float)
                for name, robot in self.robots.items()
            }
            duration = max(
                float(control["home_return_duration_s"]),
                *(
                    np.linalg.norm(home[name][:3] - start[name][:3])
                    / float(control["max_linear_velocity_m_s"])
                    for name in home
                ),
                *(
                    (
                        pose_rotation(home[name]) * pose_rotation(start[name]).inv()
                    ).magnitude()
                    / float(control["max_angular_velocity_rad_s"])
                    for name in home
                ),
            )
            begin = time.perf_counter()
            period = 1.0 / float(control["control_hz"])
            deadline = begin
            while True:
                alpha = min(1.0, (time.perf_counter() - begin) / duration)
                self._send(
                    {
                        name: interpolate_pose(start[name], home[name], alpha)
                        for name in home
                    }
                )
                if alpha >= 1.0:
                    reached = all(
                        _pose_reached(
                            self.robots[name].states().tcp_pose,
                            home[name],
                            float(control["home_position_tolerance_m"]),
                            np.deg2rad(
                                float(control["home_orientation_tolerance_deg"])
                            ),
                        )
                        for name in home
                    )
                    if reached:
                        break
                    if time.perf_counter() > begin + duration + float(
                        control["home_settle_timeout_s"]
                    ):
                        raise RuntimeError("Both Flexiv arms did not reach myq Home.")
                deadline += period
                time.sleep(max(0.0, deadline - time.perf_counter()))
        if not self._home_joints:
            self._home_joints = {
                name: np.asarray(robot.states().q, dtype=float).copy()
                for name, robot in self.robots.items()
            }
        self._initialize_gripper_after_home(
            grip_control, _initialize_and_open_follower_gripper
        )
        after_home = float(
            self.config["recording"].get("force_bias_after_home_seconds", 0.5)
        )
        if after_home > 0:
            self.force_bias = _force_bias(self.robots, after_home)
        self.target = {name: pose.copy() for name, pose in home.items()}
        if self.action_representation == "joint_absolute":
            follower = self.robots["left"]
            info = follower.info()
            low, high = (
                np.asarray(info.q_min, dtype=float),
                np.asarray(info.q_max, dtype=float),
            )
            if (
                low.shape != (7,)
                or high.shape != (7,)
                or not np.isfinite([low, high]).all()
                or np.any(low >= high)
            ):
                raise ValueError(
                    "Flexiv SDK joint limits must be seven finite increasing pairs."
                )
            self.joint_limits = (low, high)
            if self._joint_max_velocity is None:
                velocity = np.asarray(info.dq_max, dtype=float)
                if (
                    velocity.shape != (7,)
                    or not np.isfinite(velocity).all()
                    or np.any(velocity <= 0)
                ):
                    raise ValueError(
                        "Flexiv SDK joint velocity limits must be seven positive finite values."
                    )
                self._joint_max_velocity = velocity.copy()
            self.joint_target = np.asarray(follower.states().q, dtype=float).copy()
            if (
                self.joint_target.shape != (7,)
                or not np.isfinite(self.joint_target).all()
                or np.any(self.joint_target < low)
                or np.any(self.joint_target > high)
            ):
                raise ValueError(
                    "Follower joint state is invalid or outside SDK limits."
                )
            follower.SwitchMode(flexivrdk.Mode.NRT_JOINT_POSITION)
            logging.info(
                "Joint policy motion limits: velocity_rad_s=%s acceleration_rad_s2=%s",
                self._joint_max_velocity.tolist(),
                self.joint_control["max_joint_acceleration_rad_s2"],
            )
        self.aligner = WorldDeltaAligner(
            "right",
            "left",
            home["right"],
            home["left"],
            xy_motion_direction=str(control.get("xy_motion_direction", "opposite")),
            position_scale=control.get("follower_position_scale", [1.0, 1.0, 1.0]),
        )
        self.admittance = SmoothedAdmittance(period, control)
        self.follower_filter = PoseLowPass(
            float(control["follower_target_filter_hz"]), period, home["left"]
        )

    def snapshot(
        self, *, images: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Read joints, TCP and latest processed RGB pair without camera waits."""
        arm = self.robots["left"].states()
        state_read_ns = time.perf_counter_ns()
        gripper_config = self.config["robots"]["left"]
        grip = float(
            np.clip(
                self.gripper.states().width,
                gripper_config["gripper_close_width_m"],
                gripper_config["gripper_open_width_m"],
            )
        )
        pose = np.asarray(arm.tcp_pose, dtype=np.float64)
        state = np.concatenate((np.asarray(arm.q, dtype=np.float32), [grip])).astype(
            np.float32
        )
        if not images:
            return state, pose, np.empty(0), np.empty(0)
        with self.cameras.condition:
            if not all(self.cameras.buffers[name] for name in ("main", "wrist")):
                return None
            frames = {
                name: self.cameras.buffers[name][-1] for name in ("main", "wrist")
            }
            for name, frame in frames.items():
                process = self.cameras.processes.get(name)
                if process is not None and process.exitcode is not None:
                    raise RuntimeError(
                        f"myq {name} camera process exited unexpectedly."
                    )
                if time.perf_counter_ns() - frame.host_time_ns > 500_000_000:
                    raise RuntimeError(f"myq {name} camera frame is older than 500 ms.")
            main = frames["main"].image_rgb.copy()
            wrist = frames["wrist"].image_rgb.copy()
            if self.record_diagnostics:
                now_ns = time.perf_counter_ns()
                self._observation_timing = {
                    "state_read_perf_counter_ns": state_read_ns,
                    "images_read_perf_counter_ns": now_ns,
                    "camera_host_time_ns": {
                        name: int(frame.host_time_ns) for name, frame in frames.items()
                    },
                    "camera_age_ms": {
                        name: (now_ns - frame.host_time_ns) / 1e6
                        for name, frame in frames.items()
                    },
                }
        if (
            main.shape != self.cameras.shapes["main"]
            or wrist.shape != self.cameras.shapes["wrist"]
        ):
            raise ValueError(
                "Camera image shape differs from myq crop/resize configuration."
            )
        return state, pose, main, wrist

    def read_observation(self, *, tick: int, prompt: str) -> RobotObservation | None:
        """Map myq sensors to PI0.5 inputs and preserve the action reference."""
        snapshot = self.snapshot(images=True)
        if snapshot is None:
            return None
        state, pose, main, wrist = snapshot
        capture_time = time.monotonic()
        if (
            self.image_recorder is not None
            and self._captures_remaining > 0
            and (self._next_capture_at is None or capture_time >= self._next_capture_at)
        ):
            if self.image_recorder.submit(
                tick,
                main,
                wrist,
                observation={
                    "tick": tick,
                    "capture_monotonic_seconds": capture_time,
                    "prompt": prompt,
                    "action_representation": self.action_representation,
                    "states": state.tolist(),
                    "tcp_pose_wxyz": pose.tolist(),
                    "last_joint_target_rad": (
                        None
                        if self.joint_target is None
                        else self.joint_target.tolist()
                    ),
                },
            ):
                self._captures_remaining -= 1
                self._next_capture_at = capture_time + self._capture_interval_seconds
        context = (
            {"training_reference": pose}
            if self.action_representation == "eef_delta"
            else {"action_representation": "joint_absolute"}
        )
        if self.record_diagnostics:
            context["observation_timing"] = self._observation_timing.copy()
            context["tcp_pose_wxyz"] = pose.copy()
        return RobotObservation(
            policy_inputs={
                "states": state,
                "main_images": main,
                "wrist_images": wrist,
            },
            action_context=context,
        )

    def poll_control(self, *, tick: int) -> RobotTickRequest:
        """Handle operator takeover and labels before the inference handoff."""
        del tick
        command = self.console.poll()
        self._requested_gripper = None
        reset_policy = self._resume_after_gripper_close
        reset_to_coarse = False
        self._resume_after_gripper_close = False
        if self._maintenance == "stopping" and command in {
            " ",
            "r",
            "h",
            "o",
            "c",
            "i",
        }:
            logging.getLogger("flexiv.status").info(
                "Pause still stopping the arms; wait for Pause complete before %r",
                command,
            )
            return RobotTickRequest()
        if command == " " and self._maintenance is None:
            if self._episode_ended:
                self._restore_policy_control()
                self._episode_ended = False
                self._paused = self._manual = self._manual_gripper = False
                self._fine_requested = False
                self._auto_fine_ticks = 0
                self._resume_coarse_after_home = False
                logging.info(
                    "Starting another episode in coarse mode; press I at the fine stage"
                )
                return RobotTickRequest(start_episode=True)
            if self._paused:
                self._restore_policy_control()
                self._paused = False
                self._manual = False
                self._manual_gripper = False
                reset_policy = True
                reset_to_coarse = self._resume_coarse_after_home and self.fine_available
                self._resume_coarse_after_home = False
                if reset_to_coarse:
                    self._control_phase = "coarse"
                    self._fine_requested = False
                    self._auto_fine_ticks = 0
                logging.info("Model control resumed; requesting fresh predictions")
            else:
                self._begin_pause()
        elif (
            command == "r"
            and self._paused
            and not self._manual
            and not self._episode_ended
            and self._maintenance is None
        ):
            self.take_over()
        elif command == "h" and self._paused and self._maintenance is None:
            self._begin_home_return()
        elif command in {"s", "f", "d"} and self.continuous_episodes:
            if self._episode_ended:
                return RobotTickRequest()
            if self._maintenance != "stopping":
                self._restore_policy_control()
            self._manual = self._manual_gripper = False
            self._paused = self._episode_ended = True
            self._resume_after_gripper_close = False
            if self._maintenance not in {"binary_close", "stopping"}:
                self._maintenance = None
            logging.info(
                "Episode ended (%s); paused with background inference. Space starts another episode; H optionally returns Home",
                command,
            )
            return RobotTickRequest(
                end_episode=True,
                success=None if command == "d" else command == "s",
                discard_episode=command == "d",
            )
        elif command in {"s", "f", "d", "q"}:
            return RobotTickRequest(
                stop=True,
                success=(True if command == "s" else False if command == "f" else None),
                discard_episode=command == "d",
            )
        elif command == "i":
            if self._paused or self._manual:
                logging.warning(
                    "I was not applied: resume model control with Space before requesting Fine mode"
                )
            elif (
                self.action_representation == "joint_absolute"
                and not self.fine_available
            ):
                logging.warning(
                    "I was not applied: the server does not support Fine mode; use the Stage2 inference config"
                )
            else:
                self._fine_requested = True
                return RobotTickRequest(enter_fine_phase=True)
        elif (
            (self._paused or self._manual)
            and self._maintenance is None
            and command in {"o", "c"}
        ):
            self._manual_gripper = True
            self._requested_gripper = float(
                self.config["robots"]["left"][
                    "gripper_open_width_m"
                    if command == "o"
                    else "gripper_close_width_m"
                ]
            )
        elif command == "r":
            logging.warning(
                "R was not applied: pause an active episode with Space and wait for maintenance to finish"
            )
        # Keep the policy stream alive during takeover so every manual frame
        # has the corresponding policy proposal in its receipt.
        return RobotTickRequest(
            policy_enabled=True,
            reset_policy=reset_policy,
            reset_to_coarse=reset_to_coarse,
            enter_fine_phase=self._request_auto_fine(),
        )

    def _request_auto_fine(self) -> bool:
        """Request the normal Fine handoff once inside the measured TCP region."""
        if (
            self.auto_fine is None
            or not self.fine_available
            or self._control_phase != "coarse"
            or self._fine_requested
            or self._paused
            or self._manual
            or self._episode_ended
            or self._maintenance is not None
        ):
            self._auto_fine_ticks = 0
            return False
        pose = np.asarray(self.robots["left"].states().tcp_pose, dtype=float)
        position_mm = pose[:3] * 1000.0
        inside = (
            pose.shape == (7,)
            and np.isfinite(position_mm).all()
            and np.linalg.norm(position_mm[:2] - self.auto_fine["xy_center_mm"])
            <= self.auto_fine["xy_radius_mm"] + 1e-9
            and position_mm[2] < self.auto_fine["z_below_mm"]
        )
        self._auto_fine_ticks = self._auto_fine_ticks + 1 if inside else 0
        if self._auto_fine_ticks < self.auto_fine["consecutive_ticks"]:
            return False
        self._fine_requested = True
        logging.getLogger("flexiv.status").info(
            "Automatic Fine requested: measured left TCP xyz_mm=%s; waiting for a valid handoff",
            position_mm.tolist(),
        )
        return True

    def control_tick(
        self,
        *,
        tick: int,
        action: np.ndarray | None,
        action_context: Mapping[str, Any] | None,
    ) -> RobotControlResult:
        """Optionally sample measured state immediately before sending a target."""
        if action_context is not None and "control_phase" in action_context:
            phase = action_context["control_phase"]
            if phase not in {"coarse", "fine"}:
                raise ValueError("Control phase must be coarse or fine.")
            if (
                phase != self._control_phase
                and self.action_representation == "joint_absolute"
            ):
                logging.info(
                    "Joint model motion limit scale=%s phase=%s",
                    self._fine_motion_limit_scale if phase == "fine" else 1.0,
                    phase,
                )
            self._control_phase = phase
        measured = self.snapshot(images=False) if self.record_diagnostics else None
        sampled_at = time.perf_counter_ns() if measured is not None else None
        self._command_times.clear()
        if self.robots["left"].fault() or self.robots["right"].fault():
            raise RuntimeError("Flexiv fault during rollout.")
        result = self._maybe_close_binary_gripper(
            tick=tick, action=action, action_context=action_context
        )
        if result is None:
            result = self._control_tick(
                tick=tick, action=action, action_context=action_context
            )
        if "left" in self._command_times:
            started, returned = self._command_times["left"]
            result.receipt["command_started_perf_counter_ns"] = started
            result.receipt["command_returned_perf_counter_ns"] = returned
        if measured is not None:
            state, pose, _, _ = measured
            result.receipt["diagnostics"] = {
                "measured_before_control": state,
                "tcp_pose_wxyz": pose,
                "sampled_perf_counter_ns": sampled_at,
                "command_returned_perf_counter_ns": time.perf_counter_ns(),
                "manual": self._manual,
            }
        return result

    def _maybe_close_binary_gripper(self, *, tick, action, action_context):
        if (
            self.binary_gripper
            and self.hold_arm_on_close
            and not self._manual
            and not self._paused
            and self._maintenance is None
            and not self._resume_after_gripper_close
            and action is not None
        ):
            expected_dim = 8 if self.action_representation == "joint_absolute" else 7
            if (
                np.asarray(action).shape != (expected_dim,)
                or not np.isfinite(action).all()
            ):
                raise ValueError("Binary policy action has invalid shape or values.")
            if self.action_representation == "joint_absolute":
                low, high = self.joint_limits
                if np.any(action[:7] < low) or np.any(action[:7] > high):
                    raise ValueError(
                        "Absolute joint target exceeds Flexiv SDK joint limits."
                    )
            desired_width = self._policy_gripper_width(
                float(action[-1]), tick=tick, action_context=action_context
            )
            closed = float(self.config["robots"]["left"]["gripper_close_width_m"])
            if desired_width == closed and self.gripper_target > closed + 0.001 + 1e-8:
                self._begin_binary_close(closed)
                result = self._paused_tick()
                self._binary_close_move = self._gripper_worker.initialize(
                    self._close_binary_gripper
                )
                return result
        return None

    def _control_tick(
        self,
        *,
        tick: int,
        action: np.ndarray | None,
        action_context: Mapping[str, Any] | None,
    ) -> RobotControlResult:
        """Apply one policy target, manual takeover, or stale-result hold."""
        requested_gripper = self._requested_gripper
        if self.robots["left"].fault() or self.robots["right"].fault():
            raise RuntimeError("Flexiv fault during rollout.")

        if self._maintenance == "binary_close":
            return self._binary_close_tick()
        if self._resume_after_gripper_close:
            return self._paused_tick()
        if self._maintenance is not None:
            return self._maintenance_tick()
        if self._paused and not self._manual:
            return self._paused_tick()
        if self.action_representation == "joint_absolute":
            return self._joint_control_tick(
                action, tick=tick, action_context=action_context
            )
        if self._manual:
            target = self.target["left"]
            width = (
                self.gripper_target if requested_gripper is None else requested_gripper
            )
        elif action is None:
            target = self.target["left"]
            width = self.gripper_target
        else:
            if (
                action_context is None
                or action_context.get("training_reference") is None
            ):
                raise ValueError("Flexiv action is missing its TCP reference pose.")
            reference = np.asarray(action_context["training_reference"], dtype=float)
            if np.asarray(action).shape != (7,):
                raise ValueError("Flexiv policy action must have shape [7].")
            target = compose_world_delta(reference, np.asarray(action)[:6])
            width = self._policy_gripper_width(
                float(action[6]), tick=tick, action_context=action_context
            )
        proposal = None
        if action is not None:
            proposal = np.asarray(action, dtype=np.float32).copy()
            if proposal.shape != (7,) or not np.isfinite(proposal).all():
                raise ValueError("Flexiv policy proposal must be finite [7].")
            proposal[6] = self._accepted_gripper_width(
                self._policy_gripper_width(
                    float(proposal[6]), tick=tick, action_context=action_context
                )
            )
        targets = self.send(target, width, manual=self._manual)
        if action_context is None or action_context.get("training_reference") is None:
            raise ValueError("Flexiv receipt is missing its TCP reference pose.")
        reference = np.asarray(action_context["training_reference"], dtype=float)
        executed_physical = np.concatenate(
            (
                world_delta_from_target(reference, targets["left"]),
                [self.gripper_target],
            )
        ).astype(np.float32)
        human_mask, pair_mask = self._correction_masks(
            executed_physical, proposal, arm_tolerance=[1e-4] * 3 + [1e-3] * 3
        )
        proposal_fields: dict[str, Any] = {
            "policy_proposal_valid": proposal is not None
        }
        if proposal is not None:
            proposal_target = compose_world_delta(reference, proposal[:6])
            proposal_fields.update(
                policy_proposal_physical=proposal,
                policy_proposal_tcp_target_pose=proposal_target.copy(),
                policy_proposal_gripper_width_m=float(proposal[6]),
            )
        return RobotControlResult(
            receipt={
                "action_receipt_physical": executed_physical,
                "tcp_target_pose": targets["left"].copy(),
                "human_mask": human_mask,
                "pair_mask": pair_mask,
                **proposal_fields,
            },
            policy_enabled=True,
        )

    def _accepted_gripper_width(self, width: float) -> float:
        """Return the command accepted by the 1 mm gripper deadband."""
        return (
            float(width)
            if abs(width - self.gripper_target) > 0.001 + 1e-8
            else self.gripper_target
        )

    def _correction_masks(self, executed, proposal, *, arm_tolerance):
        """Mark human-owned dimensions changed from a reliable policy proposal."""
        human = np.zeros(len(executed), dtype=bool)
        if self._manual and proposal is not None:
            human[:-1] = np.abs(executed[:-1] - proposal[:-1]) > np.asarray(
                arm_tolerance
            )
            human[-1] = (
                self._manual_gripper
                and abs(float(executed[-1] - proposal[-1])) > 0.001 + 1e-8
            )
        return human, human.copy()

    def _policy_gripper_width(
        self, width_m: float, *, tick: int, action_context: Mapping[str, Any] | None
    ) -> float:
        follower = self.config["robots"]["left"]
        closed = float(follower["gripper_close_width_m"])
        opened = float(follower["gripper_open_width_m"])
        if self.binary_gripper:
            if (
                action_context is None
                or "normalized_gripper_actions" not in action_context
                or "action_start_frame" not in action_context
            ):
                raise ValueError(
                    "Binary control requires the model's normalized gripper action context."
                )
            values = np.asarray(
                action_context["normalized_gripper_actions"], dtype=np.float64
            )
            index = tick - int(action_context["action_start_frame"])
            if (
                values.ndim != 1
                or not 0 <= index < len(values)
                or not np.isfinite(values[index])
            ):
                raise ValueError(
                    "Normalized gripper action is missing, invalid or out of range for this tick."
                )
            return binarize_gripper(
                float(values[index]),
                current_width_m=self.gripper_target,
                closed_m=closed,
                open_m=opened,
                threshold=self.gripper_threshold,
            )
        return float(np.clip(width_m, closed, opened))

    def _joint_control_tick(
        self,
        action: np.ndarray | None,
        *,
        tick: int = 0,
        action_context: Mapping[str, Any] | None = None,
    ) -> RobotControlResult:
        if (
            self.joint_target is None
            or self.joint_limits is None
            or self._joint_max_velocity is None
        ):
            raise RuntimeError("Joint control has not started.")
        if action is None:
            physical = np.concatenate((self.joint_target, [self.gripper_target]))
        else:
            physical = np.asarray(action, dtype=np.float64).copy()
        if physical.shape != (8,) or not np.isfinite(physical).all():
            raise ValueError("Absolute joint action must be finite shape [8].")
        low, high = self.joint_limits
        if np.any(physical[:7] < low) or np.any(physical[:7] > high):
            raise ValueError("Absolute joint target exceeds Flexiv SDK joint limits.")
        proposal = None
        if action is not None:
            proposal = physical.copy()
            proposal[7] = self._accepted_gripper_width(
                self._policy_gripper_width(
                    float(proposal[7]), tick=tick, action_context=action_context
                )
            )
            physical[7] = proposal[7]
        if self._manual:
            targets = self.manual_targets()
            # Validate both IK results before either arm receives a new command.
            joints = {
                name: controller.command(targets[name], time.perf_counter_ns())
                for name, controller in self._joint_drag_controllers.items()
            }
            physical[:7] = joints["left"]
            physical[7] = (
                self.gripper_target
                if self._requested_gripper is None
                else self._requested_gripper
            )
            right = self._joint_drag_controllers["right"]
            self._command(
                "right",
                "SendJointPosition",
                joints["right"].tolist(),
                [0.0] * 7,
                right.max_velocity.tolist(),
                right.max_acceleration.tolist(),
            )
            self.target = {name: value.copy() for name, value in targets.items()}
        controller = self._joint_drag_controllers.get("left") if self._manual else None
        motion_scale = (
            self._fine_motion_limit_scale if self._control_phase == "fine" else 1.0
        )
        self._command(
            "left",
            "SendJointPosition",
            physical[:7].tolist(),
            [0.0] * 7,
            controller.max_velocity.tolist()
            if controller
            else (self._joint_max_velocity * motion_scale).tolist(),
            controller.max_acceleration.tolist()
            if controller
            else (
                np.asarray(self.joint_control["max_joint_acceleration_rad_s2"])
                * motion_scale
            ).tolist(),
        )
        self.joint_target = physical[:7].copy()
        if abs(float(physical[7]) - self.gripper_target) > 0.001 + 1e-8:
            assert self._gripper_worker is not None
            logging.info(
                "joint gripper target tick=%d width_m=%.4f previous_width_m=%.4f",
                tick,
                float(physical[7]),
                self.gripper_target,
            )
            self._submit_gripper_target(float(physical[7]), manual=self._manual)
            self.gripper_target = float(physical[7])
        physical[7] = self.gripper_target
        human_mask, pair_mask = self._correction_masks(
            physical, proposal, arm_tolerance=[1e-3] * 7
        )
        return RobotControlResult(
            receipt={
                "action_receipt_physical": physical.astype(np.float32),
                "joint_target_rad": self.joint_target.copy(),
                "action_representation": "joint_absolute",
                "receipt_valid": True,
                "human_mask": human_mask,
                "pair_mask": pair_mask,
                "policy_proposal_physical": None
                if proposal is None
                else proposal.astype(np.float32),
                "policy_proposal_valid": action is not None,
            }
        )

    def hold_target(self) -> None:
        """Resend the last accepted target without advancing a policy delta."""
        if self.action_representation == "joint_absolute":
            self._joint_control_tick(None)
        else:
            self._send(self.target)

    def manual_targets(self) -> dict[str, np.ndarray]:
        """Rebased right-arm drag produces the left-arm final target."""
        right_state = self.robots["right"].states()
        right_pose = np.asarray(right_state.tcp_pose, dtype=float)
        wrench = (
            np.asarray(right_state.ext_wrench_in_world, dtype=float)
            - self.force_bias["right"]["world"]
        )
        right_target = self.admittance.step(self.target["right"], wrench, right_pose)
        left_target = self.follower_filter.update(
            self.aligner.follower_target(right_target)
        )
        if self._orientation_limit is not None:
            left_target = self._orientation_limit.apply(left_target)
            self.follower_filter.reset(left_target)
        return {"right": right_target, "left": left_target}

    def _close_drag_controllers(self) -> None:
        controllers, self._joint_drag_controllers = self._joint_drag_controllers, {}
        failures = []
        for name, controller in controllers.items():
            try:
                controller.close()
            except Exception as exc:
                self._joint_drag_controllers[name] = controller
                failures.append(exc)
        if failures:
            raise failures[0]

    def _restore_policy_control(self) -> None:
        import flexivrdk

        self._close_drag_controllers()
        self.target = {
            name: np.asarray(robot.states().tcp_pose, dtype=float).copy()
            for name, robot in self.robots.items()
        }
        for name, robot in self.robots.items():
            if name == "left" and self.action_representation == "joint_absolute":
                robot.SwitchMode(flexivrdk.Mode.NRT_JOINT_POSITION)
                self.joint_target = np.asarray(robot.states().q, dtype=float).copy()
            else:
                robot.SwitchMode(flexivrdk.Mode.NRT_CARTESIAN_MOTION_FORCE)
                robot.SetForceControlAxis([False] * 6)
                if name == "right":
                    robot.SetCartesianImpedance(
                        (
                            np.asarray(robot.info().K_x_nom)
                            * float(self.config["control"]["leader_impedance_scale"])
                        ).tolist()
                    )

    def _begin_pause(self) -> None:
        """Stop in the background before restoring hold control at the settled pose."""
        self._paused = True
        self._maintenance = "stopping"
        self._pause_future = Future()
        self._pause_thread = Thread(target=self._stop_for_pause, name="flexiv-pause")
        self._pause_thread.start()
        logging.getLogger("flexiv.status").info(
            "Pause requested; stopping both arms before capturing hold positions"
        )

    def _stop_for_pause(self) -> None:
        try:
            # Independent SDK connections stop concurrently; await both even
            # when one fails before restoring any control mode.
            with ThreadPoolExecutor(max_workers=len(self.robots)) as executor:
                stops = [executor.submit(robot.Stop) for robot in self.robots.values()]
                failures = []
                for stop in stops:
                    try:
                        stop.result()
                    except Exception as exc:
                        failures.append(exc)
                if failures:
                    raise failures[0]
            self._restore_policy_control()
        except BaseException as exc:
            self._pause_future.set_exception(exc)
        else:
            self._pause_future.set_result(None)

    def _paused_tick(self) -> RobotControlResult:
        if self._requested_gripper is not None:
            width = self._accepted_gripper_width(self._requested_gripper)
            if width != self.gripper_target:
                self._gripper_worker.submit(width)
                self.gripper_target = width
        if self.action_representation == "joint_absolute":
            result = self._joint_control_tick(None)
            receipt = dict(result.receipt)
        else:
            self._send(self.target)
            receipt = {
                "action_receipt_physical": np.r_[
                    np.zeros(6), self.gripper_target
                ].astype(np.float32),
                "tcp_target_pose": self.target["left"].copy(),
            }
        width = 8 if self.action_representation == "joint_absolute" else 7
        receipt.update(
            schedule_valid=False,
            policy_proposal_valid=False,
            human_mask=np.zeros(width, dtype=bool),
            pair_mask=np.zeros(width, dtype=bool),
        )
        return RobotControlResult(receipt=receipt)

    def _begin_binary_close(self, closed_width: float) -> None:
        """Freeze measured arm poses before requesting a binary closure."""
        states = {name: robot.states() for name, robot in self.robots.items()}
        self.target = {
            name: np.asarray(state.tcp_pose, dtype=float).copy()
            for name, state in states.items()
        }
        if self.action_representation == "joint_absolute":
            self.joint_target = np.asarray(states["left"].q, dtype=float).copy()
        self.gripper_target = closed_width
        self._maintenance = "binary_close"
        self._maintenance_started = time.perf_counter()
        self._binary_close_ack_at = None
        logging.info("Binary gripper closing at 50 N; holding measured arm pose")

    def _close_binary_gripper(self) -> float:
        width = float(self.config["robots"]["left"]["gripper_close_width_m"])
        velocity = float(self.config["gripper_control"]["velocity_m_s"])
        self.gripper.Move(width, velocity, 50.0)
        return width

    def _binary_close_tick(self) -> RobotControlResult:
        result = self._paused_tick()
        now = time.perf_counter()
        if now - self._maintenance_started > 30.0:
            raise RuntimeError(
                "Binary gripper closure timed out; model control remains held."
            )
        if self._binary_close_move.done():
            self._binary_close_move.result()
            if self._binary_close_ack_at is None:
                self._binary_close_ack_at = now
            # Move acknowledges delivery; allow state telemetry to catch up.
            if (
                now - self._binary_close_ack_at >= 0.1
                and not self.gripper.states().is_moving
            ):
                self._maintenance = None
                self._resume_after_gripper_close = not self._episode_ended
                logging.info(
                    "Binary gripper closure finished; requesting fresh model predictions"
                )
        return result

    def take_over(self) -> None:
        """Start stationary force zeroing before enabling R-key manual drag."""
        if not self._paused or self._maintenance is not None:
            raise RuntimeError("Pause model control before starting manual drag.")
        self._begin_force_zero("manual")
        logging.info(
            "Manual takeover: keep both arms stationary and free of contact while zeroing force"
        )

    def _begin_force_zero(self, next_state: str) -> None:
        self._maintenance = "zero"
        self._maintenance_started = time.perf_counter()
        self._bias_next = next_state
        self._bias_samples = {name: {"world": [], "tcp": []} for name in self.robots}

    def _begin_home_return(self) -> None:
        import flexivrdk
        from myq.home_control import JointHomeReturn

        if not self._home_joints:
            raise RuntimeError("Home targets have not been initialized.")
        self._resume_coarse_after_home = True
        self._manual = False
        self._manual_gripper = False
        self._close_drag_controllers()
        infos = {name: robot.info() for name, robot in self.robots.items()}
        start = {
            name: np.asarray(robot.states().q, dtype=float)
            for name, robot in self.robots.items()
        }
        self._home_return = JointHomeReturn(
            start, self._home_joints, infos, self.config["control"]
        )
        for robot in self.robots.values():
            robot.SwitchMode(flexivrdk.Mode.NRT_JOINT_POSITION)
        self._maintenance_started = time.perf_counter()
        self._maintenance = "home"
        logging.info("Returning both arms Home; model control remains paused")

    def _open_home_gripper(self) -> float:
        """Send the H-key gripper target without running calibration."""
        config = self.config["gripper_control"]
        self.gripper.Move(
            0.06, float(config["velocity_m_s"]), float(config["force_limit_n"])
        )
        return 0.06

    def _maintenance_tick(self) -> RobotControlResult:
        if self._maintenance == "stopping":
            if self._pause_future.done():
                self._pause_future.result()
                self._pause_thread.join()
                self._pause_thread = self._pause_future = None
                self._maintenance = None
                logging.getLogger("flexiv.status").info(
                    "Pause complete; holding settled positions. R starts manual drag, H returns Home, Space resumes"
                )
                return self._paused_tick()
            width = 8 if self.action_representation == "joint_absolute" else 7
            return RobotControlResult(
                receipt={
                    "action_receipt_physical": np.zeros(width, dtype=np.float32),
                    "schedule_valid": False,
                    "receipt_valid": False,
                    "policy_proposal_valid": False,
                    "human_mask": np.zeros(width, bool),
                    "pair_mask": np.zeros(width, bool),
                }
            )
        if self._maintenance == "home":
            elapsed = time.perf_counter() - self._maintenance_started
            targets = self._home_return.targets(elapsed)
            control = self.config["control"]
            for name, target in targets.items():
                robot = self.robots[name]
                self._command(
                    name,
                    "SendJointPosition",
                    target.tolist(),
                    [0.0] * 7,
                    np.minimum(
                        float(control["joint_max_velocity_rad_s"]), robot.info().dq_max
                    ).tolist(),
                    [float(control["joint_max_acceleration_rad_s2"])] * 7,
                )
            measured = {name: robot.states().q for name, robot in self.robots.items()}
            tolerance = np.deg2rad(float(control.get("home_joint_tolerance_deg", 0.5)))
            if elapsed >= self._home_return.duration_s and self._home_return.reached(
                measured, tolerance
            ):
                self._restore_policy_control()
                self._orientation_limit = None
                self.admittance.reset()
                self.follower_filter.reset(self.target["left"])
                self.aligner.rebase(self.target["right"], self.target["left"])
                self._home_gripper_move = self._gripper_worker.initialize(
                    self._open_home_gripper
                )
                self._maintenance = "open_gripper"
            elif elapsed > self._home_return.duration_s + float(
                control["home_settle_timeout_s"]
            ):
                raise RuntimeError("Both Flexiv arms did not reach Home.")
            width = 8 if self.action_representation == "joint_absolute" else 7
            physical = (
                np.r_[targets["left"], self.gripper_target]
                if width == 8
                else np.r_[np.zeros(6), self.gripper_target]
            )
            return RobotControlResult(
                receipt={
                    "action_receipt_physical": physical.astype(np.float32),
                    "schedule_valid": False,
                    "receipt_valid": False,
                    "policy_proposal_valid": False,
                    "human_mask": np.zeros(width, bool),
                    "pair_mask": np.zeros(width, bool),
                }
            )
        result = self._paused_tick()
        if self._maintenance == "open_gripper":
            if self._home_gripper_move.done():
                self.gripper_target = float(self._home_gripper_move.result())
                self._begin_force_zero("paused")
        elif self._maintenance == "zero":
            for name, robot in self.robots.items():
                state = robot.states()
                for frame, attribute in (
                    ("world", "ext_wrench_in_world"),
                    ("tcp", "ext_wrench_in_tcp"),
                ):
                    value = np.asarray(getattr(state, attribute), dtype=float)
                    if value.shape != (6,) or not np.isfinite(value).all():
                        raise ValueError(
                            "Force zeroing requires finite six-axis wrenches."
                        )
                    self._bias_samples[name][frame].append(value.copy())
            recording = self.config.get("recording", {})
            key = (
                "force_bias_before_manual_seconds"
                if self._bias_next == "manual"
                else "force_bias_after_home_seconds"
            )
            duration = float(
                recording.get(key, recording.get("force_bias_after_home_seconds", 0.5))
            )
            if not np.isfinite(duration) or duration < 0:
                raise ValueError(
                    "Force zeroing duration must be finite and nonnegative."
                )
            if time.perf_counter() - self._maintenance_started >= duration:
                self.force_bias = {
                    name: {
                        frame: np.mean(samples, axis=0)
                        for frame, samples in frames.items()
                    }
                    for name, frames in self._bias_samples.items()
                }
                self._maintenance = None
                if self._bias_next == "manual":
                    self._enter_manual()
                logging.info(
                    "Force bias reset completed; mode=%s",
                    "manual" if self._manual else "paused",
                )
        return result

    def _enter_manual(self) -> None:
        """Rebase measured TCP anchors after force zeroing, then enable joint drag."""
        self._manual = True
        poses = {
            name: np.asarray(robot.states().tcp_pose, dtype=float)
            for name, robot in self.robots.items()
        }
        self.aligner.rebase(poses["right"], poses["left"])
        self.admittance.reset()
        self.follower_filter.reset(poses["left"])
        self.target = poses
        limits = self.config["control"].get("follower_orientation_max_deviation_deg")
        if self._orientation_limit is None and limits is not None:
            from myq.control_alignment import FollowerOrientationLimit

            self._orientation_limit = FollowerOrientationLimit(poses["left"], limits)
        if self.action_representation == "joint_absolute":
            import flexivrdk
            from myq.joint_control import JointTargetController

            control = self.config["control"]
            # Same TCP admittance/alignment, asynchronous IK and impedance as
            # myq dual_robot_drag_collect.py's joint_absolute path.
            for name, robot in self.robots.items():
                if name in self._joint_drag_controllers:
                    self._joint_drag_controllers[name].close()
                robot.SwitchMode(flexivrdk.Mode.NRT_JOINT_IMPEDANCE)
                scale = (
                    float(control["leader_joint_impedance_scale"])
                    if name == "right"
                    else 1.0
                )
                robot.SetJointImpedance(
                    (np.asarray(robot.info().K_q_nom) * scale).tolist()
                )
                self._joint_drag_controllers[name] = JointTargetController(
                    flexivrdk.Model(robot),
                    robot.states().q,
                    robot.info(),
                    control,
                    1.0 / float(control["control_hz"]),
                )
            self.joint_target = np.asarray(
                self.robots["left"].states().q, dtype=float
            ).copy()

    def send(
        self, target: np.ndarray, gripper_width: float, *, manual: bool
    ) -> dict[str, np.ndarray]:
        """Send one final target to each arm and return the accepted target."""
        targets = (
            self.manual_targets()
            if manual
            else {"right": self.target["right"], "left": target}
        )
        self._send(targets)
        if abs(gripper_width - self.gripper_target) > 0.001 + 1e-8:
            assert self._gripper_worker is not None
            self._submit_gripper_target(gripper_width, manual=manual)
            self.gripper_target = float(gripper_width)
        return targets

    def _submit_gripper_target(self, width: float, *, manual: bool) -> None:
        closed = float(self.config["robots"]["left"]["gripper_close_width_m"])
        if self.binary_gripper and not manual and width == closed:
            self._gripper_worker.submit(width, force_n=50.0)
        else:
            self._gripper_worker.submit(width)

    def measured_gripper_width(self) -> float:
        """Return the follower gripper's current opening width in metres."""
        width = float(self.gripper.states().width)
        if not np.isfinite(width):
            raise ValueError("Flexiv gripper reported a non-finite width.")
        return width

    def stop(self) -> None:
        """Attempt to stop every connected arm before releasing resources."""
        if self._stopped:
            return
        if self._pause_thread is not None:
            self._pause_thread.join()
            self._pause_thread = None
        failures = []
        for name, robot in self.robots.items():
            try:
                robot.Stop()
            except Exception as exc:
                logging.exception("Failed to stop Flexiv %s arm.", name)
                failures.append(exc)
        if failures:
            raise failures[0]
        self._stopped = True

    def close(self) -> None:
        """Release all resources after stopping; safe after partial startup."""
        if self._closed:
            return
        self._closed = True
        try:
            self.stop()
        finally:
            try:
                for controller in self._joint_drag_controllers.values():
                    try:
                        controller.close()
                    except Exception:
                        logging.exception("Failed to close manual joint IK controller.")
                if self._gripper_worker is not None:
                    self._gripper_worker.close()
            finally:
                try:
                    if self.console is not None:
                        self.console.stop()
                finally:
                    try:
                        if self.cameras is not None:
                            self.cameras.stop()
                    finally:
                        if self.image_recorder is not None:
                            self.image_recorder.close()


def run_episode(
    robot: FlexivRobotAdapter,
    policy: RemotePolicy,
    *,
    max_ticks: int | None = None,
    task_timeout_seconds: float | None = None,
    handoff_grace_ticks: int | None = None,
    monotonic=time.perf_counter,
    sleep=time.sleep,
) -> int:
    """Run the Flexiv adapter through the SDK-independent fixed-rate runtime."""
    runtime = RemoteRobotRuntime(
        robot,
        policy,
        prompt=str(robot.config["recording"]["task"]),
        control_hz=float(robot.config["control"]["control_hz"]),
        inference_horizon=getattr(policy, "inference_horizon", 6),
        request_steps=getattr(policy, "base_request_steps", None),
        coarse_origin_schedule=getattr(policy, "coarse_origin_schedule", False),
        fine_horizon=getattr(policy, "fine_horizon", None),
        fine_available=getattr(policy, "fine_available", False),
        initial_phase=getattr(policy, "initial_phase", "fine"),
        max_ticks=max_ticks,
        task_timeout_seconds=task_timeout_seconds,
        handoff_grace_ticks=handoff_grace_ticks,
        monotonic=monotonic,
        sleep=sleep,
    )
    return runtime.run()
