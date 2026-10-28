# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for split Torch/APX inference and Flexiv robot control."""

from __future__ import annotations

import importlib.util
import logging
import socket
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rlinf.algorithms.online_residual_rl.remote_client import (
    RemotePolicy as PackageRemotePolicy,
)
from rlinf.algorithms.online_residual_rl.remote_server import RemoteInferenceServer
from rlinf.algorithms.online_residual_rl.remote_wire import (
    PROTOCOL_VERSION,
    receive_message,
    send_message,
)
from rlinf.algorithms.online_residual_rl.robot_runtime import (
    RemoteRobotRuntime,
    RobotControlResult,
    RobotObservation,
    RobotTickRequest,
)
from rlinf.models.embodiment.openpi_rlinf.env_io import EnvIO
from rlinf.models.embodiment.openpi_rlinf.inference import rebase_rtc_world_deltas

_ROOT = Path(__file__).resolve().parents[2]
_CLIENT = _ROOT / "examples" / "embodiment" / "flexiv_remote_robot.py"
_SPEC = importlib.util.spec_from_file_location("flexiv_remote_robot", _CLIENT)
assert _SPEC is not None and _SPEC.loader is not None
client = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(client)


@pytest.fixture
def server_entry():
    server_path = _ROOT / "examples" / "embodiment" / "run_flexiv_online_residual.py"
    spec = importlib.util.spec_from_file_location(
        "run_flexiv_online_residual", server_path
    )
    assert spec is not None and spec.loader is not None
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    return server


def test_torch_server_preserves_jacobian_rtc_default(monkeypatch, server_entry):
    server = server_entry
    server_path = server.__file__

    monkeypatch.setattr(sys, "argv", [str(server_path), "--checkpoint", "weights.pt"])
    assert server.parse_args().rtc_guidance_mode == "jacobian"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(server_path),
            "--checkpoint",
            "weights.pt",
            "--rtc-guidance-mode",
            "jacobian",
        ],
    )
    assert server.parse_args().rtc_guidance_mode == "jacobian"


@pytest.mark.parametrize(
    "override,expected", [(None, True), ("--no-disable-debug", False)]
)
def test_server_debug_switch_from_config(
    monkeypatch, server_entry, tmp_path, override, expected
):
    config = tmp_path / "inference.yaml"
    config.write_text("checkpoint: weights.pt\ndisable_debug: true\n")
    argv = [server_entry.__file__, "--inference-config", str(config)]
    if override is not None:
        argv.append(override)
    monkeypatch.setattr(sys, "argv", argv)
    assert server_entry.parse_args().disable_debug is expected


def test_server_accepts_robot_side_auto_fine_config(
    monkeypatch, server_entry, tmp_path
):
    config = tmp_path / "inference.yaml"
    config.write_text(
        "checkpoint: weights.pt\nauto_fine:\n"
        "  xy_center_mm: [480, 53]\n  xy_radius_mm: 15\n  z_below_mm: 138\n"
    )
    monkeypatch.setattr(
        sys, "argv", [server_entry.__file__, "--inference-config", str(config)]
    )
    assert server_entry.parse_args().checkpoint == Path("weights.pt")


def test_joint_stage2_deployment_matches_stage1(monkeypatch, server_entry):
    from omegaconf import OmegaConf

    directory = _ROOT / "examples/embodiment/config"
    base = OmegaConf.load(directory / "flexiv_joint_absolute_inference.yaml")
    path = directory / "flexiv_joint_absolute_stage2.yaml"
    stage2 = OmegaConf.load(path)
    for key in (
        "checkpoint",
        "norm_stats",
        "training_config",
        "backend",
        "disable_debug",
        "num_flow_steps",
        "home_dataset_path",
        "base_chunk_steps",
        "base_request_steps",
        "rtc_guidance_mode",
        "joint_control",
        "gripper_policy",
    ):
        assert stage2[key] == base[key], key
    monkeypatch.setattr(
        sys, "argv", [server_entry.__file__, "--inference-config", str(path)]
    )
    args = server_entry.parse_args()
    assert args.serve and args.online_train and not args.stage1_only
    assert args.binary_gripper
    assert args.base_only_episodes == 50
    assert args.episode_dir != args.checkpoint_dir
    monkeypatch.setattr(
        sys,
        "argv",
        [
            server_entry.__file__,
            "--inference-config",
            str(path),
            "--no-online-train",
            "--actor-checkpoint",
            "actor.pt",
        ],
    )
    args = server_entry.parse_args()
    assert not args.online_train and args.actor_checkpoint == Path("actor.pt")


@pytest.mark.parametrize(
    "override,disabled", [(None, True), ("--no-disable-debug", False)]
)
def test_robot_debug_switch_disables_recorders(
    monkeypatch, tmp_path, override, disabled, caplog
):
    caplog.set_level(logging.INFO)
    path = _ROOT / "examples/embodiment/run_flexiv_online_residual_robot.py"
    monkeypatch.setitem(sys.modules, "flexiv_remote_robot", client)
    spec = importlib.util.spec_from_file_location("flexiv_robot_entry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = tmp_path / "inference.yaml"
    config.write_text(
        "disable_debug: true\nauto_fine:\n"
        "  xy_center_mm: [480, 53]\n  xy_radius_mm: 15\n  z_below_mm: 138\n"
    )
    argv = [
        str(path),
        "--inference-config",
        str(config),
        "--diagnostic-dir",
        str(tmp_path / "trace"),
        "--capture-dir",
        str(tmp_path / "images"),
    ]
    if override is not None:
        argv.append(override)
    monkeypatch.setattr(sys, "argv", argv)
    robot_options = {}
    policy_options = {}

    def build_robot(*args, **kwargs):
        robot_options.update(kwargs)
        return SimpleNamespace(binary_gripper=False, gripper_threshold=0.5)

    def connect_policy(*args, **kwargs):
        policy_options.update(kwargs)
        return SimpleNamespace(
            fine_available=False,
            initial_phase="coarse",
            base_request_steps=16,
            base_horizon=20,
            coarse_origin_schedule=True,
            fine_horizon=6,
        )

    monkeypatch.setattr(
        module,
        "load_myq_config",
        lambda _: {
            "recording": {"task": "insert connector"},
            "cameras": {
                name: {"height": 224, "width": 224} for name in ("main", "wrist")
            },
        },
    )
    monkeypatch.setattr(module, "FlexivRobotAdapter", build_robot)
    monkeypatch.setattr(module, "RemotePolicy", connect_policy)
    monkeypatch.setattr(module, "run_episode", lambda *args, **kwargs: 0)
    monkeypatch.setattr(logging.getLogger(), "level", logging.INFO)
    monkeypatch.setattr(logging.getLogger("flexiv.startup"), "level", logging.NOTSET)
    monkeypatch.setattr(logging.getLogger("flexiv.status"), "level", logging.NOTSET)
    assert module.main() == 0
    assert robot_options["auto_fine"] == {
        "xy_center_mm": [480, 53],
        "xy_radius_mm": 15,
        "z_below_mm": 138,
    }
    assert robot_options["capture_dir"] == (None if disabled else tmp_path / "images")
    assert robot_options["record_diagnostics"] is (not disabled)
    assert policy_options["diagnostic_dir"] == (
        None if disabled else tmp_path / "trace"
    )
    assert "Policy prompt: insert connector" in caplog.text
    assert "fine_available=False" in caplog.text


@pytest.mark.parametrize("guidance_mode", (None, "jacobian", "approx"))
def test_torch_server_disables_rtc_without_jacobian_setup(
    monkeypatch, server_entry, guidance_mode
):
    from rlinf.models.embodiment.openpi_rlinf import inference

    argv = [
        server_entry.__file__,
        "--checkpoint",
        "weights.pt",
        "--serve",
        "--stage1-only",
        "--disable-rtc",
    ]
    if guidance_mode is not None:
        argv.extend(["--rtc-guidance-mode", guidance_mode])
    monkeypatch.setattr(sys, "argv", argv)
    args = server_entry.parse_args()
    model = SimpleNamespace(action_horizon=32)
    model.eval = lambda: model
    loader_options = []

    def load_model(*args, **kwargs):
        loader_options.append(kwargs)
        return model

    monkeypatch.setattr(inference, "load_stage1_model", load_model)
    policy = server_entry._build_policy(
        args,
        None,
        SimpleNamespace(action_representation=SimpleNamespace(value="eef_delta")),
    )
    assert not policy.conditions_on_committed
    assert loader_options[0]["rtc_guidance_mode"] == "approx"

    def stop_at_config(path):
        raise RuntimeError("configuration loading reached")

    monkeypatch.setattr(server_entry.OmegaConf, "load", stop_at_config)
    with pytest.raises(RuntimeError, match="configuration loading reached"):
        server_entry.main()


def test_remote_wire_roundtrip_and_version_rejection():
    left, right = socket.socketpair()
    with left, right:
        image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        send_message(left, {"version": PROTOCOL_VERSION, "image": image})
        np.testing.assert_array_equal(receive_message(right)["image"], image)
        send_message(left, {"version": 9000})
        with pytest.raises(ValueError, match="version"):
            receive_message(right)


def test_remote_client_is_importable_without_flexiv_adapter():
    assert PackageRemotePolicy.__module__.endswith("remote_client")
    assert client.RemotePolicy.__module__ == "remote_client"


def test_runtime_accepts_an_unrelated_robot_adapter():
    class BenchAdapter:
        def __init__(self):
            self.applied = []
            self.closed = False

        def start(self):
            pass

        def read_observation(self, *, tick, prompt):
            return RobotObservation(
                policy_inputs={"joint_vector": np.asarray([tick], np.float32)},
                action_context={"reference": tick},
            )

        def poll_control(self, *, tick):
            return RobotTickRequest()

        def control_tick(self, *, tick, action, action_context):
            if action is None:
                action = self.applied[-1]
            self.applied.append(action.copy())
            return RobotControlResult(
                receipt={"executed": action.copy(), "context": action_context}
            )

        def stop(self):
            pass

        def close(self):
            self.closed = True

    class BenchPolicy:
        session = "bench"

        def __init__(self):
            self.requests = []
            self.receipts = []
            self.finished = None

        def call(self, kind, **payload):
            future = Future()
            if kind == "infer":
                self.requests.append(payload)
                future.set_result(
                    {
                        "frame": payload["frame"],
                        "actions": np.full((2, 3), len(self.requests), np.float32),
                        "action_context": payload["action_context"],
                        "model_version": 2,
                    }
                )
            else:
                self.receipts.append(payload)
                future.set_result({})
            return future

        def close(self, frame, success):
            self.finished = (frame, success)

    robot, policy = BenchAdapter(), BenchPolicy()
    runtime = RemoteRobotRuntime(
        robot,
        policy,
        prompt="move the marker",
        control_hz=10.0,
        inference_horizon=2,
        max_ticks=5,
        monotonic=lambda: 0.0,
        sleep=lambda _seconds: None,
    )
    assert runtime.run() == 0
    assert [request["frame"] for request in policy.requests] == [0, 0, 2, 4]
    assert policy.requests[2]["accepted"]["inference_delay_steps"] == 0
    assert [action[0] for action in robot.applied] == [1, 1, 2, 2, 3]
    assert len(policy.receipts) == 5
    assert robot.closed
    assert policy.finished == (4, None)


def test_runtime_rtc_warmup_keeps_the_executed_bootstrap_block():
    class Robot:
        def __init__(self):
            self.actions = []

        def start(self):
            pass

        def read_observation(self, *, tick, prompt):
            return RobotObservation(policy_inputs={"states": np.zeros(8)})

        def poll_control(self, *, tick):
            return RobotTickRequest()

        def control_tick(self, *, tick, action, action_context):
            self.actions.append(float(action[0]))
            return RobotControlResult(receipt={})

        def stop(self):
            pass

        def close(self):
            pass

    class Policy:
        rtc_enabled = True

        def __init__(self):
            self.requests = []

        def call(self, kind, **payload):
            future = Future()
            if kind == "infer":
                self.requests.append(payload)
                future.set_result(
                    {
                        "frame": payload["frame"],
                        "actions": np.full((2, 7), len(self.requests), np.float32),
                        "model_version": 0,
                    }
                )
            else:
                future.set_result({})
            return future

        def close(self, frame, success):
            pass

    robot, policy = Robot(), Policy()
    runtime = RemoteRobotRuntime(
        robot,
        policy,
        prompt="pick",
        control_hz=30,
        inference_horizon=2,
        max_ticks=1,
        monotonic=lambda: 0.0,
        sleep=lambda _seconds: None,
    )
    assert runtime.run() == 0
    assert [request["warmup"] for request in policy.requests] == [False, True, False]
    assert robot.actions == [1.0]


@pytest.mark.parametrize("base_horizon", (10, 32))
def test_runtime_switches_from_configured_base_chunk_to_six_frame_fine_phase(
    base_horizon,
):
    class BenchAdapter:
        def __init__(self):
            self.executed = []
            self.phases = []

        def start(self):
            pass

        def read_observation(self, *, tick, prompt):
            return RobotObservation(policy_inputs={"state": np.asarray([tick])})

        def poll_control(self, *, tick):
            return RobotTickRequest(enter_fine_phase=tick == 7)

        def control_tick(self, *, tick, action, action_context):
            self.executed.append((tick, float(action[0])))
            self.phases.append(action_context["control_phase"])
            return RobotControlResult(receipt={"action_receipt_physical": action})

        def stop(self):
            pass

        def close(self):
            pass

    class BenchPolicy:
        session = "bench"

        def __init__(self):
            self.requests = []

        def call(self, kind, **payload):
            future = Future()
            if kind == "infer":
                self.requests.append(payload)
                width = base_horizon if payload["phase"] == "coarse" else 6
                start = 100 * len(self.requests)
                future.set_result(
                    {
                        "frame": payload["frame"],
                        "actions": np.arange(start, start + width, dtype=np.float32)[
                            :, None
                        ],
                        "model_version": 1,
                    }
                )
            else:
                future.set_result({})
            return future

        def close(self, frame, success):
            pass

    robot, policy = BenchAdapter(), BenchPolicy()
    runtime = RemoteRobotRuntime(
        robot,
        policy,
        prompt="insert",
        control_hz=30,
        inference_horizon=base_horizon,
        fine_horizon=6,
        fine_available=True,
        initial_phase="coarse",
        max_ticks=base_horizon + 8,
        monotonic=lambda: 0.0,
        sleep=lambda _seconds: None,
    )
    assert runtime.run() == 0
    assert [(r["frame"], r["phase"]) for r in policy.requests] == [
        (0, "coarse"),
        (0, "coarse"),
        (base_horizon, "fine"),
        (base_horizon + 6, "fine"),
    ]
    assert [value for _, value in robot.executed] == list(
        range(100, 100 + base_horizon)
    ) + list(range(200, 206)) + list(range(300, 302))
    assert policy.requests[2]["accepted"]["accept_frame"] == base_horizon
    assert policy.requests[3]["accepted"]["accept_frame"] == base_horizon + 6
    assert robot.phases == ["coarse"] * base_horizon + ["fine"] * 8


@pytest.mark.parametrize(("horizon", "late_until"), ((10, None), (32, None), (10, 18)))
def test_runtime_submits_coarse_inference_before_fixed_chunk_handoff(
    horizon, late_until, caplog
):
    class BenchAdapter:
        def __init__(self):
            self.executed = []

        def start(self):
            pass

        def read_observation(self, *, tick, prompt):
            return RobotObservation(policy_inputs={"state": np.asarray([tick])})

        def poll_control(self, *, tick):
            return RobotTickRequest()

        def control_tick(self, *, tick, action, action_context):
            value = self.executed[-1][1] if action is None else float(action[0])
            self.executed.append((tick, value))
            return RobotControlResult(receipt={"action_receipt_physical": action})

        def stop(self):
            pass

        def close(self):
            pass

    class DelayedFuture(Future):
        def done(self):
            return len(robot.executed) >= late_until

    class BenchPolicy:
        session = "bench"

        def __init__(self):
            self.requests = []

        def call(self, kind, **payload):
            if kind == "infer":
                self.requests.append(payload)
                future = (
                    DelayedFuture()
                    if late_until is not None and len(self.requests) == 2
                    else Future()
                )
                if payload.get("bootstrap"):
                    actions = np.arange(horizon, dtype=np.float32)[:, None]
                else:
                    actions = np.arange(100, 100 + horizon, dtype=np.float32)[:, None]
                future.set_result(
                    {
                        "frame": payload["frame"],
                        "actions": actions,
                        "model_version": 1,
                    }
                )
            else:
                future = Future()
                future.set_result({})
            return future

        def close(self, frame, success):
            pass

    robot, policy = BenchAdapter(), BenchPolicy()
    runtime = RemoteRobotRuntime(
        robot,
        policy,
        prompt="insert",
        control_hz=30,
        inference_horizon=horizon,
        request_steps=7,
        initial_phase="coarse",
        max_ticks=horizon + 3 if late_until is None else 21,
        monotonic=lambda: 0.0,
        sleep=lambda _seconds: None,
    )
    assert runtime.run() == 0
    if late_until is not None:
        assert "request_tick=7 budget_ms=100.00" in caplog.text
    requests = [(r["frame"], r["target_frame"]) for r in policy.requests]
    if late_until is None:
        assert requests == [(0, 0), (7, horizon)]
        assert [value for tick, value in robot.executed if tick >= horizon] == [
            100,
            101,
            102,
        ]
    else:
        assert requests == [(0, 0), (7, 10), (18, 20)]
        assert [value for tick, value in robot.executed if tick in (18, 19, 20)] == [
            108,
            109,
            100,
        ]


@pytest.mark.parametrize("fine_at", (None, 30))
def test_runtime_uses_each_coarse_prediction_origin_for_request_and_handoff(fine_at):
    class Robot:
        def __init__(self):
            self.executed = {}

        def start(self):
            pass

        def read_observation(self, *, tick, prompt):
            return RobotObservation(policy_inputs={"states": np.asarray([tick])})

        def poll_control(self, *, tick):
            return RobotTickRequest(enter_fine_phase=tick == fine_at)

        def control_tick(self, *, tick, action, action_context):
            self.executed[tick] = float(action[0])
            return RobotControlResult(receipt={})

        def stop(self):
            pass

        def close(self):
            pass

    class Policy:
        session = "prediction-origin"

        def __init__(self):
            self.requests = []

        def call(self, kind, **payload):
            future = Future()
            if kind == "infer":
                self.requests.append(payload)
                if payload["phase"] == "fine":
                    actions = np.arange(600, 606, dtype=np.float32)
                elif payload["bootstrap"]:
                    actions = np.arange(22, dtype=np.float32)
                else:
                    coarse_index = (
                        sum(request["phase"] == "coarse" for request in self.requests)
                        - 1
                    )
                    actions = np.arange(
                        100 * coarse_index + 6,
                        100 * coarse_index + 22,
                        dtype=np.float32,
                    )
                future.set_result(
                    {
                        "frame": payload["frame"],
                        "phase": payload["phase"],
                        "actions": actions[:, None],
                        "model_version": 1,
                    }
                )
            else:
                future.set_result({})
            return future

        def close(self, frame, success):
            pass

    robot, policy = Robot(), Policy()
    runtime = RemoteRobotRuntime(
        robot,
        policy,
        prompt="pick",
        control_hz=30,
        inference_horizon=22,
        request_steps=16,
        coarse_origin_schedule=True,
        fine_horizon=6,
        fine_available=True,
        initial_phase="coarse",
        max_ticks=56 if fine_at is None else 46,
        monotonic=lambda: 0.0,
        sleep=lambda _seconds: None,
    )
    assert runtime.run() == 0
    assert [(r["frame"], r["target_frame"]) for r in policy.requests[:3]] == [
        (0, 0),
        (16, 22),
        (32, 38),
    ]
    assert [robot.executed[tick] for tick in (0, 16, 21, 22, 32, 37, 38)] == [
        0,
        16,
        21,
        106,
        116,
        121,
        206,
    ]
    if fine_at is None:
        assert robot.executed[43] == 211
        assert (policy.requests[3]["frame"], policy.requests[3]["target_frame"]) == (
            48,
            54,
        )
        assert robot.executed[54] == 306
    else:
        assert [robot.executed[tick] for tick in range(38, 44)] == list(range(206, 212))
        assert [
            (r["frame"], r["target_frame"], r["phase"]) for r in policy.requests[3:]
        ] == [
            (38, 44, "fine"),
            (44, 50, "fine"),
        ]
        assert robot.executed[44] == 600


@pytest.mark.parametrize(
    ("horizon", "request_steps"), ((16, 12), (22, 11), (22, 14), (22, 18), (24, 18))
)
@pytest.mark.parametrize("disable_debug", (False, True))
def test_runtime_adapts_coarse_schedule_to_configured_indices(
    horizon, request_steps, caplog, monkeypatch, disable_debug
):
    caplog.set_level(logging.INFO)
    if disable_debug:
        caplog.set_level(logging.WARNING)
        caplog.set_level(logging.INFO, logger="flexiv.status")

    class Robot:
        def __init__(self):
            self.executed = {}

        def start(self):
            pass

        def read_observation(self, *, tick, prompt):
            return RobotObservation(
                policy_inputs={"states": np.full(8, tick, dtype=np.float32)}
            )

        def poll_control(self, *, tick):
            return RobotTickRequest()

        def control_tick(self, *, tick, action, action_context):
            self.executed[tick] = int(action[0])
            return RobotControlResult(
                receipt={"joint_target_rad": np.repeat(action[0], 7)}
            )

        def stop(self):
            pass

        def close(self):
            pass

    class Policy:
        session = "configurable-coarse"

        def __init__(self):
            self.requests = []

        def call(self, kind, **payload):
            future = Future()
            if kind == "infer":
                self.requests.append(payload)
                chunk_index = len(self.requests) - 1
                start = 0 if payload["bootstrap"] else horizon - request_steps
                actions = np.arange(start, horizon, dtype=np.float32)
                future.set_result(
                    {
                        "frame": payload["frame"],
                        "phase": payload["phase"],
                        "actions": (100 * chunk_index + actions)[:, None],
                        "model_version": 1,
                    }
                )
            else:
                future.set_result({})
            return future

        def close(self, frame, success):
            pass

    robot, policy = Robot(), Policy()
    runtime = RemoteRobotRuntime(
        robot,
        policy,
        prompt="pick",
        control_hz=30,
        inference_horizon=horizon,
        request_steps=request_steps,
        coarse_origin_schedule=True,
        initial_phase="coarse",
        max_ticks=horizon + 2 * request_steps + 1,
        monotonic=lambda: 0.0,
        sleep=lambda _seconds: None,
    )
    assert runtime.run() == 0
    expected_requests = [
        (0, 0),
        (request_steps, horizon),
        (2 * request_steps, horizon + request_steps),
        (3 * request_steps, horizon + 2 * request_steps),
    ]
    if 2 * request_steps == horizon:
        expected_requests.append((4 * request_steps, horizon + 3 * request_steps))
    assert [(r["frame"], r["target_frame"]) for r in policy.requests] == (
        expected_requests
    )
    assert robot.executed[horizon - 1] == horizon - 1
    assert robot.executed[horizon] == 100 + horizon - request_steps
    assert robot.executed[horizon + request_steps] == 200 + horizon - request_steps
    assert robot.executed[horizon + 2 * request_steps] == (
        300 + horizon - request_steps
    )
    handoffs = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("joint handoff ")
    ]
    if disable_debug:
        assert handoffs == []
        assert "joint request tracking" not in caplog.text
        metrics = [
            r.getMessage()
            for r in caplog.records
            if r.getMessage().startswith("robot handoff ")
        ]
        assert len(metrics) == 3
        assert all(
            "response_ms=" in message
            and "handoff_ms=" in message
            and "model_hz=" in message
            for message in metrics
        )
        return
    assert len(handoffs) == 3
    assert (
        f"joint request tracking tick={request_steps} "
        f"max_command_tracking_error_deg={np.rad2deg(1):.4f}"
    ) in caplog.text
    expected_step = np.rad2deg(101 - request_steps)
    expected_change = np.rad2deg(100 - request_steps)
    assert handoffs[0] == (
        f"joint handoff tick={horizon} late_ticks=0 "
        f"max_target_step_deg={expected_step:.4f} "
        f"max_target_step_change_deg={expected_change:.4f}"
    )


@pytest.mark.parametrize("late_until", (22, 24, 27, 38))
def test_runtime_origin_schedule_skips_actions_missed_by_a_late_handoff(
    late_until, caplog
):
    caplog.set_level(logging.INFO)

    class Robot:
        def __init__(self):
            self.actions = []

        def start(self):
            pass

        def read_observation(self, *, tick, prompt):
            return RobotObservation(policy_inputs={"states": np.asarray([tick])})

        def poll_control(self, *, tick):
            return RobotTickRequest()

        def control_tick(self, *, tick, action, action_context):
            self.actions.append(None if action is None else float(action[0]))
            return RobotControlResult(receipt={})

        def stop(self):
            pass

        def close(self):
            pass

    robot = Robot()

    class DelayedFuture(Future):
        def done(self):
            return len(robot.actions) >= late_until

    class Policy:
        def __init__(self):
            self.requests = []
            self.discarded = []

        def call(self, kind, **payload):
            if kind == "discard":
                self.discarded.append(payload["frame"])
            if kind != "infer":
                future = Future()
                future.set_result({})
                return future
            self.requests.append(payload)
            future = DelayedFuture() if len(self.requests) == 2 else Future()
            if payload["bootstrap"]:
                actions = np.arange(22, dtype=np.float32)
            else:
                actions = np.arange(
                    100 * (len(self.requests) - 1) + 6,
                    100 * (len(self.requests) - 1) + 22,
                    dtype=np.float32,
                )
            future.set_result(
                {
                    "frame": payload["frame"],
                    "actions": actions[:, None],
                    "model_version": 1,
                }
            )
            return future

        def close(self, frame, success):
            pass

    policy = Policy()
    runtime = RemoteRobotRuntime(
        robot,
        policy,
        prompt="pick",
        control_hz=30,
        inference_horizon=22,
        request_steps=16,
        coarse_origin_schedule=True,
        initial_phase="coarse",
        max_ticks=46,
        monotonic=lambda: 0.0,
        sleep=lambda _seconds: None,
    )
    assert runtime.run() == 0
    if late_until < 38:
        assert robot.actions[22:late_until] == [None] * (late_until - 22)
        assert robot.actions[late_until] == 100 + late_until - 16
        assert (
            f"coarse chunk entry tick={late_until} request_frame=16 "
            f"planned_boundary=22 first_prediction_index={late_until - 16} "
            f"held_ticks={late_until - 22}"
        ) in caplog.text
        assert robot.actions[37:39] == [121.0, 206.0]
        assert [(r["frame"], r["target_frame"]) for r in policy.requests] == [
            (0, 0),
            (16, 22),
            (32, 38),
        ]
    else:
        assert policy.discarded == [16]
        assert robot.actions[22:44] == [None] * 22
        assert robot.actions[44] == 206.0
        assert [(r["frame"], r["target_frame"]) for r in policy.requests] == [
            (0, 0),
            (16, 22),
            (38, 44),
        ]


def test_runtime_holds_and_retries_when_entire_inference_block_is_stale(caplog):
    class BenchAdapter:
        def __init__(self):
            self.executed = []

        def start(self):
            pass

        def read_observation(self, *, tick, prompt):
            return RobotObservation(policy_inputs={"state": np.asarray([tick])})

        def poll_control(self, *, tick):
            return RobotTickRequest()

        def control_tick(self, *, tick, action, action_context):
            value = self.executed[-1][1] if action is None else float(action[0])
            self.executed.append((tick, value))
            return RobotControlResult(receipt={})

        def stop(self):
            pass

        def close(self):
            pass

    class DelayedFuture(Future):
        def done(self):
            return len(robot.executed) >= 21

    class BenchPolicy:
        session = "bench"

        def __init__(self):
            self.requests = []
            self.discarded = []

        def call(self, kind, **payload):
            future = Future()
            if kind == "infer":
                self.requests.append(payload)
                if payload.get("bootstrap"):
                    values = np.arange(10, dtype=np.float32)
                elif len(self.requests) == 2:
                    values = np.arange(100, 110, dtype=np.float32)
                    future = DelayedFuture()
                else:
                    values = np.arange(200, 210, dtype=np.float32)
                future.set_result(
                    {
                        "frame": payload["frame"],
                        "actions": values[:, None],
                        "model_version": 1,
                    }
                )
            elif kind == "discard":
                self.discarded.append(payload["frame"])
                future.set_result({})
            else:
                future.set_result({})
            return future

        def close(self, frame, success):
            pass

    robot, policy = BenchAdapter(), BenchPolicy()
    runtime = RemoteRobotRuntime(
        robot,
        policy,
        prompt="insert",
        control_hz=30,
        inference_horizon=10,
        request_steps=7,
        initial_phase="coarse",
        max_ticks=34,
        monotonic=lambda: 0.0,
        sleep=lambda _seconds: None,
    )

    assert runtime.run() == 0
    assert policy.discarded == [7]
    assert [request["frame"] for request in policy.requests] == [0, 7, 21]
    assert [value for tick, value in robot.executed if 10 <= tick < 31] == [9] * 21
    assert [value for tick, value in robot.executed if tick >= 31] == [200, 201, 202]
    assert "continuing to hold target" in caplog.text
    assert "discarding stale inference frame=7" in caplog.text


class _Policy:
    actor_version = 4

    def __init__(self):
        self.calls = []
        self.closed = False

    def predict_action_batch(self, observation, *, mode, committed):
        self.calls.append((observation, mode, committed))
        return torch.arange(84, dtype=torch.float32).reshape(1, 12, 7), {}

    def close(self):
        self.closed = True

    def encode_receipts(self, actions):
        return actions


class _TorchPolicy(_Policy):
    conditions_on_committed = True

    def __init__(self):
        super().__init__()
        self.rtc_delays = []

    def predict_action_batch(
        self, observation, *, mode, committed, rtc_delay_steps=None
    ):
        self.calls.append((observation, mode, committed))
        self.rtc_delays.append(rtc_delay_steps)
        raw = torch.arange(32 * 32, dtype=torch.float32).reshape(1, 32, 32)
        raw += 1000 * (len(self.calls) - 1)
        physical = torch.full((1, 12, 7), 0.025 * len(self.calls))
        return physical, {"raw_model_actions": raw}


def test_torch_server_guidance_uses_only_accepted_raw_actions_and_resets(caplog):
    caplog.set_level(logging.INFO)
    policy = _TorchPolicy()
    server = RemoteInferenceServer(
        policy,
        SimpleNamespace(inference_horizon=6),
        backend="torch",
        rtc_delay_buffer_size=2,
    )
    request = {
        "type": "infer",
        "session": "one",
        "frame": 0,
        "observation": {
            "states": np.zeros(8, np.float32),
            "main_images": np.zeros((2, 2, 3), np.uint8),
            "wrist_images": np.zeros((2, 2, 3), np.uint8),
        },
        "action_context": {
            "training_reference": np.array([0, 0, 0, 1, 0, 0, 0], float)
        },
        "prompt": "task",
    }
    with pytest.raises(ValueError, match="robot='apx', server='torch'"):
        server.handle({"type": "start", "session": "one", "backend": "apx"})
    server.handle({"type": "start", "session": "one", "backend": "torch"})
    bootstrap = server.handle({**request, "bootstrap": True})
    first = server.handle(request)
    assert bootstrap["actions"].shape == first["actions"].shape == (6, 7)
    assert policy.calls[0][2] is None
    torch.testing.assert_close(
        policy.calls[1][2],
        torch.arange(32 * 32, dtype=torch.float32).reshape(1, 32, 32),
    )
    # A command receipt carries physical units and never changes RTC guidance.
    server.handle(
        {
            "type": "receipt",
            "session": "one",
            "frame": 0,
            "info": {"action_receipt_physical": np.full(7, 99, np.float32)},
        }
    )
    server.handle(
        {
            **request,
            "frame": 6,
            "accepted": {
                "frame": 0,
                "accept_frame": 6,
                "accept_tick": 6,
                "inference_delay_steps": 4,
                "model_version": 4,
            },
        }
    )
    expected = torch.arange(32 * 32, dtype=torch.float32).reshape(1, 32, 32)
    torch.testing.assert_close(policy.calls[2][2], (expected + 1000)[:, 6:])
    server.handle(
        {
            **request,
            "frame": 12,
            "accepted": {
                "frame": 6,
                "accept_frame": 12,
                "accept_tick": 12,
                "inference_delay_steps": 1,
                "model_version": 4,
            },
        }
    )
    server.handle(
        {
            **request,
            "frame": 18,
            "accepted": {
                "frame": 12,
                "accept_frame": 18,
                "accept_tick": 18,
                "inference_delay_steps": 2,
                "model_version": 4,
            },
        }
    )
    assert policy.rtc_delays == [0, 6, 6, 6, 6]
    server.handle({"type": "finish", "session": "one", "frame": 5, "success": None})
    server.handle({"type": "start", "session": "two", "backend": "torch"})
    server.handle({**request, "session": "two", "bootstrap": True})
    assert policy.calls[5][2] is None
    assert policy.rtc_delays[5] == 0
    assert "frame=6 conditioned=True overlap_steps=26 handoff_offset=6" in caplog.text
    assert "session=two frame=0 conditioned=False overlap_steps=0" in caplog.text
    server.close()


def test_torch_server_reports_lost_rtc_context_after_late_acceptance(caplog):
    caplog.set_level(logging.INFO)
    policy = _TorchPolicy()
    server = RemoteInferenceServer(
        policy, SimpleNamespace(inference_horizon=6), backend="torch"
    )
    server.handle({"type": "start", "session": "late", "backend": "torch"})
    request = {
        "type": "infer",
        "session": "late",
        "frame": 0,
        "observation": {"states": np.zeros(8, np.float32)},
        "prompt": "pick",
    }
    server.handle({**request, "bootstrap": True})
    server.handle(request)
    server.handle(
        {
            **request,
            "frame": 6,
            "accepted": {
                "frame": 0,
                "accept_frame": 6,
                "accept_tick": 7,
                "model_version": 4,
            },
        }
    )
    assert policy.calls[-1][2] is None
    assert (
        "RTC context cleared after late handoff session=late frame=0 late_ticks=1"
        in caplog.text
    )
    assert "frame=6 conditioned=False overlap_steps=0" in caplog.text
    server.close()


@pytest.mark.parametrize("request_steps", (14, 16))
def test_torch_server_warmup_preserves_executed_raw_context(request_steps):
    class StagePolicy(_TorchPolicy):
        supports_fine_phase = False

        def __init__(self):
            super().__init__()
            self.rebases = []

        def rebase_committed_actions(self, actions, previous_pose, current_pose):
            self.rebases.append((previous_pose.copy(), current_pose.copy()))
            return actions

        def predict_base_action_batch(
            self, observation, *, horizon, committed, rtc_delay_steps=None
        ):
            self.calls.append((observation, "coarse", committed))
            self.rtc_delays.append(rtc_delay_steps)
            raw = torch.arange(32 * 32, dtype=torch.float32).reshape(1, 32, 32)
            raw += 1000 * (len(self.calls) - 1)
            values = torch.arange(32, dtype=torch.float32).reshape(1, 32, 1)
            return values.expand(1, 32, 7), {"raw_model_actions": raw}

    policy = StagePolicy()
    server = RemoteInferenceServer(
        policy,
        SimpleNamespace(inference_horizon=6),
        backend="torch",
        base_horizon=22,
        base_request_steps=request_steps,
        start_base=True,
    )
    started = server.handle({"type": "start", "session": "warmup", "backend": "torch"})
    assert started["coarse_origin_schedule"]
    request = {
        "type": "infer",
        "session": "warmup",
        "frame": 0,
        "phase": "coarse",
        "observation": {"states": np.zeros(8, np.float32)},
        "action_context": {
            "training_reference": np.array([0, 0, 0, 1, 0, 0, 0], float)
        },
        "prompt": "pick",
    }

    def at_frame(frame):
        return {
            "action_context": {
                "training_reference": np.array([frame / 1000, 0, 0, 1, 0, 0, 0], float)
            }
        }

    bootstrap = server.handle({**request, "bootstrap": True})
    server.handle({**request, "bootstrap": True, "warmup": True})
    coarse = server.handle(
        {
            **request,
            **at_frame(request_steps),
            "frame": request_steps,
            "target_frame": 22,
        }
    )
    np.testing.assert_array_equal(bootstrap["actions"][:, 0], np.arange(22))
    np.testing.assert_array_equal(
        coarse["actions"][:, 0], np.arange(22 - request_steps, 22)
    )
    raw = torch.arange(32 * 32, dtype=torch.float32).reshape(1, 32, 32)
    torch.testing.assert_close(policy.calls[1][2], raw)
    torch.testing.assert_close(policy.calls[2][2], raw[:, request_steps:])
    next_coarse = server.handle(
        {
            **request,
            **at_frame(2 * request_steps),
            "frame": 2 * request_steps,
            "target_frame": 22 + request_steps,
            "accepted": {
                "frame": request_steps,
                "accept_frame": 22,
                "model_version": 4,
            },
        }
    )
    np.testing.assert_array_equal(
        next_coarse["actions"][:, 0], np.arange(22 - request_steps, 22)
    )
    torch.testing.assert_close(policy.calls[3][2], (raw + 2000)[:, request_steps:])
    assert policy.rtc_delays == [0, 0, 22 - request_steps, 22 - request_steps]
    assert [(before[0], after[0]) for before, after in policy.rebases] == [
        (0, 0),
        (0, request_steps / 1000),
        (request_steps / 1000, 2 * request_steps / 1000),
    ]
    server.close()


@pytest.mark.parametrize("use_quantiles", (False, True))
def test_rtc_world_deltas_rebase_to_current_observation_pose(use_quantiles):
    class Model(EnvIO):
        _action_norm_stats = SimpleNamespace(
            mean=np.zeros(32, dtype=np.float32),
            std=np.ones(32, dtype=np.float32),
            q01=np.full(32, -0.1, dtype=np.float32),
            q99=np.full(32, 0.1, dtype=np.float32),
        )
        _action_use_quantiles = use_quantiles

    model = Model()
    physical = torch.tensor([[[0.04, 0.01, 0, 0, 0, np.pi / 2, 0.04]]])
    raw = torch.full((1, 1, 32), 0.25)
    raw[..., :7] = model.normalize_rlt_actions(physical)
    old_pose = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float64)
    new_pose = np.array([0.02, 0, 0, np.sqrt(0.5), 0, 0, np.sqrt(0.5)])

    rebased = rebase_rtc_world_deltas(model, raw, old_pose, new_pose)
    decoded = model.denormalize_rlt_actions(rebased[..., :7])
    torch.testing.assert_close(
        decoded,
        torch.tensor([[[0.02, 0.01, 0, 0, 0, 0, 0.04]]]),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(rebased[..., 7:], raw[..., 7:])


@pytest.mark.parametrize("base_horizon", (10, 22))
def test_server_base_chunk_and_actor_transition_preserve_raw_rtc_guidance(
    base_horizon,
):
    class StagePolicy(_TorchPolicy):
        def predict_base_action_batch(
            self, observation, *, horizon, committed, rtc_delay_steps=None
        ):
            self.calls.append((observation, "coarse", committed))
            raw = torch.arange(32 * 32, dtype=torch.float32).reshape(1, 32, 32)
            raw += 1000 * (len(self.calls) - 1)
            return torch.zeros(1, 32, 7), {"raw_model_actions": raw}

    policy = StagePolicy()
    server = RemoteInferenceServer(
        policy,
        SimpleNamespace(inference_horizon=6),
        backend="torch",
        base_horizon=base_horizon,
        base_request_steps=base_horizon - 3,
        start_base=True,
    )
    started = server.handle({"type": "start", "session": "one", "backend": "torch"})
    decisions = []
    server.collector = SimpleNamespace(
        decision=lambda *args, **kwargs: decisions.append((args, kwargs)),
        finish=lambda *args: None,
    )
    assert (started["initial_phase"], started["inference_horizon"]) == (
        "coarse",
        base_horizon,
    )
    request = {
        "type": "infer",
        "session": "one",
        "frame": 0,
        "phase": "coarse",
        "observation": {"states": np.zeros(8, np.float32)},
        "prompt": "task",
    }
    bootstrap = server.handle({**request, "bootstrap": True})
    request_frame = base_horizon - 3
    coarse = server.handle(
        {**request, "frame": request_frame, "target_frame": base_horizon}
    )
    assert bootstrap["actions"].shape == (base_horizon, 7)
    assert coarse["actions"].shape == (base_horizon - 3, 7)
    torch.testing.assert_close(
        policy.calls[1][2],
        torch.arange(32 * 32, dtype=torch.float32).reshape(1, 32, 32)[
            :, request_frame:
        ],
    )
    with pytest.raises(ValueError, match="accepted coarse handoff"):
        server.handle({**request, "phase": "fine", "frame": base_horizon})
    fine = server.handle(
        {
            **request,
            "phase": "fine",
            "frame": base_horizon,
            "accepted": {
                "frame": request_frame,
                "accept_frame": base_horizon,
                "model_version": 4,
            },
        }
    )
    assert fine["actions"].shape == (6, 7)
    assert decisions == []
    torch.testing.assert_close(
        policy.calls[2][2],
        (torch.arange(32 * 32, dtype=torch.float32).reshape(1, 32, 32) + 1000)[:, 3:],
    )
    server.handle(
        {
            "type": "accept",
            "session": "one",
            "frame": base_horizon,
            "accept_frame": base_horizon + 6,
            "model_version": 4,
        }
    )
    assert len(decisions) == 1
    assert decisions[0][0][0] == base_horizon
    server.close()


@pytest.mark.parametrize(
    ("horizon", "request_frame"),
    ((10, 7), (22, 11), (22, 14), (22, 16), (24, 18)),
)
def test_server_uses_one_base_prediction_for_early_handoff(horizon, request_frame):
    class StagePolicy:
        actor_version = 0
        supports_fine_phase = False

        def predict_base_action_batch(
            self, observation, *, horizon, committed, rtc_delay_steps=None
        ):
            values = torch.arange(32, dtype=torch.float32).reshape(1, -1, 1)
            return values.expand(1, -1, 7), {}

        def predict_action_batch(self, observation, *, mode, committed):
            raise AssertionError("Coarse inference must use the base policy path.")

        def close(self):
            pass

    server = RemoteInferenceServer(
        StagePolicy(),
        SimpleNamespace(inference_horizon=6),
        base_horizon=horizon,
        base_request_steps=request_frame,
        start_base=True,
    )
    server.handle({"type": "start", "session": "offset"})
    request = {
        "type": "infer",
        "session": "offset",
        "frame": request_frame,
        "target_frame": horizon,
        "phase": "coarse",
        "observation": {"states": np.zeros(8, np.float32)},
        "prompt": "task",
    }
    offset = horizon - request_frame
    response = server.handle(request)
    np.testing.assert_array_equal(response["actions"][:, 0], np.arange(offset, horizon))
    server.close()


@pytest.mark.parametrize(
    ("horizon", "request_steps", "error"),
    ((22, 0, "exceeds the 32-step"), (22, 10, "requires 2 \\* request_steps")),
)
def test_server_rejects_invalid_coarse_schedule(horizon, request_steps, error):
    with pytest.raises(ValueError, match=error):
        RemoteInferenceServer(
            _TorchPolicy(),
            SimpleNamespace(inference_horizon=6),
            base_horizon=horizon,
            base_request_steps=request_steps,
            start_base=True,
        )


def test_remote_server_uses_learner_phase_for_base_and_rollout_modes():
    class Trainer:
        rollout_mode = "base"

        def submit(self, rows, success, terminal_frame):
            del rows, success, terminal_frame

        def close(self):
            pass

    policy = _Policy()
    server = RemoteInferenceServer(
        policy,
        SimpleNamespace(inference_horizon=6),
        trainer=Trainer(),
        backend="torch",
    )
    request = {
        "session": "phase-test",
        "frame": 0,
        "observation": {"states": np.zeros(8, np.float32)},
        "action_context": None,
        "prompt": "task",
    }
    server.handle({"type": "start", "session": "phase-test", "backend": "torch"})
    server.handle({**request, "type": "infer", "bootstrap": True})
    assert policy.calls[-1][1] == "base"
    server.trainer.rollout_mode = "rollout"
    server.handle({**request, "type": "infer"})
    assert policy.calls[-1][1] == "rollout"
    server.handle(
        {
            "type": "finish",
            "session": "phase-test",
            "frame": 0,
            "success": None,
        }
    )
    server.close()


@pytest.mark.parametrize("disable_debug", (False, True))
def test_server_reports_completed_inference_rate_and_resets_per_session(
    monkeypatch,
    caplog,
    disable_debug,
):
    from rlinf.algorithms.online_residual_rl import remote_server

    caplog.set_level(logging.INFO)
    if disable_debug:
        caplog.set_level(logging.WARNING)
        caplog.set_level(logging.INFO, logger="flexiv.status")
    clock = iter((0.0, 0.1, 1.0, 1.1, 2.0, 2.1, 3.0, 3.1))
    monkeypatch.setattr(remote_server.time, "perf_counter", lambda: next(clock))
    server = RemoteInferenceServer(
        _Policy(), SimpleNamespace(inference_horizon=6), backend="torch"
    )
    request = {
        "type": "infer",
        "session": "first",
        "frame": 0,
        "observation": {"states": np.zeros(8, np.float32)},
        "prompt": "task",
    }
    server.handle({"type": "start", "session": "first", "backend": "torch"})
    bootstrap = server.handle({**request, "bootstrap": True})
    assert bootstrap["inference_hz"] is None
    first = server.handle(request)
    assert first["inference_hz"] is None
    second = server.handle(
        {
            **request,
            "frame": 6,
            "accepted": {"frame": 0, "accept_frame": 6, "model_version": 4},
        }
    )
    assert second["inference_hz"] == pytest.approx(1.0)
    assert "completion_hz=1.00 model_hz=10.00 model_ms=100.00" in caplog.text
    if disable_debug:
        assert "inference started" not in caplog.text
        assert "rollout_mode=eval model_version=4" in caplog.text
    server.handle({"type": "finish", "session": "first", "frame": 6})
    server.handle({"type": "start", "session": "second", "backend": "torch"})
    new_session = server.handle({**request, "session": "second"})
    assert new_session["inference_hz"] is None
    server.close()


def test_torch_server_drops_guidance_after_late_acceptance():
    policy = _TorchPolicy()
    server = RemoteInferenceServer(
        policy, SimpleNamespace(inference_horizon=6), backend="torch"
    )
    server.handle({"type": "start", "session": "late", "backend": "torch"})
    request = {
        "type": "infer",
        "session": "late",
        "frame": 0,
        "observation": {
            "states": np.zeros(8, np.float32),
            "main_images": np.zeros((2, 2, 3), np.uint8),
            "wrist_images": np.zeros((2, 2, 3), np.uint8),
        },
        "action_context": {
            "training_reference": np.array([0, 0, 0, 1, 0, 0, 0], float)
        },
        "prompt": "task",
    }
    server.handle({**request, "bootstrap": True})
    server.handle(request)
    server.handle(
        {
            **request,
            "frame": 6,
            "accepted": {
                "frame": 0,
                "accept_frame": 6,
                "accept_tick": 7,
                "model_version": 4,
            },
        }
    )
    assert policy.calls[2][2] is None
    server.close()


def test_torch_server_rejects_physical_actions_as_rtc_guidance():
    class InvalidPolicy(_TorchPolicy):
        def predict_action_batch(
            self, observation, *, mode, committed, rtc_delay_steps=None
        ):
            physical, _ = super().predict_action_batch(
                observation,
                mode=mode,
                committed=committed,
                rtc_delay_steps=rtc_delay_steps,
            )
            return physical, {"raw_model_actions": physical}

    server = RemoteInferenceServer(
        InvalidPolicy(), SimpleNamespace(inference_horizon=6), backend="torch"
    )
    server.handle({"type": "start", "session": "invalid", "backend": "torch"})
    with pytest.raises(ValueError, match="raw model actions"):
        server.handle(
            {
                "type": "infer",
                "session": "invalid",
                "frame": 0,
                "bootstrap": True,
                "observation": {
                    "states": np.zeros(8, np.float32),
                    "main_images": np.zeros((2, 2, 3), np.uint8),
                    "wrist_images": np.zeros((2, 2, 3), np.uint8),
                },
                "action_context": {
                    "training_reference": np.array([0, 0, 0, 1, 0, 0, 0], float)
                },
                "prompt": "task",
            }
        )
    server.close()


def test_torch_server_resets_session_when_collector_finish_fails():
    server = RemoteInferenceServer(
        _TorchPolicy(), SimpleNamespace(inference_horizon=6), backend="torch"
    )
    server.handle({"type": "start", "session": "one", "backend": "torch"})

    def fail_finish(frame, success):
        raise RuntimeError("flush")

    server.collector = SimpleNamespace(finish=fail_finish)
    with pytest.raises(RuntimeError, match="flush"):
        server.handle({"type": "finish", "session": "one", "frame": -1})
    server.handle({"type": "start", "session": "two", "backend": "torch"})
    server.close()


def test_apx_server_keeps_unconditioned_twelve_action_contract():
    policy = _Policy()
    server = RemoteInferenceServer(
        policy, SimpleNamespace(inference_horizon=6), backend="apx"
    )
    server.handle({"type": "start", "session": "apx", "backend": "apx"})
    request = {
        "type": "infer",
        "session": "apx",
        "frame": 0,
        "observation": {
            "states": np.zeros(8, np.float32),
            "main_images": np.zeros((2, 2, 3), np.uint8),
            "wrist_images": np.zeros((2, 2, 3), np.uint8),
        },
        "action_context": {
            "training_reference": np.array([0, 0, 0, 1, 0, 0, 0], float)
        },
        "prompt": "task",
    }
    bootstrap = server.handle({**request, "bootstrap": True})
    prefetched = server.handle(request)
    next_block = server.handle(
        {
            **request,
            "frame": 6,
            "accepted": {"frame": 0, "accept_frame": 6, "model_version": 4},
        }
    )
    assert [
        block["actions"].shape for block in (bootstrap, prefetched, next_block)
    ] == [(6, 7)] * 3
    assert all(committed is None for _, _, committed in policy.calls)
    server.close()


def test_remote_server_bootstrap_then_execution_half_and_shapes():
    policy = _Policy()
    server = RemoteInferenceServer(
        policy,
        SimpleNamespace(inference_horizon=6),
        expected_inputs={
            "states": (8,),
            "main_images": (320, 320, 3),
            "wrist_images": (224, 224, 3),
        },
    )
    start = server.handle(
        {
            "type": "start",
            "session": "one",
            "input_shapes": {
                "states": (8,),
                "main_images": (320, 320, 3),
                "wrist_images": (224, 224, 3),
            },
        }
    )
    assert start["type"] == "started"
    request = {
        "type": "infer",
        "session": "one",
        "frame": 0,
        "observation": {
            "states": np.zeros(8, np.float32),
            "main_images": np.zeros((320, 320, 3), np.uint8),
            "wrist_images": np.zeros((224, 224, 3), np.uint8),
        },
        "action_context": {
            "training_reference": np.array([0, 0, 0, 1, 0, 0, 0], float)
        },
        "prompt": "task",
    }
    first = server.handle({**request, "bootstrap": True})
    second = server.handle(request)
    np.testing.assert_array_equal(first["actions"], np.arange(84).reshape(12, 7)[:6])
    np.testing.assert_array_equal(second["actions"], np.arange(84).reshape(12, 7)[6:])
    assert policy.calls[0][0]["main_images"].shape == (1, 320, 320, 3)
    assert all(call[2] is None for call in policy.calls)
    server.handle(
        {
            "type": "accept",
            "session": "one",
            "frame": 0,
            "accept_frame": 6,
            "model_version": 4,
        }
    )
    with pytest.raises(ValueError, match="shape"):
        server.handle(
            {
                **request,
                "observation": {
                    **request["observation"],
                    "states": np.zeros(7, np.float32),
                },
            }
        )
    with pytest.raises(ValueError, match="shape"):
        server.handle(
            {
                **request,
                "observation": {
                    **request["observation"],
                    "main_images": np.zeros((320, 320), np.uint8),
                },
            }
        )
    with pytest.raises(ValueError, match="shape"):
        server.handle(
            {
                **request,
                "observation": {
                    **request["observation"],
                    "main_images": np.zeros((224, 224, 3), np.uint8),
                },
            }
        )
    with pytest.raises(ValueError, match="Non-finite"):
        server.handle(
            {
                **request,
                "observation": {**request["observation"], "states": np.full(8, np.nan)},
            }
        )
    server.handle({"type": "finish", "session": "one", "frame": 0, "success": None})
    server.close()
    server.close()
    assert policy.closed


def test_remote_server_discards_stale_inference_before_replacement():
    server = RemoteInferenceServer(_Policy(), SimpleNamespace(inference_horizon=6))
    server.handle({"type": "start", "session": "stale"})
    request = {
        "type": "infer",
        "session": "stale",
        "frame": 0,
        "target_frame": 6,
        "observation": {"states": np.zeros(8, np.float32)},
        "prompt": "insert",
    }
    server.handle(request)

    discarded = server.handle({"type": "discard", "session": "stale", "frame": 0})
    replacement = server.handle(
        {**request, "frame": 6, "target_frame": 12, "type": "infer"}
    )

    assert discarded["type"] == "discarded"
    assert replacement["frame"] == 6
    accepted = server.handle(
        {
            "type": "accept",
            "session": "stale",
            "frame": 6,
            "accept_frame": 12,
            "model_version": 4,
        }
    )
    assert accepted["type"] == "accepted"
    server.close()


def test_server_records_only_robot_accepted_decisions():
    class Collector:
        def __init__(self):
            self.decisions = []
            self.finished = []

        def decision(self, *args, **kwargs):
            self.decisions.append((args, kwargs))

        def finish(self, frame, success):
            self.finished.append((frame, success))

    server = RemoteInferenceServer(_Policy(), SimpleNamespace(inference_horizon=6))
    server.handle({"type": "start", "session": "one"})
    collector = Collector()
    server.collector = collector
    request = {
        "type": "infer",
        "session": "one",
        "frame": 0,
        "observation": {
            "states": np.zeros(8, np.float32),
            "main_images": np.zeros((2, 2, 3), np.uint8),
            "wrist_images": np.zeros((2, 2, 3), np.uint8),
        },
        "action_context": {
            "training_reference": np.array([0, 0, 0, 1, 0, 0, 0], float)
        },
        "prompt": "task",
    }
    server.handle(request)
    assert not collector.decisions
    with pytest.raises(RuntimeError, match="not been accepted"):
        server.handle({**request, "frame": 6})
    server.handle(
        {
            **request,
            "frame": 6,
            "accepted": {"frame": 0, "accept_frame": 6, "model_version": 4},
        }
    )
    assert len(collector.decisions) == 1
    assert collector.decisions[0][0][0] == 0
    server.handle({"type": "finish", "session": "one", "frame": 11, "success": None})
    assert len(collector.decisions) == 1
    assert collector.finished == [(11, None)]
    server.close()


def test_server_does_not_train_on_late_handoff():
    server = RemoteInferenceServer(_Policy(), SimpleNamespace(inference_horizon=6))
    server.handle({"type": "start", "session": "late"})
    decisions = []
    server.collector = SimpleNamespace(
        decision=lambda *args, **kwargs: decisions.append(args),
        finish=lambda *args: None,
    )
    server.handle(
        {
            "type": "infer",
            "session": "late",
            "frame": 0,
            "observation": {
                "states": np.zeros(8, np.float32),
                "main_images": np.zeros((2, 2, 3), np.uint8),
                "wrist_images": np.zeros((2, 2, 3), np.uint8),
            },
            "action_context": {
                "training_reference": np.array([0, 0, 0, 1, 0, 0, 0], float)
            },
            "prompt": "task",
        }
    )
    server.handle(
        {
            "type": "accept",
            "session": "late",
            "frame": 0,
            "accept_frame": 6,
            "accept_tick": 7,
            "model_version": 4,
        }
    )
    assert decisions == []
    server.close()


@pytest.mark.parametrize("base_horizon", (6, 10, 22))
@pytest.mark.parametrize("diagnostics", (False, True))
def test_remote_policy_cross_process_wire_with_fake_server(
    base_horizon, diagnostics, tmp_path
):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    class BasePolicy(_Policy):
        supports_fine_phase = False

        def predict_action_batch(self, observation, *, mode, committed):
            self.calls.append((observation, mode, committed))
            return torch.zeros(1, 32, 7), {}

    policy = _Policy() if base_horizon == 6 else BasePolicy()
    server = RemoteInferenceServer(
        policy,
        SimpleNamespace(inference_horizon=6),
        base_horizon=base_horizon,
        base_request_steps=max(0, 2 * base_horizon - 32),
        start_base=base_horizon != 6,
        expected_inputs={
            "states": (8,),
            "main_images": (2, 2, 3),
            "wrist_images": (2, 2, 3),
        },
    )
    stop = threading.Event()
    thread = threading.Thread(target=server.serve, args=("127.0.0.1", port, stop))
    thread.start()
    remote = None
    try:
        for attempt in range(20):
            try:
                remote = client.RemotePolicy(
                    "127.0.0.1",
                    port,
                    "socket-session",
                    diagnostic_dir=tmp_path if diagnostics else None,
                    input_shapes={
                        "states": (8,),
                        "main_images": (2, 2, 3),
                        "wrist_images": (2, 2, 3),
                    },
                )
                break
            except ConnectionRefusedError:
                time.sleep(0.01)
        assert remote is not None
        assert remote.inference_horizon == base_horizon
        assert remote.fine_horizon == 6
        assert remote.initial_phase == ("coarse" if base_horizon != 6 else "fine")
        assert remote.coarse_origin_schedule == (base_horizon == 22)
        result = remote.call(
            "infer",
            frame=0,
            bootstrap=True,
            observation={
                "states": np.zeros(8, np.float32),
                "main_images": np.zeros((2, 2, 3), np.uint8),
                "wrist_images": np.zeros((2, 2, 3), np.uint8),
            },
            action_context={
                "training_reference": np.array([0, 0, 0, 1, 0, 0, 0], float)
            },
            prompt="task",
        ).result(timeout=2)
        assert result["actions"].shape == (base_horizon, 7)
        assert (
            remote.completion_upper_bound(result["model_completed_at"])
            == result["model_completed_at"]
        )
        remote.call(
            "receipt",
            frame=0,
            info={"action_receipt_physical": np.zeros(7, np.float32)},
        ).result(timeout=2)
        remote.close(0, None)
        if diagnostics:
            from rlinf.algorithms.online_residual_rl.inference_trace import iter_trace

            events = list(iter_trace(tmp_path / "session_socket-session"))
            assert [event["request"]["type"] for event in events] == [
                "start",
                "infer",
                "receipt",
                "finish",
            ]
            saved = events[1]
            assert saved["request"]["observation"]["main_images"].dtype == np.uint8
            np.testing.assert_array_equal(
                saved["response"]["actions"], result["actions"]
            )
            assert saved["response"]["diagnostics"]["full_actions_physical"].shape == (
                1,
                12 if base_horizon == 6 else 32,
                7,
            )
    finally:
        stop.set()
        thread.join(timeout=2)
        server.close()
    assert not thread.is_alive()


def test_remote_policy_preserves_robot_event_order_while_inference_runs():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    started, release = threading.Event(), threading.Event()

    class BlockingPolicy(_Policy):
        def predict_action_batch(self, observation, *, mode, committed):
            if not self.calls:
                started.set()
                assert release.wait(2.0)
            return super().predict_action_batch(
                observation, mode=mode, committed=committed
            )

    server = RemoteInferenceServer(
        BlockingPolicy(), SimpleNamespace(inference_horizon=6)
    )
    received = []
    handle = server.handle

    def recorded(message):
        received.append((message["type"], message.get("frame")))
        return handle(message)

    server.handle = recorded
    stop = threading.Event()
    thread = threading.Thread(target=server.serve, args=("127.0.0.1", port, stop))
    thread.start()
    remote = None
    try:
        for _ in range(20):
            try:
                remote = client.RemotePolicy(
                    "127.0.0.1", port, "priority-session", input_shapes={}
                )
                break
            except ConnectionRefusedError:
                time.sleep(0.01)
        assert remote is not None
        request = {
            "observation": {
                "states": np.zeros(8, np.float32),
                "main_images": np.zeros((2, 2, 3), np.uint8),
                "wrist_images": np.zeros((2, 2, 3), np.uint8),
            },
            "action_context": {
                "training_reference": np.array([0, 0, 0, 1, 0, 0, 0], float)
            },
            "prompt": "task",
        }
        first = remote.call("infer", frame=0, **request)
        assert started.wait(2.0)
        receipt = remote.call(
            "receipt",
            frame=0,
            info={"action_receipt_physical": np.zeros(7, np.float32)},
        )
        accepted = remote.call("accept", frame=0, accept_frame=6, model_version=4)
        next_infer = remote.call("infer", frame=6, **request)
        release.set()
        for future in (first, accepted, next_infer, receipt):
            future.result(timeout=2.0)
        assert received[-4:] == [
            ("infer", 0),
            ("receipt", 0),
            ("accept", 0),
            ("infer", 6),
        ]
        remote.close(6, None)
    finally:
        release.set()
        stop.set()
        thread.join(timeout=2.0)
        server.close()
    assert not thread.is_alive()


def test_myq_camera_serial_crop_and_home_geometry():
    path = Path("/home/qccc/Flexiv/flexiv_rdk/example_py/myq/drag_collect.yaml")
    config = client.load_myq_config(path)
    assert config["cameras"]["main"]["serial_number"] == "254122302900"
    assert config["cameras"]["wrist"]["serial_number"] == "250222073667"
    assert (
        config["cameras"]["main"]["crop_width"],
        config["cameras"]["main"]["crop_height"],
    ) == (320, 320)
    assert (
        config["cameras"]["wrist"]["resize_width"],
        config["cameras"]["wrist"]["resize_height"],
    ) == (224, 224)
    reference = np.asarray(
        config["home"].get(
            "left_tcp_pose_wxyz", config["home"].get("right_tcp_pose_wxyz")
        )
    )
    delta = np.array([0.01, -0.02, 0.03, 0, 0, 0.1])
    target = client.compose_world_delta(reference, delta)
    np.testing.assert_allclose(
        client.world_delta_from_target(reference, target), delta, atol=1e-6
    )


@pytest.mark.parametrize(
    "home,valid",
    [
        ({"left_tcp_pose_wxyz": [0, 0, 0, 1, 0, 0, 0]}, True),
        (
            {
                "left_joint_positions_rad": [0] * 7,
                "right_joint_positions_rad": [0.1] * 7,
            },
            True,
        ),
        ({"left_joint_positions_rad": [0] * 7}, False),
        (
            {"left_joint_positions_rad": [0] * 6, "right_joint_positions_rad": [0] * 7},
            False,
        ),
        ({}, False),
    ],
)
def test_myq_config_accepts_joint_home_or_legacy_tcp_home(tmp_path, home, valid):
    import yaml

    config = {
        "control": {"leader": "right", "control_hz": 30},
        "home": home,
        "cameras": {
            name: {"serial_number": name, "width": 224, "height": 224}
            for name in ("main", "wrist")
        },
    }
    path = tmp_path / "myq.yaml"
    path.write_text(yaml.safe_dump(config))
    if valid:
        assert client.load_myq_config(path)["home"] == home
    else:
        with pytest.raises(ValueError, match="Home|home"):
            client.load_myq_config(path)


@pytest.fixture
def dataset_home_episode(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    directory = tmp_path / "demonstrations" / "data"
    directory.mkdir(parents=True)
    joints = np.array([0.3, -0.2, 0.1, 2.1, -0.1, 0.9, 0.2])
    first = directory / "episode_000000.parquet"
    pq.write_table(
        pa.table(
            {
                "frame_index": [0, 1],
                "observation.joint_state": [joints.tolist(), [0.5] * 7],
                "action": [[1.0] * 8, [1.0] * 8],
            }
        ),
        first,
    )
    pq.write_table(
        pa.table({"observation.joint_state": [[0.6] * 7]}),
        directory / "episode_000001.parquet",
    )
    return first, joints


@pytest.mark.parametrize("source", ("root", "data", "episode"))
def test_dataset_home_reads_first_measured_frame(dataset_home_episode, source):
    first, expected = dataset_home_episode
    path = {
        "root": first.parent.parent,
        "data": first.parent,
        "episode": first,
    }[source]
    selected, joints = client.read_dataset_joint_home(path)
    assert selected == first
    np.testing.assert_array_equal(joints, expected)


@pytest.mark.parametrize(
    "values,frame", [([0.1] * 6, 0), ([np.nan] * 7, 0), ([0.1] * 7, 1)]
)
def test_dataset_home_rejects_invalid_first_frame(tmp_path, values, frame):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "episode_000000.parquet"
    pq.write_table(
        pa.table({"frame_index": [frame], "observation.joint_state": [values]}), path
    )
    with pytest.raises(ValueError, match="seven finite joints|frame_index 0"):
        client.read_dataset_joint_home(path)


def test_dataset_home_rejects_missing_episodes_and_joint_column(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    with pytest.raises(FileNotFoundError, match="No Parquet episodes"):
        client.read_dataset_joint_home(tmp_path)
    path = tmp_path / "episode_000000.parquet"
    pq.write_table(pa.table({"action": [[0.1] * 8]}), path)
    with pytest.raises(ValueError, match="observation.joint_state"):
        client.read_dataset_joint_home(path)


def test_server_accepts_robot_only_home_dataset_config(monkeypatch, server_entry):
    config = _ROOT / "examples/embodiment/config/flexiv_joint_absolute_inference.yaml"
    monkeypatch.setattr(
        sys, "argv", [server_entry.__file__, "--inference-config", str(config)]
    )
    args = server_entry.parse_args()
    assert args.action_representation == "joint_absolute"
    assert args.stage1_only


@pytest.mark.parametrize(
    "representation,error",
    [("eef_delta", "joint_absolute"), ("joint_absolute", "right joint Home")],
)
def test_dataset_home_adapter_rejects_invalid_configuration(
    dataset_home_episode, representation, error
):
    with pytest.raises(ValueError, match=error):
        client.FlexivRobotAdapter(
            {},
            myq_path=Path("unused.yaml"),
            action_representation=representation,
            home_dataset_path=dataset_home_episode[0],
        )


@pytest.mark.parametrize("offset", ([0, 0, 0], [-0.05, 0.02, 0]))
@pytest.mark.parametrize("left_offset", ([0, 0, 0], [0, -0.05, 0]))
@pytest.mark.parametrize("representation", ("joint_absolute", "eef_delta"))
@pytest.mark.parametrize("velocity_setting", ([0.2] * 7, None, "omit"))
@pytest.mark.parametrize("dataset_home", (False, True))
def test_robot_start_returns_independent_joint_homes_without_legacy_pose(
    monkeypatch,
    offset,
    representation,
    left_offset,
    velocity_setting,
    dataset_home,
    dataset_home_episode,
):
    from types import ModuleType

    import yaml

    myq_root = Path("/home/qccc/Flexiv/flexiv_rdk/example_py")
    monkeypatch.syspath_prepend(str(myq_root))
    path = myq_root / "myq/drag_collect.yaml"
    config = yaml.safe_load(path.read_text())
    config["home"] = {
        "left_joint_positions_rad": [0.1] * 7,
        "right_joint_positions_rad": [0.2] * 7,
        "right_tcp_pose_wxyz": [0, 0, 0, 1, 0, 0, 0],
    }
    config["control"]["follower_position_scale"] = [2.0, 3.0, 1.0]
    config["control"]["home_right_position_offset_m"] = offset
    config["control"]["home_left_position_offset_m"] = left_offset
    robots = {}
    modes = SimpleNamespace(
        NRT_JOINT_POSITION="joint", NRT_CARTESIAN_MOTION_FORCE="tcp"
    )
    info = SimpleNamespace(
        q_min=np.full(7, -3), q_max=np.full(7, 3), dq_max=np.ones(7), K_x_nom=np.ones(6)
    )

    class Robot:
        def __init__(self, serial):
            robots[serial] = self
            self.is_left = serial == str(config["robots"]["left"]["serial_number"])
            self.q = np.zeros(7)
            self.mode = None
            self.commands = []

        def connected(self):
            return True

        def fault(self):
            return False

        def operational(self):
            return True

        def info(self):
            return info

        def states(self):
            return SimpleNamespace(
                q=self.q.copy(), tcp_pose=np.r_[self.q[:3], [1, 0, 0, 0]]
            )

        def SwitchMode(self, mode):
            self.mode = mode

        def SetForceControlAxis(self, axes):
            pass

        def SetCartesianImpedance(self, stiffness):
            pass

        def SendJointPosition(self, q, velocities, max_vel, max_acc):
            assert self.mode == "joint"
            self.commands.append((q, velocities, max_vel, max_acc))
            self.q = np.asarray(q)

        def Stop(self):
            pass

    gripper = SimpleNamespace(
        Enable=lambda name: None,
        params=lambda: {},
        states=lambda: SimpleNamespace(width=0.06),
        Stop=lambda: None,
    )
    sdk = ModuleType("flexivrdk")
    sdk.Robot, sdk.Mode = Robot, modes
    sdk.Gripper = lambda robot: gripper
    sdk.Tool = lambda robot: SimpleNamespace(Switch=lambda name: None)

    def model(robot):
        def reachable(pose, seed, free):
            q = np.full(7, 0.1 if robot.is_left else 0.2)
            q[:3] = np.asarray(pose[:3]) + (0 if robot.is_left else 0.2)
            return True, q.tolist()

        return SimpleNamespace(reachable=reachable)

    sdk.Model = model
    monkeypatch.setitem(sys.modules, "flexivrdk", sdk)

    def service(config):
        return SimpleNamespace(start=lambda: None, stop=lambda: None)

    camera = ModuleType("myq.camera_capture")
    camera.CameraCapture = service
    monkeypatch.setitem(sys.modules, "myq.camera_capture", camera)
    drag = ModuleType("myq.drag_runtime")
    drag.ConsoleCommands = service
    drag.SmoothedAdmittance = lambda period, control: None
    drag._force_bias = lambda robots, seconds: {}
    drag._initialize_and_open_follower_gripper = lambda *args: None
    monkeypatch.setitem(sys.modules, "myq.drag_runtime", drag)
    grip = ModuleType("myq.follower_gripper")
    grip.FollowerGripperControl = lambda *args: SimpleNamespace(
        validate_hardware_params=lambda params: None
    )
    monkeypatch.setitem(sys.modules, "myq.follower_gripper", grip)
    clock = _Clock()
    monkeypatch.setattr(
        client,
        "time",
        SimpleNamespace(
            perf_counter=clock.monotonic,
            perf_counter_ns=lambda: int(clock.now * 1e9),
            sleep=clock.sleep,
        ),
    )
    joint_control = {"max_joint_acceleration_rad_s2": [3.0] * 7}
    if velocity_setting != "omit":
        joint_control["max_joint_velocity_rad_s"] = velocity_setting
    adapter = client.FlexivRobotAdapter(
        config,
        myq_path=path,
        initialize_gripper=False,
        action_representation=representation,
        joint_control=joint_control,
        home_dataset_path=(
            dataset_home_episode[0].parent
            if dataset_home and representation == "joint_absolute"
            else None
        ),
    )
    try:
        adapter.start()
        leader = adapter.target["right"].copy()
        leader[:3] += [0.01, 0.02, 0.03]
        sign = (
            1
            if config["control"].get("xy_motion_direction", "opposite") == "same"
            else -1
        )
        np.testing.assert_allclose(
            adapter.aligner.follower_target(leader)[:3] - adapter.target["left"][:3],
            [sign * 0.02, sign * 0.06, 0.03],
        )
        expected_left = np.full(7, 0.1)
        expected_left[:3] += left_offset
        if dataset_home and representation == "joint_absolute":
            expected_left = dataset_home_episode[1]
        np.testing.assert_allclose(adapter.robots["left"].states().q, expected_left)
        assert "left_tcp_pose_wxyz" not in config["home"]
        expected = np.full(7, 0.2)
        expected[:3] += offset
        np.testing.assert_allclose(adapter.robots["right"].states().q, expected)
        assert adapter.robots["left"].mode == (
            "joint" if representation == "joint_absolute" else "tcp"
        )
        assert adapter.robots["right"].mode == "tcp"
        for name, robot in adapter.robots.items():
            np.testing.assert_allclose(adapter.target[name], robot.states().tcp_pose)
        if representation == "joint_absolute":
            action = np.r_[expected_left, 0.06]
            adapter.control_tick(tick=0, action=action, action_context=None)
            command = adapter.robots["left"].commands[-1]
            np.testing.assert_allclose(command[0], action[:7])
            assert command[1] == [0.0] * 7
            assert command[2] == (
                [0.2] * 7
                if isinstance(velocity_setting, list)
                else info.dq_max.tolist()
            )
            assert command[3] == [3.0] * 7
    finally:
        adapter.close()


def test_processed_image_capture_preserves_offline_observation(tmp_path):
    import json

    from PIL import Image

    main = np.full((2, 3, 3), [240, 20, 10], dtype=np.uint8)
    wrist = np.full((3, 2, 3), [10, 20, 240], dtype=np.uint8)
    observation = {
        "tick": 16,
        "states": [0.1] * 7 + [0.06],
        "prompt": "pick up the peg",
        "action_representation": "joint_absolute",
        "tcp_pose_wxyz": [0, 0, 0, 1, 0, 0, 0],
        "last_joint_target_rad": [0.2] * 7,
    }
    recorder = client.ProcessedImageRecorder(tmp_path)
    try:
        assert recorder.submit(16, main, wrist, observation=observation)
    finally:
        recorder.close()
    recorder.close()
    output = recorder.output_dir
    assert (
        json.loads((output / "tick_000016_observation.json").read_text()) == observation
    )
    for name, rgb in (("main", main), ("wrist", wrist)):
        with Image.open(output / f"tick_000016_{name}.png") as image:
            np.testing.assert_array_equal(np.asarray(image), rgb)


def test_robot_snapshot_rejects_stale_or_dead_camera():
    config = {
        "robots": {
            "left": {"gripper_open_width_m": 0.06, "gripper_close_width_m": 0.003}
        }
    }
    robot = client.FlexivRobotAdapter(config, myq_path=Path("unused.yaml"))
    robot.robots["left"] = SimpleNamespace(
        states=lambda: SimpleNamespace(
            q=np.zeros(7), tcp_pose=np.array([0, 0, 0, 1, 0, 0, 0])
        )
    )
    robot.gripper = SimpleNamespace(states=lambda: SimpleNamespace(width=0.06))
    old = SimpleNamespace(
        host_time_ns=time.perf_counter_ns() - 1_000_000_000,
        image_rgb=np.zeros((2, 2, 3), np.uint8),
    )
    camera = SimpleNamespace(
        condition=threading.Condition(),
        buffers={"main": deque([old]), "wrist": deque([old])},
        processes={
            "main": SimpleNamespace(exitcode=None),
            "wrist": SimpleNamespace(exitcode=None),
        },
        shapes={"main": (2, 2, 3), "wrist": (2, 2, 3)},
    )
    robot.cameras = camera
    with pytest.raises(RuntimeError, match="older than 500 ms"):
        robot.snapshot(images=True)
    fresh = SimpleNamespace(
        host_time_ns=time.perf_counter_ns(), image_rgb=old.image_rgb
    )
    camera.buffers = {"main": deque([fresh]), "wrist": deque([fresh])}
    assert robot.snapshot(images=True)[2].shape == (2, 2, 3)
    camera.processes["wrist"].exitcode = 1
    with pytest.raises(RuntimeError, match="exited"):
        robot.snapshot(images=True)


class _Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class _FakeRobot:
    def __init__(self, commands):
        self.config = {
            "control": {"control_hz": 30},
            "recording": {"task": "task"},
            "robots": {
                "left": {"gripper_close_width_m": 0.003, "gripper_open_width_m": 0.06}
            },
        }
        self.console = SimpleNamespace(
            poll=lambda: commands.pop(0) if commands else None
        )
        self.robots = {
            "left": SimpleNamespace(fault=lambda: False),
            "right": SimpleNamespace(fault=lambda: False),
        }
        self.target = {
            "right": np.array([0, 0, 0, 1, 0, 0, 0], float),
            "left": np.array([0, 0, 0, 1, 0, 0, 0], float),
        }
        self.gripper_target = 0.06
        self.gripper_width = 0.06
        self.sent = []
        self.received_actions = []
        self.took_over = False
        self.closed = False
        self._command = None

    def start(self):
        pass

    def snapshot(self, *, images):
        return (
            np.zeros(8, np.float32),
            self.target["left"].copy(),
            np.zeros((2, 2, 3), np.uint8),
            np.zeros((2, 2, 3), np.uint8),
        )

    def read_observation(self, *, tick, prompt):
        del tick, prompt
        state, pose, main, wrist = self.snapshot(images=True)
        return RobotObservation(
            policy_inputs={
                "states": state,
                "main_images": main,
                "wrist_images": wrist,
            },
            action_context={"training_reference": pose},
        )

    def control_tick(self, *, tick, action, action_context):
        del tick
        self.received_actions.append(action)
        command = self._command
        manual = self.took_over
        if manual:
            target = self.target["left"].copy()
            if command == "o":
                self.gripper_target = 0.06
            elif command == "c":
                self.gripper_target = 0.003
        elif action is None:
            target = self.target["left"].copy()
        else:
            target = self.target["left"].copy()
            target[0] = action[0]
        self.send(target, self.gripper_target, manual=manual)
        return RobotControlResult(
            receipt={
                "action_receipt_physical": np.asarray(
                    action if action is not None else np.zeros(7)
                ).copy(),
                "measured_gripper_width": self.gripper_width,
                "human_mask": np.asarray([manual] * 7, dtype=bool),
            },
            policy_enabled=True,
        )

    def poll_control(self, *, tick):
        del tick
        command = self.console.poll()
        self._command = command
        if command == " ":
            self.take_over()
        elif command in {"s", "f", "d", "q"}:
            return RobotTickRequest(
                stop=True,
                success=(True if command == "s" else False if command == "f" else None),
            )
        return RobotTickRequest(policy_enabled=True)

    def stop(self):
        pass

    def take_over(self):
        self.took_over = True

    def send(self, target, gripper_width, *, manual):
        self.target["left"] = target.copy()
        self.gripper_target = gripper_width
        self.sent.append((target.copy(), manual))
        return self.target

    def measured_gripper_width(self):
        return self.gripper_width

    def close(self):
        self.closed = True


class _FakeRemote:
    def __init__(self):
        self.requests = []
        self.receipts = []
        self.accepted = []
        self.standalone_accept_calls = 0
        self.finished = None

    def call(self, kind, **payload):
        future = Future()
        if kind == "infer":
            self.requests.append(payload)
            if payload.get("accepted") is not None:
                self.accepted.append(payload["accepted"])
            physical = np.zeros((6, 7), np.float32)
            physical[:, 0] = np.arange(6) + (0 if payload.get("bootstrap") else 6)
            physical[:, 6] = 0.06
            future.set_result(
                {
                    "frame": payload["frame"],
                    "actions": physical,
                    "action_context": payload["action_context"],
                    "model_version": 0,
                }
            )
        elif kind == "accept":
            self.standalone_accept_calls += 1
            self.accepted.append(payload)
            future.set_result({})
        else:
            self.receipts.append(payload)
            future.set_result({})
        return future

    def close(self, frame, success):
        self.finished = (frame, success)


def test_split_control_handoff_and_manual_receipts():
    clock = _Clock()
    robot = _FakeRobot([None] * 7 + [" "] + [None] * 5)
    policy = _FakeRemote()
    assert (
        client.run_episode(
            robot, policy, max_ticks=10, monotonic=clock.monotonic, sleep=clock.sleep
        )
        == 0
    )
    assert [int(target[0]) for target, _ in robot.sent[:7]] == list(range(7))
    assert policy.requests[0]["bootstrap"]
    assert [request["frame"] for request in policy.requests] == [0, 0, 6]
    assert [accepted["frame"] for accepted in policy.accepted] == [0]
    assert policy.standalone_accept_calls == 0
    assert robot.took_over and robot.sent[7][1]
    assert policy.receipts[7]["info"]["human_mask"][:6].all()
    assert policy.receipts[7]["info"]["policy_proposal_valid"]
    np.testing.assert_array_equal(
        policy.receipts[7]["info"]["policy_proposal_physical"],
        np.asarray([7, 0, 0, 0, 0, 0, 0.06], dtype=np.float32),
    )
    assert policy.finished == (9, None)
    assert robot.closed
    np.testing.assert_allclose(clock.sleeps, [1 / 30] * 10)


@pytest.mark.parametrize("command, success", [(" ", None), ("s", True)])
def test_operator_command_at_handoff_precedes_chunk_accept(command, success):
    robot = _FakeRobot([None] * 6 + [command])
    policy, clock = _FakeRemote(), _Clock()

    assert (
        client.run_episode(
            robot, policy, max_ticks=7, monotonic=clock.monotonic, sleep=clock.sleep
        )
        == 0
    )
    if command == " ":
        assert len(robot.sent) == 7
        assert robot.sent[6][1]
        assert robot.received_actions[6] is not None
        assert policy.receipts[6]["info"]["human_mask"].all()
        assert policy.accepted[0]["accept_frame"] == 6
        assert [request["frame"] for request in policy.requests] == [0, 0, 6]
        assert policy.finished == (6, None)
    else:
        assert len(robot.sent) == 6
        assert [request["frame"] for request in policy.requests] == [0, 0]
        assert policy.finished == (5, success)


def test_initial_camera_timeout_starts_after_robot_home():
    clock = _Clock()

    class SlowHomeRobot(_FakeRobot):
        def __init__(self):
            super().__init__([])
            self.observations = 0

        def start(self):
            clock.now += 20.0

        def read_observation(self, *, tick, prompt):
            self.observations += 1
            if self.observations == 1:
                return None
            return super().read_observation(tick=tick, prompt=prompt)

    robot, policy = SlowHomeRobot(), _FakeRemote()
    assert (
        client.run_episode(
            robot, policy, max_ticks=1, monotonic=clock.monotonic, sleep=clock.sleep
        )
        == 0
    )
    assert robot.observations == 2
    assert robot.closed


def test_flexiv_hold_sends_each_arm_once_and_reports_unchanged_target():
    robot = client.FlexivRobotAdapter(
        {"robots": {"left": {"gripper_open_width_m": 0.06}}},
        myq_path=Path("unused.yaml"),
    )
    pose = np.asarray([0, 0, 0, 1, 0, 0, 0], dtype=float)
    robot.target = {"left": pose.copy(), "right": pose.copy()}
    robot.robots = {
        name: SimpleNamespace(fault=lambda: False) for name in ("left", "right")
    }
    robot.gripper = SimpleNamespace(states=lambda: SimpleNamespace(width=0.025))
    sent = []

    def send_once(targets):
        sent.append({name: value.copy() for name, value in targets.items()})
        robot.target = {name: value.copy() for name, value in targets.items()}

    robot._send = send_once
    receipt = robot.control_tick(
        tick=6,
        action=None,
        action_context={"training_reference": pose},
    )
    assert len(sent) == 1
    np.testing.assert_array_equal(sent[0]["left"], pose)
    np.testing.assert_array_equal(sent[0]["right"], pose)
    np.testing.assert_array_equal(receipt.receipt["action_receipt_physical"][:6], 0)
    assert receipt.receipt["action_receipt_physical"][6] == pytest.approx(0.06)


def test_gripper_receipt_uses_measured_width_after_command():
    robot = _FakeRobot([])
    robot.gripper_width = 0.025
    policy = _FakeRemote()
    clock = _Clock()

    assert (
        client.run_episode(
            robot, policy, max_ticks=1, monotonic=clock.monotonic, sleep=clock.sleep
        )
        == 0
    )
    assert policy.receipts[0]["info"]["measured_gripper_width"] == pytest.approx(0.025)
    assert robot.gripper_target == pytest.approx(0.06)


def test_flexiv_operator_requests_fine_phase_with_i_key():
    robot = client.FlexivRobotAdapter(
        {"robots": {"left": {"gripper_open_width_m": 0.06}}},
        myq_path=Path("unused.yaml"),
    )
    robot.console = SimpleNamespace(poll=lambda: "i")
    assert robot.poll_control(tick=3).enter_fine_phase


@pytest.mark.parametrize(
    ("configured", "override", "expected_init", "expected_width"),
    [
        (True, None, True, 0.06),
        (True, False, False, 0.025),
        (False, None, False, 0.025),
        (False, True, True, 0.06),
    ],
)
@pytest.mark.parametrize("opening_reaches", [True, False])
def test_flexiv_gripper_start_can_keep_current_opening(
    configured, override, expected_init, expected_width, monkeypatch, opening_reaches
):
    events = []
    clock = _Clock()
    monkeypatch.setattr(
        client, "time", SimpleNamespace(perf_counter=clock.monotonic, sleep=clock.sleep)
    )

    class FakeGripper:
        width = 0.025

        def Enable(self, name):
            events.append(("enable", name))

        def params(self):
            return object()

        def states(self):
            if expected_init and opening_reaches:
                self.width = min(0.06, self.width + 0.005)
            return SimpleNamespace(width=self.width, is_moving=False)

    gripper = FakeGripper()
    rdk = SimpleNamespace(
        Gripper=lambda _robot: gripper,
        Tool=lambda _robot: SimpleNamespace(
            Switch=lambda name: events.append(("switch", name))
        ),
    )

    class FakeControl:
        def __init__(self, follower, config):
            del follower, config

        def validate_hardware_params(self, params):
            assert params is not None
            events.append(("validated", None))

    def initialize(_gripper, _control, _config, _name):
        events.append(("initialize", None))
        # Move may return before telemetry reports motion; opening is incomplete.
        gripper.width = 0.04089

    robot = client.FlexivRobotAdapter(
        {
            "robots": {"left": {"gripper_name": "test", "gripper_open_width_m": 0.06}},
            "gripper_control": {
                "initialize_on_start": configured,
                "velocity_m_s": 0.03,
                "force_limit_n": 25.0,
                "initialization_timeout_s": 0.1,
            },
        },
        myq_path=Path("unused.yaml"),
        initialize_gripper=override,
    )
    robot.robots["left"] = object()
    if expected_init and not opening_reaches:
        with pytest.raises(RuntimeError, match="Startup gripper did not reach"):
            robot._start_gripper(rdk, FakeControl, initialize)
        assert clock.now >= 0.1
        assert robot._gripper_worker is None
        return
    robot._start_gripper(rdk, FakeControl, initialize)
    assert (("initialize", None) in events) is expected_init
    assert clock.now > 0 if expected_init else clock.now == 0
    assert events[:3] == [
        ("enable", "test"),
        ("switch", "test"),
        ("validated", None),
    ]
    assert robot.gripper_target == pytest.approx(expected_width)
    robot._gripper_worker.close()


def test_stop_attempts_both_arms_after_first_failure():
    robot = client.FlexivRobotAdapter(
        {"robots": {"left": {"gripper_open_width_m": 0.06}}},
        myq_path=Path("unused.yaml"),
    )
    stopped = []

    def fail_stop():
        stopped.append("left")
        raise RuntimeError("left stop failed")

    robot.robots = {
        "left": SimpleNamespace(Stop=fail_stop),
        "right": SimpleNamespace(Stop=lambda: stopped.append("right")),
    }
    with pytest.raises(RuntimeError, match="left stop failed"):
        robot.stop()
    assert stopped == ["left", "right"]


def test_pending_remote_chunk_holds_until_run_limit_without_stopping():
    class LateRemote(_FakeRemote):
        def call(self, kind, **payload):
            if kind == "infer" and not payload.get("bootstrap"):
                self.requests.append(payload)
                return Future()
            return super().call(kind, **payload)

    robot = _FakeRobot([])
    policy = LateRemote()
    clock = _Clock()
    assert (
        client.run_episode(
            robot, policy, max_ticks=13, monotonic=clock.monotonic, sleep=clock.sleep
        )
        == 0
    )
    assert len(robot.sent) == 13
    for target, _manual in robot.sent[6:]:
        np.testing.assert_array_equal(target, robot.sent[5][0])
    assert policy.finished == (12, None)
    assert robot.closed


def test_fine_network_late_block_is_discarded_even_when_model_was_on_time():
    class DelayedFuture(Future):
        def __init__(self, result):
            super().__init__()
            self.set_result(result)
            self.polls = 0

        def done(self):
            self.polls += 1
            return self.polls >= 8

    class Policy(_FakeRemote):
        def call(self, kind, **payload):
            ready = super().call(kind, **payload)
            if kind == "infer" and not payload.get("bootstrap"):
                result = ready.result()
                result["model_completed_at"] = 0.19
                return DelayedFuture(result)
            return ready

        def completion_upper_bound(self, server_monotonic):
            return server_monotonic

    robot, policy, clock = _FakeRobot([]), Policy(), _Clock()
    assert (
        client.run_episode(
            robot, policy, max_ticks=8, monotonic=clock.monotonic, sleep=clock.sleep
        )
        == 0
    )
    assert [int(target[0]) for target, _ in robot.sent] == [0, 1, 2, 3, 4, 5, 5, 5]
    assert policy.accepted == []
    assert any(item.get("reason") == "late_fine_block" for item in policy.receipts)
    assert [request["frame"] for request in policy.requests] == [0, 0, 7]
    assert policy.finished == (7, None)


def test_fine_model_over_200ms_is_discarded_and_requests_recovery():
    class DelayedFuture(Future):
        def __init__(self, result):
            super().__init__()
            self.set_result(result)
            self.polls = 0

        def done(self):
            self.polls += 1
            return self.polls >= 8

    class Policy(_FakeRemote):
        def call(self, kind, **payload):
            result = super().call(kind, **payload)
            if kind == "infer" and not payload.get("bootstrap"):
                value = result.result()
                value["model_completed_at"] = 0.2334
                return DelayedFuture(value)
            return result

        def completion_upper_bound(self, server_monotonic):
            return server_monotonic

    robot, policy, clock = _FakeRobot([]), Policy(), _Clock()
    assert (
        client.run_episode(
            robot, policy, max_ticks=8, monotonic=clock.monotonic, sleep=clock.sleep
        )
        == 0
    )
    assert [int(target[0]) for target, _ in robot.sent] == [0, 1, 2, 3, 4, 5, 5, 5]
    assert policy.accepted == []
    assert any(item.get("reason") == "late_fine_block" for item in policy.receipts)
    assert policy.finished == (7, None)


def test_boundary_second_poll_accepts_result_and_reuses_bootstrap_observation():
    class BlinkFuture(Future):
        def __init__(self, result):
            super().__init__()
            self.set_result(result)
            self.polls = 0

        def done(self):
            self.polls += 1
            return self.polls >= 7

    class Policy(_FakeRemote):
        def call(self, kind, **payload):
            ready = super().call(kind, **payload)
            if kind == "infer" and not payload.get("bootstrap"):
                return BlinkFuture(ready.result())
            return ready

    class Robot(_FakeRobot):
        def __init__(self):
            super().__init__([])
            self.snapshots = 0

        def snapshot(self, *, images):
            self.snapshots += 1
            state, pose, main, wrist = super().snapshot(images=images)
            state[0] = self.snapshots
            return state, pose, main, wrist

    robot, policy, clock = Robot(), Policy(), _Clock()
    assert (
        client.run_episode(
            robot, policy, max_ticks=7, monotonic=clock.monotonic, sleep=clock.sleep
        )
        == 0
    )
    assert robot.snapshots == 2  # bootstrap/frame 0 shared; frame 6 is new
    np.testing.assert_array_equal(
        policy.requests[0]["observation"]["states"],
        policy.requests[1]["observation"]["states"],
    )
    assert policy.accepted[0]["accept_frame"] == 6
    assert len(robot.sent) == 7


def test_deadline_miss_invalidates_receipts_but_keeps_terminal_label():
    clock = _Clock()

    class SlowRobot(_FakeRobot):
        def send(self, target, gripper_width, *, manual):
            result = super().send(target, gripper_width, manual=manual)
            clock.now += 0.04
            return result

    robot = SlowRobot([None, " ", "c", "s"])
    policy = _FakeRemote()
    assert (
        client.run_episode(
            robot, policy, max_ticks=8, monotonic=clock.monotonic, sleep=clock.sleep
        )
        == 0
    )
    assert policy.receipts[2]["info"]["human_mask"].all()
    assert policy.finished == (2, True)
    assert all(not item["info"]["schedule_valid"] for item in policy.receipts)


@pytest.mark.parametrize("mode,width", [("eef_delta", 7), ("joint_absolute", 8)])
def test_flexiv_inference_config_and_cli_override(monkeypatch, mode, width):
    from omegaconf import OmegaConf

    path = _ROOT / "examples/embodiment/run_flexiv_online_residual.py"
    spec = importlib.util.spec_from_file_location("flexiv_inference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config_name = "joint_absolute" if width == 8 else "tcp_delta"
    config = _ROOT / f"examples/embodiment/config/flexiv_{config_name}_inference.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [str(path), "--inference-config", str(config), "--checkpoint", "override.pt"],
    )
    args = module.parse_args()
    assert args.checkpoint == Path("override.pt")
    assert args.action_representation == mode
    assert args.stage1_only and args.serve and args.backend == "torch"
    recipe = OmegaConf.load(args.training_config).actor.model
    assert recipe.action_dim == recipe.openpi.action_env_dim == width
    assert recipe.openpi.action_horizon == recipe.openpi.model_action_dim == 32


def test_joint_stage1_keeps_absolute_rtc_overlap():
    from rlinf.models.embodiment.openpi_rlinf.inference import OpenPIRLInfStage1Adapter

    model = SimpleNamespace(action_env_dim=8, action_horizon=32)
    model.eval = lambda: model
    policy = OpenPIRLInfStage1Adapter(model, 22)
    raw = torch.randn(1, 16, 32)
    previous = np.array([0, 0, 0, 1, 0, 0, 0])
    current = np.array([1, 2, 3, 0, 1, 0, 0])
    torch.testing.assert_close(
        policy.rebase_committed_actions(raw, previous, current), raw
    )


@pytest.mark.parametrize("width", [7, 8])
def test_stage1_norm_stats_reject_wrong_width_and_invalid_channels(tmp_path, width):
    import json

    from rlinf.models.embodiment.openpi_rlinf.inference import (
        validate_stage1_norm_stats,
    )

    stats = {}
    for name, size in (("state", 8), ("actions", width)):
        stats[name] = {
            "mean": [0.0] * size,
            "std": [1.0] * size,
            "q01": [-1.0] * size,
            "q99": [1.0] * size,
        }
    path = tmp_path / "norm_stats.json"
    path.write_text(json.dumps({"norm_stats": stats}))
    validate_stage1_norm_stats(path, width)
    with pytest.raises(ValueError):
        validate_stage1_norm_stats(path, 15 - width)
    stats["actions"]["q99"][0] = -1.0
    path.write_text(json.dumps({"norm_stats": stats}))
    with pytest.raises(ValueError, match="degenerate"):
        validate_stage1_norm_stats(path, width)
    stats["actions"]["q99"][0] = float("nan")
    path.write_text(json.dumps({"norm_stats": stats}))
    with pytest.raises(ValueError, match="finite"):
        validate_stage1_norm_stats(path, width)


@pytest.mark.parametrize(
    "config_name,width,low,high",
    [
        (
            "pi05_flexiv_online_residual",
            7,
            [-0.01, -0.02, -0.03, -0.1, -0.2, -0.3, 0.003],
            [0.01, 0.02, 0.03, 0.1, 0.2, 0.3, 0.06],
        ),
        (
            "pi05_flexiv_joint_absolute",
            8,
            [-0.3, -0.7, -0.1, 2.1, -0.6, 0.9, -0.1, 0.025],
            [-0.1, 0.0, 0.4, 2.6, 0.1, 1.6, 0.7, 0.06],
        ),
    ],
)
def test_apx_output_transform_decodes_configured_physical_actions(
    config_name, width, low, high
):
    from openpi.models import pi0_config
    from openpi.shared.normalize import NormStats
    from openpi.transforms import Unnormalize, compose

    from rlinf.models.embodiment.openpi.apxinf_adapter import _RLInfOpenPITransforms
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    data = get_openpi_config(config_name).data.create(
        Path("unused"), pi0_config.Pi0Config(pi05=True, action_horizon=32)
    )
    padded_low = np.zeros(32, np.float32)
    padded_high = np.zeros(32, np.float32)
    padded_low[:width] = low
    padded_high[:width] = high
    processor = _RLInfOpenPITransforms.__new__(_RLInfOpenPITransforms)
    processor.action_dim = width
    processor.num_action_chunks = 32
    processor._output_transform = compose(
        [
            Unnormalize(
                {
                    "actions": NormStats(
                        mean=padded_low,
                        std=np.ones(32),
                        q01=padded_low,
                        q99=padded_high,
                    )
                },
                use_quantiles=True,
            ),
            *data.data_transforms.outputs,
        ]
    )
    raw = np.zeros((1, 32, 32), dtype=np.float32)
    first = processor.postprocess_batch(raw, [{"state": np.zeros(8)}])
    second = processor.postprocess_batch(raw, [{"state": np.ones(8) * 10}])
    assert first.shape == (1, 32, width)
    torch.testing.assert_close(first, second)
    np.testing.assert_allclose(
        first[0, 0].numpy(), (np.asarray(low) + np.asarray(high)) / 2, atol=1e-6
    )


def _joint_robot(gripper_policy=None, *, fine_motion_limit_scale=0.5, auto_fine=None):
    commands = []
    gripper = []
    robot = client.FlexivRobotAdapter(
        {
            "robots": {
                "left": {"gripper_open_width_m": 0.06, "gripper_close_width_m": 0.003}
            }
        },
        myq_path=Path("unused.yaml"),
        gripper_policy=gripper_policy,
        auto_fine=auto_fine,
        action_representation="joint_absolute",
        joint_control={
            "max_joint_velocity_rad_s": [0.2] * 7,
            "max_joint_acceleration_rad_s2": [0.5] * 7,
            "fine_motion_limit_scale": fine_motion_limit_scale,
        },
    )
    robot.robots = {
        "left": SimpleNamespace(
            fault=lambda: False, SendJointPosition=lambda *args: commands.append(args)
        ),
        "right": SimpleNamespace(fault=lambda: False),
    }
    for arm in robot.robots.values():
        arm.Stop = lambda: None
        arm.states = lambda: SimpleNamespace(
            q=np.zeros(7), tcp_pose=[0, 0, 0, 1, 0, 0, 0]
        )
        arm.SwitchMode = lambda mode: None
        arm.SetForceControlAxis = lambda axes: None
        arm.SetCartesianImpedance = lambda stiffness: None
        arm.info = lambda: SimpleNamespace(K_x_nom=np.ones(6))
    robot.config["control"] = {"leader_impedance_scale": 0.3}
    robot.joint_target = np.zeros(7)
    robot.joint_limits = (np.full(7, -3.0), np.full(7, 3.0))
    robot._gripper_worker = SimpleNamespace(
        submit=lambda width, **kwargs: gripper.append(width)
    )
    return robot, commands, gripper


@pytest.fixture
def flexiv_pause_sdk(monkeypatch):
    import types

    sdk = types.ModuleType("flexivrdk")
    sdk.Mode = SimpleNamespace(
        NRT_JOINT_POSITION="joint", NRT_CARTESIAN_MOTION_FORCE="tcp"
    )
    monkeypatch.setitem(sys.modules, "flexivrdk", sdk)
    return sdk


def _complete_pause(robot):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = robot.control_tick(tick=0, action=None, action_context=None)
        if result.receipt.get("receipt_valid", True):
            return result
        time.sleep(0.001)
    pytest.fail("Pause did not finish")


@pytest.mark.parametrize(
    "xyz_mm,triggered",
    [
        ([480, 53, 137], True),
        ([495, 53, 137], True),
        ([465, 53, 137], True),
        ([496, 53, 137], False),
        ([494, 67, 137], False),
        ([480, 53, 138], False),
        ([480, 53, 139], False),
        ([float("nan"), 53, 137], False),
    ],
)
def test_auto_fine_uses_measured_tcp_region_and_triggers_once(xyz_mm, triggered):
    robot, _, _ = _joint_robot(
        auto_fine={"xy_center_mm": [480, 53], "xy_radius_mm": 15, "z_below_mm": 138}
    )
    robot.fine_available = True
    robot.console = SimpleNamespace(poll=lambda: None)
    robot.robots["left"].states = lambda: SimpleNamespace(
        tcp_pose=np.r_[np.asarray(xyz_mm) / 1000, [1, 0, 0, 0]]
    )
    assert not robot.poll_control(tick=0).enter_fine_phase
    assert not robot.poll_control(tick=1).enter_fine_phase
    assert robot.poll_control(tick=2).enter_fine_phase is triggered
    assert not robot.poll_control(tick=3).enter_fine_phase


def test_auto_fine_requires_consecutive_samples_and_rearms_for_next_episode(
    monkeypatch,
):
    robot, _, _ = _joint_robot(
        auto_fine={"xy_center_mm": [480, 53], "xy_radius_mm": 15, "z_below_mm": 138}
    )
    robot.fine_available = robot.continuous_episodes = True
    position = np.array([0.480, 0.053, 0.137])
    commands = iter([None, None, None, None, None, None, "f", " ", None, None, None])
    robot.console = SimpleNamespace(poll=lambda: next(commands))
    robot.robots["left"].states = lambda: SimpleNamespace(
        tcp_pose=np.r_[position, [1, 0, 0, 0]]
    )
    monkeypatch.setattr(robot, "_restore_policy_control", lambda: None)
    for tick in (0, 1):
        assert not robot.poll_control(tick=tick).enter_fine_phase
    position[2] = 0.138
    assert not robot.poll_control(tick=2).enter_fine_phase
    position[2] = 0.137
    assert not robot.poll_control(tick=3).enter_fine_phase
    assert not robot.poll_control(tick=4).enter_fine_phase
    assert robot.poll_control(tick=5).enter_fine_phase
    assert robot.poll_control(tick=6).end_episode
    assert robot.poll_control(tick=7).start_episode
    assert not robot.poll_control(tick=8).enter_fine_phase
    assert not robot.poll_control(tick=9).enter_fine_phase
    assert robot.poll_control(tick=10).enter_fine_phase


@pytest.mark.parametrize(
    "state", ["paused", "manual", "ended", "maintenance", "fine", "stage1"]
)
def test_auto_fine_does_not_trigger_outside_coarse_autonomous_control(state):
    robot, _, _ = _joint_robot(
        auto_fine={"xy_center_mm": [480, 53], "xy_radius_mm": 15, "z_below_mm": 138}
    )
    robot.fine_available = state != "stage1"
    robot._paused = state == "paused"
    robot._manual = state == "manual"
    robot._episode_ended = state == "ended"
    robot._maintenance = "home" if state == "maintenance" else None
    robot._control_phase = "fine" if state == "fine" else "coarse"
    robot.console = SimpleNamespace(poll=lambda: None)
    # No SDK state read is needed while the trigger is ineligible.
    for tick in range(4):
        assert not robot.poll_control(tick=tick).enter_fine_phase


@pytest.mark.parametrize(
    "override",
    [
        {"xy_center_mm": [480]},
        {"xy_radius_mm": 0},
        {"z_below_mm": float("nan")},
        {"consecutive_ticks": 0},
        {"consecutive_ticks": True},
        {"consecutive_ticks": 1.5},
        {"unknown": 1},
    ],
)
def test_auto_fine_rejects_invalid_trigger_config(override):
    config = {"xy_center_mm": [480, 53], "xy_radius_mm": 15, "z_below_mm": 138}
    with pytest.raises(ValueError, match="auto_fine"):
        _joint_robot(auto_fine=config | override)


def test_joint_control_sends_absolute_radians_and_holds_accepted_target():
    robot, commands, gripper = _joint_robot()
    action = np.array([-0.2, -0.4, 0.2, 2.3, -0.3, 1.2, 0.3, 0.04])
    result = robot.control_tick(tick=0, action=action, action_context=None)
    np.testing.assert_allclose(commands[0][0], action[:7])
    assert commands[0][1:] == ([0.0] * 7, [0.2] * 7, [0.5] * 7)
    np.testing.assert_allclose(result.receipt["action_receipt_physical"], action)
    held = robot.control_tick(tick=1, action=None, action_context=None)
    assert commands[0] == commands[1]
    assert gripper == [0.04]
    np.testing.assert_array_equal(
        held.receipt["action_receipt_physical"],
        result.receipt["action_receipt_physical"],
    )


@pytest.mark.parametrize("fine_scale", [0.5, 1 / 3, 1.0])
def test_joint_fine_phase_scales_motion_limits_and_coarse_restores_them(fine_scale):
    robot, commands, _ = _joint_robot(fine_motion_limit_scale=fine_scale)
    action = np.r_[np.full(7, 0.1), 0.06]
    for tick, phase in enumerate(("coarse", "fine", "fine", "coarse")):
        robot.control_tick(
            tick=tick,
            action=None if tick == 2 else action,
            action_context={"control_phase": phase},
        )
        scale = fine_scale if phase == "fine" else 1.0
        np.testing.assert_allclose(commands[-1][2], np.full(7, 0.2 * scale))
        np.testing.assert_allclose(commands[-1][3], np.full(7, 0.5 * scale))
        np.testing.assert_allclose(commands[-1][0], action[:7])
    assert robot.joint_control["max_joint_velocity_rad_s"] == [0.2] * 7
    assert robot.joint_control["max_joint_acceleration_rad_s2"] == [0.5] * 7


@pytest.mark.parametrize(
    "scale", [None, True, "0.3", 0, -0.1, 1.1, float("nan"), float("inf")]
)
def test_joint_fine_motion_limit_scale_rejects_invalid_config(scale):
    with pytest.raises(ValueError, match="fine_motion_limit_scale"):
        _joint_robot(fine_motion_limit_scale=scale)


def test_joint_diagnostics_record_measured_state_before_command():
    robot, commands, _ = _joint_robot()
    robot.record_diagnostics = True
    measured = np.arange(7, dtype=float) / 100
    robot.robots["left"].states = lambda: SimpleNamespace(
        q=measured.copy(), tcp_pose=[0, 0, 0, 1, 0, 0, 0]
    )
    robot.gripper = SimpleNamespace(states=lambda: SimpleNamespace(width=0.03))
    action = np.array([-0.2, -0.4, 0.2, 2.3, -0.3, 1.2, 0.3, 0.04])
    result = robot.control_tick(tick=7, action=action, action_context=None)
    np.testing.assert_array_equal(
        result.receipt["diagnostics"]["measured_before_control"],
        np.r_[measured, np.float32(0.03)].astype(np.float32),
    )
    np.testing.assert_allclose(commands[0][0], action[:7])


def test_inference_trace_preserves_arrays_and_reports_completeness(tmp_path):
    import json

    from rlinf.algorithms.online_residual_rl.inference_trace import (
        InferenceTrace,
        iter_trace,
    )

    rgb = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    expected = rgb.copy()
    recorder = InferenceTrace(tmp_path, "recording", {"backend": "torch"})
    recorder.submit(
        {
            "sequence": 0,
            "request": {"type": "infer", "image": rgb},
            "response": {"actions": np.zeros((32, 8), dtype=np.float32)},
        }
    )
    rgb[:] = 0
    recorder.close()
    recorder.close()
    event = next(iter_trace(tmp_path / "session_recording"))
    np.testing.assert_array_equal(event["request"]["image"], expected)
    assert event["response"]["actions"].shape == (32, 8)
    assert json.loads((tmp_path / "session_recording/summary.json").read_text()) == {
        "submitted": 1,
        "written": 1,
        "dropped": 0,
        "errors": 0,
    }


@pytest.mark.parametrize(
    "action",
    [np.zeros(7), np.full(8, np.nan), np.full(8, np.inf), np.array([4.0] + [0.0] * 7)],
)
def test_joint_control_rejects_invalid_action_before_sdk_send(action):
    robot, commands, gripper = _joint_robot()
    with pytest.raises(ValueError):
        robot.control_tick(tick=0, action=action, action_context=None)
    assert not commands and not gripper


def test_joint_server_rejects_tcp_client_before_starting_session():
    from rlinf.algorithms.online_residual_rl.config import OnlineResidualRLConfig

    server = RemoteInferenceServer(
        SimpleNamespace(supports_fine_phase=False),
        OnlineResidualRLConfig(action_representation="joint_absolute"),
        base_horizon=22,
        base_request_steps=16,
        start_base=True,
    )
    with pytest.raises(ValueError, match="representations differ"):
        server.handle(
            {"type": "start", "session": "wrong", "action_representation": "eef_delta"}
        )
    started = server.handle(
        {"type": "start", "session": "joint", "action_representation": "joint_absolute"}
    )
    assert started["action_representation"] == "joint_absolute"
    assert started["action_dim"] == 8


@pytest.mark.parametrize(
    "extra,error",
    [
        (["--online-train"], "stage1-only"),
        (["--actor-checkpoint", "actor.pt"], "stage1-only"),
    ],
)
def test_joint_stage1_rejects_actor_loading(monkeypatch, extra, error):
    path = _ROOT / "examples/embodiment/run_flexiv_online_residual.py"
    spec = importlib.util.spec_from_file_location("flexiv_inference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(path),
            "--inference-config",
            "examples/embodiment/config/flexiv_joint_absolute_inference.yaml",
            *extra,
        ],
    )
    with pytest.raises(ValueError, match=error):
        module.main()


@pytest.mark.parametrize(
    "mode,training_config,width",
    [
        (
            "eef_delta",
            "examples/sft/config/flexiv_rlt_stage1_sft_openpi_pi05.yaml",
            7,
        ),
        (
            "joint_absolute",
            "examples/sft/config/flexiv_joint_stage1_inference.yaml",
            8,
        ),
    ],
)
def test_apx_config_selects_matching_flexiv_decoder(
    tmp_path, mode, training_config, width
):
    import json

    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openpi_rlinf.inference import (
        build_apx_stage1_model_config,
    )

    stats = {
        name: {
            "mean": [0.0] * size,
            "std": [1.0] * size,
            "q01": [-1.0] * size,
            "q99": [1.0] * size,
        }
        for name, size in (("state", 8), ("actions", width))
    }
    stats_path = tmp_path / "norm_stats.json"
    stats_path.write_text(json.dumps({"norm_stats": stats}), encoding="utf-8")
    args = SimpleNamespace(
        training_config=_ROOT / training_config,
        checkpoint=Path("stage1.pt"),
        norm_stats=stats_path,
        num_flow_steps=10,
        stage1_only=True,
        apx_checkpoint=Path("model.safetensors"),
        actor_checkpoint=None,
    )
    stage2 = OmegaConf.load(
        _ROOT / "examples/embodiment/config/flexiv_online_residual_rl_pi05.yaml"
    )

    model_cfg = build_apx_stage1_model_config(
        stage2.rollout.model,
        training_config=args.training_config,
        checkpoint=args.checkpoint,
        norm_stats_path=args.norm_stats,
        apx_checkpoint=args.apx_checkpoint,
        action_representation=mode,
        num_flow_steps=args.num_flow_steps,
        stage1_only=args.stage1_only,
        actor_checkpoint=args.actor_checkpoint,
    )

    assert model_cfg.action_dim == width
    assert model_cfg.openpi_data.norm_stats_path == str(stats_path.resolve())
    assert ("joint_absolute" in model_cfg.openpi.config_name) == (
        mode == "joint_absolute"
    )
    assert model_cfg.apxinf.return_prefix_hidden_states is False


def test_apx_config_rejects_recipe_from_other_action_representation(tmp_path):
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openpi_rlinf.inference import (
        build_apx_stage1_model_config,
    )

    args = SimpleNamespace(
        training_config=_ROOT
        / "examples/sft/config/flexiv_joint_stage1_inference.yaml",
        checkpoint=Path("stage1.pt"),
        norm_stats=tmp_path / "unused.json",
        num_flow_steps=10,
        stage1_only=True,
        apx_checkpoint=Path("model.safetensors"),
        actor_checkpoint=None,
    )
    stage2 = OmegaConf.load(
        _ROOT / "examples/embodiment/config/flexiv_online_residual_rl_pi05.yaml"
    )

    with pytest.raises(ValueError, match="differ from action_representation=eef_delta"):
        build_apx_stage1_model_config(
            stage2.rollout.model,
            training_config=args.training_config,
            checkpoint=args.checkpoint,
            norm_stats_path=args.norm_stats,
            apx_checkpoint=args.apx_checkpoint,
            action_representation="eef_delta",
            num_flow_steps=args.num_flow_steps,
            stage1_only=args.stage1_only,
            actor_checkpoint=args.actor_checkpoint,
        )


@pytest.mark.parametrize(
    "key", ["max_joint_velocity_rad_s", "max_joint_acceleration_rad_s2"]
)
@pytest.mark.parametrize("invalid", [[0.2] * 6, [0.0] * 7, [np.nan] * 7])
def test_joint_motion_config_rejected_before_hardware_start(key, invalid):
    limits = {
        "max_joint_velocity_rad_s": [0.2] * 7,
        "max_joint_acceleration_rad_s2": [0.5] * 7,
        key: invalid,
    }
    with pytest.raises(ValueError, match="positive finite"):
        client.FlexivRobotAdapter(
            {},
            myq_path=Path("unused"),
            action_representation="joint_absolute",
            joint_control=limits,
        )


def test_joint_operator_space_pauses_and_r_starts_manual_request(flexiv_pause_sdk):
    robot, _, _ = _joint_robot()
    took_over = []
    robot.take_over = lambda: took_over.append(True)
    robot.console = SimpleNamespace(poll=lambda: "i")
    assert not robot.poll_control(tick=0).enter_fine_phase
    robot.fine_available = True
    assert robot.poll_control(tick=1).enter_fine_phase
    robot.console = SimpleNamespace(poll=lambda: " ")
    assert not robot.poll_control(tick=2).stop
    assert not took_over and robot._paused and not robot._manual
    _complete_pause(robot)
    robot.console = SimpleNamespace(poll=lambda: "r")
    robot.poll_control(tick=3)
    assert took_over == [True]


@pytest.mark.parametrize("request_steps", (16, 18))
@pytest.mark.parametrize("late_ticks", (0, 2))
def test_joint_server_keeps_raw_rtc_coordinates_through_coarse_handoff(
    request_steps, late_ticks, caplog
):
    caplog.set_level(logging.INFO)
    from rlinf.algorithms.online_residual_rl.config import OnlineResidualRLConfig
    from rlinf.models.embodiment.openpi_rlinf.inference import OpenPIRLInfStage1Adapter

    class JointPolicy:
        supports_fine_phase = False
        conditions_on_committed = True
        actor_version = 0
        model = SimpleNamespace(action_env_dim=8)
        rebase_committed_actions = OpenPIRLInfStage1Adapter.rebase_committed_actions

        def __init__(self):
            self.committed = []
            self.raw = torch.arange(32 * 32, dtype=torch.float32).reshape(1, 32, 32)

        def predict_base_action_batch(
            self, observation, *, horizon, committed, rtc_delay_steps
        ):
            self.committed.append(committed)
            physical = torch.arange(32, dtype=torch.float32).reshape(1, 32, 1)
            physical = physical.expand(1, 32, 8) / 1000 + len(self.committed) / 100
            return physical, {"raw_model_actions": self.raw}

        def close(self):
            pass

    policy = JointPolicy()
    server = RemoteInferenceServer(
        policy,
        OnlineResidualRLConfig(action_representation="joint_absolute"),
        backend="torch",
        base_horizon=22,
        base_request_steps=request_steps,
        start_base=True,
    )
    server.handle(
        {
            "type": "start",
            "session": "joint",
            "backend": "torch",
            "action_representation": "joint_absolute",
        }
    )
    request = {
        "type": "infer",
        "session": "joint",
        "frame": 0,
        "bootstrap": True,
        "target_frame": 0,
        "prompt": "insert",
        "observation": {"states": np.zeros(8)},
        "action_context": {"training_reference": np.array([0, 0, 0, 1, 0, 0, 0])},
    }
    first = server.handle(request)
    assert first["actions"].shape == (22, 8)
    following = server.handle(
        {
            **request,
            "frame": request_steps,
            "bootstrap": False,
            "target_frame": 22,
            "action_context": {"training_reference": np.array([1, 2, 3, 0, 1, 0, 0])},
        }
    )
    assert following["actions"].shape == (request_steps, 8)
    torch.testing.assert_close(policy.committed[-1], policy.raw[:, request_steps:])
    expected_change = np.rad2deg(abs(0.01 - request_steps / 1000))
    assert (
        f"joint prediction continuity frame={request_steps} target_frame=22 "
        f"planned_first_index={22 - request_steps} "
        f"max_same_time_change_deg={expected_change:.4f}"
    ) in caplog.text
    # A late switch skips the expired prefix but retains the original prediction
    # origin, so the next observation gets the correctly aligned remaining tail.
    next_frame = 2 * request_steps
    server.handle(
        {
            **request,
            "frame": next_frame,
            "bootstrap": False,
            "target_frame": request_steps + 22,
            "accepted": {
                "frame": request_steps,
                "accept_frame": 22,
                "accept_tick": 22 + late_ticks,
                "inference_delay_steps": 4 + late_ticks,
                "model_version": 0,
            },
        }
    )
    torch.testing.assert_close(policy.committed[-1], policy.raw[:, request_steps:])
    assert "RTC context cleared" not in caplog.text
    server.close()


@pytest.mark.parametrize("mode", ["eef_delta", "joint_absolute"])
def test_binary_gripper_switches_on_raw_action_and_preserves_physical_receipts(mode):
    if mode == "joint_absolute":
        robot, _, widths = _joint_robot({"binary": True, "threshold": 0.5})
        action = np.array([-0.2, -0.4, 0.2, 2.3, -0.3, 1.2, 0.3, 0.04])
    else:
        widths = []
        robot = client.FlexivRobotAdapter(
            {
                "robots": {
                    "left": {
                        "gripper_open_width_m": 0.06,
                        "gripper_close_width_m": 0.003,
                    }
                }
            },
            myq_path=Path("unused.yaml"),
            gripper_policy={"binary": True},
        )
        robot.robots = {
            name: SimpleNamespace(fault=lambda: False) for name in ("left", "right")
        }
        robot.target = {
            name: np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)
            for name in ("left", "right")
        }
        robot._send = lambda targets: None
        robot._gripper_worker = SimpleNamespace(
            submit=lambda width, **kwargs: widths.append(width)
        )
        action = np.array([0, 0, 0, 0, 0, 0, 0.04])
    context = {
        "training_reference": np.array([0, 0, 0, 1, 0, 0, 0]),
        "normalized_gripper_actions": np.array([0.4, 0.49, 0.5, 0.6, 0.7, 0.4]),
        "action_start_frame": 10,
    }
    receipts = []
    for tick in range(10, 16):
        result = (
            robot._joint_control_tick(action, tick=tick, action_context=context)
            if mode == "joint_absolute"
            else robot._control_tick(tick=tick, action=action, action_context=context)
        )
        receipts.append(result.receipt["action_receipt_physical"][-1])
    np.testing.assert_allclose(receipts, [0.003, 0.003, 0.003, 0.06, 0.06, 0.003])
    assert widths == [0.003, 0.06, 0.003]
    # A stale-result hold must not threshold the last decoded width again.
    held = robot.control_tick(tick=16, action=None, action_context=context)
    assert held.receipt["action_receipt_physical"][-1] == pytest.approx(0.003)
    assert widths == [0.003, 0.06, 0.003]
    np.testing.assert_array_equal(action[-1], 0.04)


@pytest.mark.parametrize(
    "context",
    [
        None,
        {"normalized_gripper_actions": [0.4], "action_start_frame": 1},
        {"normalized_gripper_actions": [np.nan], "action_start_frame": 0},
    ],
)
def test_binary_gripper_rejects_missing_or_invalid_raw_value_before_sending(context):
    robot, commands, widths = _joint_robot({"binary": True})
    action = np.array([-0.2, -0.4, 0.2, 2.3, -0.3, 1.2, 0.3, 0.04])
    with pytest.raises(ValueError, match="gripper"):
        robot.control_tick(tick=0, action=action, action_context=context)
    assert not commands and not widths


def test_binary_gripper_uses_current_tick_after_a_late_handoff():
    robot, _, widths = _joint_robot({"binary": True})
    action = np.array([-0.2, -0.4, 0.2, 2.3, -0.3, 1.2, 0.3, 0.04])
    context = {"normalized_gripper_actions": [0.1, 0.2, 0.8], "action_start_frame": 10}
    robot.gripper_target = 0.003
    robot.control_tick(tick=12, action=action, action_context=context)
    assert widths == [0.06]


@pytest.mark.parametrize("mode,width", [("eef_delta", 7), ("joint_absolute", 8)])
def test_binary_server_slices_raw_gripper_in_step_with_physical_actions(mode, width):
    from rlinf.algorithms.online_residual_rl.config import OnlineResidualRLConfig

    class Policy(_TorchPolicy):
        supports_fine_phase = False

        def predict_base_action_batch(
            self, obs, *, horizon, committed, rtc_delay_steps
        ):
            raw = torch.full((1, 32, 32), -100.0)
            raw[0, :, width - 1] = torch.arange(32.0)
            return torch.full((1, 32, width), 0.04), {"raw_model_actions": raw}

    server = RemoteInferenceServer(
        Policy(),
        OnlineResidualRLConfig(action_representation=mode),
        base_horizon=22,
        base_request_steps=16,
        start_base=True,
        binary_gripper=True,
    )
    with pytest.raises(ValueError, match="binary gripper settings differ"):
        server.handle({"type": "start", "session": "wrong"})
    server.handle({"type": "start", "session": "binary", "binary_gripper": True})
    request = {
        "type": "infer",
        "session": "binary",
        "frame": 0,
        "bootstrap": True,
        "target_frame": 0,
        "prompt": "insert",
        "observation": {"states": np.zeros(8)},
    }
    first = server.handle(request)
    np.testing.assert_array_equal(
        first["action_context"]["normalized_gripper_actions"], np.arange(22)
    )
    assert first["action_context"]["action_start_frame"] == 0
    next_block = server.handle(
        {**request, "frame": 16, "bootstrap": False, "target_frame": 22}
    )
    np.testing.assert_array_equal(
        next_block["action_context"]["normalized_gripper_actions"], np.arange(6, 22)
    )
    assert next_block["action_context"]["action_start_frame"] == 22
    np.testing.assert_allclose(next_block["actions"][:, -1], 0.04)
    server.close()


def test_binary_fine_phase_uses_actor_normalized_gripper_over_base_actions():
    from rlinf.algorithms.online_residual_rl.config import OnlineResidualRLConfig

    class ActorPolicy(_TorchPolicy):
        def predict_action_batch(self, obs, *, mode, committed, rtc_delay_steps):
            raw = torch.full((1, 32, 32), -0.2)
            actor = torch.full((1, 12, 7), 0.7)
            return torch.full((1, 12, 7), 0.04), {
                "raw_model_actions": raw,
                "normalized_actions": actor,
            }

    server = RemoteInferenceServer(
        ActorPolicy(), OnlineResidualRLConfig(), binary_gripper=True
    )
    server.handle({"type": "start", "session": "actor", "binary_gripper": True})
    request = {
        "type": "infer",
        "session": "actor",
        "frame": 0,
        "bootstrap": True,
        "prompt": "insert",
        "observation": {"states": np.zeros(8)},
    }
    result = server.handle(request)
    np.testing.assert_allclose(
        result["action_context"]["normalized_gripper_actions"], 0.7
    )
    next_block = server.handle(
        {**request, "frame": 6, "bootstrap": False, "target_frame": 12}
    )
    assert next_block["action_context"]["action_start_frame"] == 12
    np.testing.assert_allclose(
        next_block["action_context"]["normalized_gripper_actions"], 0.7
    )
    server.close()


@pytest.mark.parametrize(
    "metadata", [{}, {"normalized_actions": torch.full((1, 12, 7), float("nan"))}]
)
def test_binary_server_rejects_unusable_normalized_gripper_output(metadata):
    from rlinf.algorithms.online_residual_rl.config import OnlineResidualRLConfig

    class Policy(_Policy):
        def predict_action_batch(self, obs, *, mode, committed):
            return torch.full((1, 12, 7), 0.04), metadata

    server = RemoteInferenceServer(
        Policy(), OnlineResidualRLConfig(), binary_gripper=True
    )
    server.handle({"type": "start", "session": "invalid", "binary_gripper": True})
    with pytest.raises(ValueError, match="(unnormalization|finite)"):
        server.handle(
            {
                "type": "infer",
                "session": "invalid",
                "frame": 0,
                "bootstrap": True,
                "prompt": "insert",
                "observation": {"states": np.zeros(8)},
            }
        )
    server.close()


def test_inference_cli_can_enable_binary_gripper_without_changing_config(monkeypatch):
    path = _ROOT / "examples/embodiment/run_flexiv_online_residual.py"
    spec = importlib.util.spec_from_file_location("flexiv_inference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(path),
            "--inference-config",
            "examples/embodiment/config/flexiv_joint_absolute_inference.yaml",
            "--binary-gripper",
        ],
    )
    assert module.parse_args().binary_gripper


@pytest.mark.parametrize("binary", [False, True])
def test_joint_stage2_server_robot_receipts_persist_without_tcp_reference(
    tmp_path, binary
):
    from rlinf.algorithms.online_residual_rl.action_mapping import (
        FlexivJointActionMapping,
    )
    from rlinf.algorithms.online_residual_rl.config import OnlineResidualRLConfig
    from rlinf.algorithms.online_residual_rl.networks import ResidualActor
    from rlinf.algorithms.online_residual_rl.replay import EpisodeReplay

    cfg = OnlineResidualRLConfig(
        action_representation="joint_absolute", z_dim=4, hidden_dim=8
    )
    offset = torch.tensor([-0.2, -0.4, 0.2, 2.3, -0.3, 1.2, 0.3, 0.04])
    mapping = FlexivJointActionMapping(
        offset,
        torch.tensor([0.1] * 7 + [0.02]),
        gripper_range=(0.003, 0.06),
        binary_gripper=binary,
    )

    class Policy:
        conditions_on_committed = True
        actor_version = 0

        def __init__(self):
            self.actor = ResidualActor(cfg)

        def predict_action_batch(self, obs, *, mode, committed, rtc_delay_steps):
            reference = torch.full((1, 12, 8), 0.4)
            z, proprio = torch.zeros(1, 4), torch.zeros(1, 8)
            proposal = self.actor(z, proprio, reference)
            raw = torch.zeros(1, 32, 32)
            raw[:, :, :8] = 0.4
            return mapping.decode(proposal), {
                "raw_model_actions": raw,
                "normalized_actions": proposal,
                "forward_inputs": {
                    "z_rl": z,
                    "proprio": proprio,
                    "base_ref": reference,
                    "actor_output": proposal,
                },
            }

        def rebase_committed_actions(self, *args):
            raise AssertionError("Absolute joint chunks must not use TCP rebasing")

        def encode_receipts(self, physical):
            return mapping.encode(physical)

        def close(self):
            pass

    replay = EpisodeReplay(cfg, storage_dir=tmp_path)
    server = RemoteInferenceServer(
        Policy(),
        cfg,
        binary_gripper=binary,
        commit_episode=lambda rows, success, terminal: replay.commit_episode(
            rows, success=success, terminal_frame=terminal
        ),
    )
    server.handle(
        {
            "type": "start",
            "session": "joint-stage2",
            "action_representation": "joint_absolute",
            "binary_gripper": binary,
        }
    )
    robot, commands, _ = _joint_robot({"binary": binary})
    if binary:
        # Already closed: this test checks transition persistence during execution.
        robot.gripper_target = 0.003
    robot.fine_available = True
    robot.console = SimpleNamespace(poll=lambda: "i")
    assert robot.poll_control(tick=0).enter_fine_phase
    request = {
        "type": "infer",
        "session": "joint-stage2",
        "prompt": "insert",
        "observation": {"states": np.zeros(8)},
    }
    queued = server.handle(
        {**request, "frame": 0, "bootstrap": True, "target_frame": 0}
    )
    receipts = []
    pending = None
    for tick in range(30):
        if tick % 6 == 0:
            if tick:
                server.handle(
                    {
                        "type": "accept",
                        "session": "joint-stage2",
                        "frame": tick - 6,
                        "accept_frame": tick,
                        "model_version": 0,
                    }
                )
                assert pending is not None
                queued = pending
            pending = server.handle(
                {**request, "frame": tick, "bootstrap": False, "target_frame": tick + 6}
            )
        result = robot.control_tick(
            tick=tick,
            action=queued["actions"][tick % 6],
            action_context=queued["action_context"],
        )
        info = {
            **result.receipt,
            "policy_proposal_physical": queued["actions"][tick % 6],
        }
        receipts.append(info["action_receipt_physical"])
        server.handle(
            {"type": "receipt", "session": "joint-stage2", "frame": tick, "info": info}
        )
    server.handle(
        {"type": "finish", "session": "joint-stage2", "frame": 29, "success": True}
    )
    assert len(replay) >= 1
    payload = replay.store.load_episode("joint-stage2")
    for row in payload["transitions"]:
        start = payload["features"][row["obs_id"]]["frame_index"]
        np.testing.assert_allclose(
            row["exec_physical"].numpy(),
            np.stack(receipts[start : start + row["valid_len"]]),
        )
        torch.testing.assert_close(row["exec"], mapping.encode(row["exec_physical"]))
    np.testing.assert_allclose(commands[0][0], (offset[:7] + 0.04).numpy())
    assert payload["action_representation"] == "joint_absolute"
    assert payload["transitions"][-1]["terminal"]
    server.close()


@pytest.mark.parametrize("binary", [False, True])
def test_joint_projection_matches_sequential_robot_receipts_and_deadband(binary):
    from rlinf.algorithms.online_residual_rl.action_mapping import (
        FlexivJointActionMapping,
    )

    robot, _, _ = _joint_robot({"binary": binary})
    robot.gripper_target = 0.04
    offset = torch.tensor([0.0] * 7 + [0.04])
    scale = torch.tensor([1.0] * 7 + [0.02])
    mapping = FlexivJointActionMapping(
        offset, scale, gripper_range=(0.003, 0.06), binary_gripper=binary
    )
    raw = torch.zeros(1, 6, 8, requires_grad=True)
    delta = torch.zeros_like(raw)
    delta[0, :, 7] = torch.tensor([0.025, 0.05, 0.1, 0.5, 0.6, 0.4])
    values = raw + delta
    committed = mapping.encode(offset.expand(1, 6, 8))
    predicted = mapping.decode(mapping(values, committed=committed))
    context = {
        "normalized_gripper_actions": values.detach()[0, :, 7].numpy(),
        "action_start_frame": 0,
    }
    receipts = []
    for frame, action in enumerate(mapping.decode(values).detach()[0].numpy()):
        receipts.append(
            robot._joint_control_tick(
                action, tick=frame, action_context=context
            ).receipt["action_receipt_physical"]
        )
    np.testing.assert_allclose(predicted.detach()[0].numpy(), receipts, atol=1e-7)
    if not binary:
        np.testing.assert_allclose(
            predicted.detach()[0, :3, 7], [0.04, 0.04, 0.042], atol=1e-7
        )
        predicted.sum().backward()
        assert raw.grad[0, 0, 7] == 0 and raw.grad[0, 2, 7] > 0


def test_joint_manual_records_only_changed_dimensions_and_persistent_gripper():
    robot, commands, gripper = _joint_robot()
    robot._manual = True
    human_q = np.zeros(7)
    human_q[2] = 0.05
    right_commands = []
    robot.robots["right"].SendJointPosition = lambda *args: right_commands.append(args)
    robot.manual_targets = lambda: {"left": np.zeros(7), "right": np.zeros(7)}
    robot._joint_drag_controllers = {
        name: SimpleNamespace(
            command=lambda pose, now: human_q.copy(),
            max_velocity=np.full(7, 0.12),
            max_acceleration=np.full(7, 0.3),
        )
        for name in ("left", "right")
    }
    action = np.array([0.0] * 7 + [0.06])
    initial = robot.control_tick(tick=0, action=action, action_context=None).receipt
    assert np.flatnonzero(initial["human_mask"]).tolist() == [2]
    assert np.flatnonzero(initial["pair_mask"]).tolist() == [2]
    assert commands[0][1:] == ([0.0] * 7, [0.12] * 7, [0.3] * 7)
    robot.console = SimpleNamespace(poll=lambda: "c")
    robot.poll_control(tick=1)
    closed = robot.control_tick(tick=1, action=action, action_context=None).receipt
    robot.console = SimpleNamespace(poll=lambda: None)
    robot.poll_control(tick=2)
    held = robot.control_tick(tick=2, action=action, action_context=None).receipt
    for receipt in (closed, held):
        assert np.flatnonzero(receipt["human_mask"]).tolist() == [2, 7]
        np.testing.assert_array_equal(receipt["pair_mask"], receipt["human_mask"])
        np.testing.assert_allclose(
            receipt["action_receipt_physical"], [0, 0, 0.05, 0, 0, 0, 0, 0.003]
        )
        np.testing.assert_allclose(receipt["policy_proposal_physical"], action)
    assert gripper == [0.003] and len(right_commands) == 3
    # With the model proposing the same targets there is no human correction.
    same = robot.control_tick(
        tick=3, action=np.r_[human_q, 0.003], action_context=None
    ).receipt
    assert not same["human_mask"].any() and not same["pair_mask"].any()


def test_joint_r_zeroes_force_then_uses_myq_joint_impedance_ik_contract(monkeypatch):
    import types

    robot, _, _ = _joint_robot()
    pose = np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)
    modes, stiffness, builds = [], [], []
    for name in ("left", "right"):
        robot.robots[name].states = lambda: SimpleNamespace(
            tcp_pose=pose,
            q=np.zeros(7),
            ext_wrench_in_world=np.ones(6),
            ext_wrench_in_tcp=np.ones(6) * 2,
        )
        robot.robots[name].info = lambda: SimpleNamespace(
            K_q_nom=np.ones(7), K_x_nom=np.ones(6)
        )
        robot.robots[name].SwitchMode = lambda mode, name=name: modes.append(
            (name, mode)
        )
        robot.robots[name].SetJointImpedance = lambda values, name=name: (
            stiffness.append((name, values))
        )
    robot.config["control"] = {
        "leader_joint_impedance_scale": 0.3,
        "leader_impedance_scale": 0.3,
        "control_hz": 30,
    }
    robot.config["recording"] = {"force_bias_before_manual_seconds": 0.0}
    robot.aligner = SimpleNamespace(rebase=lambda a, b: None)
    robot.admittance = SimpleNamespace(reset=lambda: None)
    robot.follower_filter = SimpleNamespace(reset=lambda pose: None)
    robot.console = SimpleNamespace(poll=lambda: " ")
    sdk = types.ModuleType("flexivrdk")
    sdk.Mode = SimpleNamespace(
        NRT_JOINT_IMPEDANCE="joint_impedance",
        NRT_JOINT_POSITION="joint",
        NRT_CARTESIAN_MOTION_FORCE="tcp",
    )
    sdk.Model = lambda arm: arm
    module = types.ModuleType("myq.joint_control")

    def controller(*args):
        builds.append(args)
        return SimpleNamespace(close=lambda: None)

    module.JointTargetController = controller
    monkeypatch.setitem(sys.modules, "flexivrdk", sdk)
    monkeypatch.setitem(sys.modules, "myq", types.ModuleType("myq"))
    monkeypatch.setitem(sys.modules, "myq.joint_control", module)
    assert not robot.poll_control(tick=0).stop
    _complete_pause(robot)
    modes.clear()
    robot.console = SimpleNamespace(poll=lambda: "r")
    robot.poll_control(tick=1)
    result = robot.control_tick(tick=1, action=np.zeros(8), action_context=None)
    assert not result.receipt["schedule_valid"]
    np.testing.assert_allclose(robot.force_bias["right"]["world"], np.ones(6))
    assert robot._manual
    assert modes == [("left", "joint_impedance"), ("right", "joint_impedance")]
    assert stiffness == [("left", [1.0] * 7), ("right", [0.3] * 7)]
    assert len(builds) == 2 and all(args[-1] == 1 / 30 for args in builds)


def test_task_timeout_labels_failure_and_stops_robot():
    clock, policy, robot = _Clock(), _FakeRemote(), _FakeRobot([])
    client.run_episode(
        robot,
        policy,
        task_timeout_seconds=0.1,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    assert robot.closed and policy.finished == (2, False)


def test_invalid_fine_block_holds_and_requests_fresh_prediction():
    class Policy(_FakeRemote):
        def call(self, kind, **payload):
            future = super().call(kind, **payload)
            if (
                kind == "infer"
                and not payload.get("bootstrap")
                and len(self.requests) == 2
            ):
                response = future.result()
                response["actions"][:] = np.nan
            return future

    robot, clock, policy = _FakeRobot([]), _Clock(), Policy()
    client.run_episode(
        robot, policy, max_ticks=13, monotonic=clock.monotonic, sleep=clock.sleep
    )
    assert [request["frame"] for request in policy.requests] == [0, 0, 6, 12]
    assert any(item.get("reason") == "invalid_action_block" for item in policy.receipts)
    assert all(target[0] == 5 for target, _ in robot.sent[6:12])
    assert robot.sent[12][0][0] == 6


def test_joint_manual_sdk_targets_and_masks_are_saved_in_transition(tmp_path):
    from rlinf.algorithms.online_residual_rl.config import OnlineResidualRLConfig
    from rlinf.algorithms.online_residual_rl.online import EpisodeCollector
    from rlinf.algorithms.online_residual_rl.replay import EpisodeReplay

    cfg = OnlineResidualRLConfig(
        action_representation="joint_absolute", z_dim=4, hidden_dim=8
    )
    replay = EpisodeReplay(cfg, storage_dir=tmp_path)
    robot, commands, _ = _joint_robot()
    robot._manual = True
    robot.robots["right"].SendJointPosition = lambda *args: None
    robot.manual_targets = lambda: {"left": np.zeros(7), "right": np.zeros(7)}
    target = np.zeros(7)
    robot._joint_drag_controllers = {
        name: SimpleNamespace(
            command=lambda pose, now: target.copy(),
            max_velocity=np.full(7, 0.12),
            max_acceleration=np.full(7, 0.3),
        )
        for name in ("left", "right")
    }
    metadata = {
        "forward_inputs": {
            "z_rl": torch.zeros(1, 4),
            "proprio": torch.zeros(1, 8),
            "base_ref": torch.zeros(1, 12, 8),
            "actor_output": torch.zeros(1, 12, 8),
        }
    }
    collector = EpisodeCollector(
        cfg,
        lambda x: x,
        lambda rows, success, terminal: replay.commit_episode(
            rows, success=success, terminal_frame=terminal
        ),
        episode_id="joint-manual",
    )
    for frame in range(18):
        target[:] = 0
        if frame == 7:
            target[3] = 0.04
        result = robot.control_tick(
            tick=frame, action=np.r_[np.zeros(7), 0.06], action_context=None
        )
        collector.receipt(frame, result.receipt)
        if frame % 6 == 5:
            collector.decision(frame - 5, metadata, 0)
    collector.finish(17, True)
    assert len(replay) == 2
    saved = replay.store.load_episode("joint-manual")
    row = saved["transitions"][0]
    np.testing.assert_allclose(
        row["exec_physical"][:, :7], np.asarray([cmd[0] for cmd in commands[:12]])
    )
    correction = saved["corrections"][row["correction_id"]]
    assert torch.nonzero(correction["human_mask"]).tolist() == [[7, 3]]
    assert torch.nonzero(correction["pair_mask"]).tolist() == [[7, 3]]
    torch.testing.assert_close(correction["proposal_values"], torch.tensor([0.0]))


@pytest.mark.parametrize("discard", [False, True])
def test_server_unknown_label_stages_valid_episode_unless_operator_discards(
    tmp_path, discard
):
    from rlinf.algorithms.online_residual_rl.config import OnlineResidualRLConfig
    from rlinf.algorithms.online_residual_rl.storage import PendingEpisodeStore

    cfg = OnlineResidualRLConfig(
        action_representation="joint_absolute", z_dim=4, hidden_dim=8
    )

    class JointPolicy(_Policy):
        def predict_action_batch(self, observation, *, mode, committed):
            return torch.zeros(1, 12, 8), {
                "forward_inputs": {
                    "z_rl": torch.zeros(1, 4),
                    "proprio": torch.zeros(1, 8),
                    "base_ref": torch.zeros(1, 12, 8),
                    "actor_output": torch.zeros(1, 12, 8),
                }
            }

    committed = []
    store = PendingEpisodeStore(tmp_path, cfg)
    server = RemoteInferenceServer(
        JointPolicy(),
        cfg,
        pending_store=store,
        commit_episode=lambda *args: committed.append(args),
    )
    server.handle(
        {
            "type": "start",
            "session": "stopped",
            "action_representation": "joint_absolute",
        }
    )
    server.handle(
        {
            "type": "infer",
            "session": "stopped",
            "frame": 0,
            "observation": {"states": np.zeros(8)},
            "prompt": "task",
        }
    )
    server.handle(
        {
            "type": "accept",
            "session": "stopped",
            "frame": 0,
            "accept_frame": 6,
            "model_version": 4,
        }
    )
    for frame in range(12):
        server.handle(
            {
                "type": "receipt",
                "session": "stopped",
                "frame": frame,
                "info": {
                    "action_receipt_physical": np.zeros(8),
                    "action_representation": "joint_absolute",
                },
            }
        )
    server.handle(
        {
            "type": "finish",
            "session": "stopped",
            "frame": 11,
            "success": None,
            "discard": discard,
        }
    )
    server.close()
    assert not committed
    assert len(list((tmp_path / "pending").glob("*.pt"))) == (0 if discard else 1)


def test_joint_pause_ignores_policy_and_gripper_keys_hold_the_arm(flexiv_pause_sdk):
    robot, commands, gripper = _joint_robot()
    robot.console = SimpleNamespace(poll=lambda: " ")
    robot.poll_control(tick=0)
    _complete_pause(robot)
    result = robot.control_tick(
        tick=0, action=np.r_[np.ones(7), 0.01], action_context=None
    )
    np.testing.assert_array_equal(commands[-1][0], np.zeros(7))
    assert not result.receipt["schedule_valid"]
    assert not result.receipt["human_mask"].any()
    started = result.receipt["command_started_perf_counter_ns"]
    assert started <= result.receipt["command_returned_perf_counter_ns"]
    for tick, key, expected in ((1, "c", 0.003), (2, "o", 0.06)):
        robot.console = SimpleNamespace(poll=lambda key=key: key)
        robot.poll_control(tick=tick)
        result = robot.control_tick(
            tick=tick, action=np.r_[np.ones(7), 0.01], action_context=None
        )
        np.testing.assert_array_equal(commands[-1][0], np.zeros(7))
        assert gripper[-1] == expected
        assert not result.receipt["schedule_valid"]
    restored = []
    robot._restore_policy_control = lambda: restored.append(True)
    robot.console = SimpleNamespace(poll=lambda: " ")
    assert robot.poll_control(tick=3).reset_policy
    assert restored and not robot._paused


@pytest.mark.parametrize("representation", ["joint_absolute", "eef_delta"])
def test_space_stops_both_arms_before_holding_settled_positions(
    flexiv_pause_sdk, representation
):
    robot, commands, gripper = _joint_robot()
    robot.action_representation = representation
    robot.config["control"].update(
        max_linear_velocity_m_s=0.1,
        max_angular_velocity_rad_s=0.2,
        max_linear_acceleration_m_s2=0.3,
        max_angular_acceleration_rad_s2=0.4,
    )
    entered = {name: threading.Event() for name in robot.robots}
    release = threading.Event()
    settled = {name: threading.Event() for name in robot.robots}
    modes = []
    cartesian = {}

    def stop(name):
        entered[name].set()
        assert release.wait(5)
        settled[name].set()

    for name, arm in robot.robots.items():
        arm.Stop = lambda name=name: stop(name)
        arm.states = lambda name=name: SimpleNamespace(
            q=np.full(7, 0.3 if settled[name].is_set() else 0.1),
            tcp_pose=[0.5 if settled[name].is_set() else 0.4, 0, 0.2, 1, 0, 0, 0],
        )
        arm.SwitchMode = lambda mode, name=name: modes.append((name, mode))
        arm.SendCartesianMotionForce = lambda pose, *args, name=name: cartesian.update(
            {name: pose}
        )
    robot.joint_target = np.ones(7)
    robot.console = SimpleNamespace(poll=lambda: " ")
    try:
        robot.poll_control(tick=0)
        assert all(event.wait(2) for event in entered.values())
        result = robot.control_tick(tick=0, action=None, action_context=None)
        assert not result.receipt["receipt_valid"]
        assert not commands and not cartesian and not modes
        for key in (" ", "r", "h", "o", "c", "i"):
            robot.console = SimpleNamespace(poll=lambda key=key: key)
            request = robot.poll_control(tick=1)
            assert not request.reset_policy and not request.enter_fine_phase
        assert not gripper and not commands and not modes
        release.set()
        result = _complete_pause(robot)
        assert not result.receipt["schedule_valid"]
        if representation == "joint_absolute":
            np.testing.assert_allclose(commands[-1][0], np.full(7, 0.3))
        else:
            for pose in cartesian.values():
                np.testing.assert_allclose(pose, [0.5, 0, 0.2, 1, 0, 0, 0])
        assert len(modes) == 2
        robot.console = SimpleNamespace(poll=lambda: " ")
        assert robot.poll_control(tick=2).reset_policy
    finally:
        release.set()
        robot.stop()


def test_pause_stop_failure_excludes_receipts_and_never_restores_control(
    flexiv_pause_sdk,
):
    robot, commands, _ = _joint_robot()
    stopped = threading.Event()
    modes = []

    def fail():
        raise RuntimeError("SDK stop failed")

    robot.robots["left"].Stop = fail
    robot.robots["right"].Stop = stopped.set
    for arm in robot.robots.values():
        arm.SwitchMode = modes.append
    robot.console = SimpleNamespace(poll=lambda: " ")
    robot.poll_control(tick=0)
    try:
        with pytest.raises(RuntimeError, match="SDK stop failed"):
            _complete_pause(robot)
        assert stopped.is_set() and not modes and not commands
    finally:
        with pytest.raises(RuntimeError, match="SDK stop failed"):
            robot.stop()


def test_shutdown_waits_for_pause_restoration_before_final_stop(flexiv_pause_sdk):
    robot, _, _ = _joint_robot()
    entered = {name: threading.Event() for name in robot.robots}
    release = threading.Event()
    cleanup_entered = threading.Event()
    events = []

    def stop(name):
        entered[name].set()
        assert release.wait(5)
        events.append(("stop", name))

    for name, arm in robot.robots.items():
        arm.Stop = lambda name=name: stop(name)
        arm.SwitchMode = lambda mode, name=name: events.append(("restore", name))
    robot.console = SimpleNamespace(poll=lambda: " ")
    robot.poll_control(tick=0)
    assert all(event.wait(2) for event in entered.values())

    def shutdown():
        cleanup_entered.set()
        robot.stop()

    cleanup = threading.Thread(target=shutdown)
    cleanup.start()
    try:
        assert cleanup_entered.wait(2)
        assert not events
    finally:
        release.set()
        cleanup.join(5)
    assert not cleanup.is_alive()
    assert events[-2:] == [("stop", "left"), ("stop", "right")]
    assert sum(kind == "restore" for kind, _ in events) == 2
    saved = list(events)
    robot.stop()
    assert events == saved


@pytest.mark.parametrize("label", ["s", "f", "d"])
def test_episode_label_during_pause_waits_for_stop_before_hold(flexiv_pause_sdk, label):
    robot, commands, _ = _joint_robot()
    robot.continuous_episodes = True
    release = threading.Event()
    for arm in robot.robots.values():
        arm.Stop = lambda: release.wait(5)
    robot.console = SimpleNamespace(poll=lambda: " ")
    robot.poll_control(tick=0)
    try:
        robot.console = SimpleNamespace(poll=lambda: label)
        request = robot.poll_control(tick=1)
        assert request.end_episode and not request.stop
        assert request.discard_episode == (label == "d")
        receipt = robot.control_tick(tick=1, action=None, action_context=None).receipt
        assert not receipt["receipt_valid"] and not commands
        release.set()
        _complete_pause(robot)
        robot.console = SimpleNamespace(poll=lambda: " ")
        assert robot.poll_control(tick=2).start_episode
    finally:
        release.set()
        robot.stop()


def test_runtime_home_resume_returns_to_coarse_until_next_i():
    class Robot(_FakeRobot):
        def __init__(self):
            super().__init__([])
            self.phases = []

        def poll_control(self, *, tick):
            return RobotTickRequest(
                reset_policy=tick == 2,
                reset_to_coarse=tick == 2,
                enter_fine_phase=tick == 15,
            )

        def control_tick(self, *, tick, action, action_context):
            self.phases.append(action_context["control_phase"])
            return super().control_tick(
                tick=tick, action=action, action_context=action_context
            )

    class Policy(_FakeRemote):
        def __init__(self):
            super().__init__()
            self.resets = []

        def call(self, kind, **payload):
            if kind == "reset_context":
                self.resets.append(payload)
            future = super().call(kind, **payload)
            if kind == "infer":
                steps = 10 if payload["phase"] == "coarse" else 6
                future.result()["actions"] = np.zeros((steps, 7), np.float32)
            return future

    clock, robot, policy = _Clock(), Robot(), Policy()
    runtime = RemoteRobotRuntime(
        robot,
        policy,
        prompt="task",
        control_hz=30,
        inference_horizon=10,
        fine_horizon=6,
        fine_available=True,
        initial_phase="fine",
        max_ticks=28,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    assert runtime.run() == 0
    assert policy.resets == [{"frame": 2, "phase": "coarse"}]
    assert robot.phases[:2] == ["fine"] * 2
    assert robot.phases[2:22] == ["coarse"] * 20
    assert robot.phases[22:] == ["fine"] * 6
    assert all(action is None for action in robot.received_actions[2:12])


def test_runtime_resume_discards_old_chunk_until_fresh_handoff():
    class Robot(_FakeRobot):
        def poll_control(self, *, tick):
            return RobotTickRequest(reset_policy=tick == 2)

    class Policy(_FakeRemote):
        def __init__(self):
            super().__init__()
            self.kinds = []

        def call(self, kind, **payload):
            self.kinds.append(kind)
            future = super().call(kind, **payload)
            if kind == "infer" and "reset_context" in self.kinds:
                future.result()["actions"][:, 0] += 100
            return future

    clock, robot, policy = _Clock(), Robot([]), Policy()
    client.run_episode(
        robot, policy, max_ticks=10, monotonic=clock.monotonic, sleep=clock.sleep
    )
    assert "reset_context" in policy.kinds
    assert any(request["frame"] == 2 for request in policy.requests)
    assert all(action is None for action in robot.received_actions[2:8])
    assert robot.received_actions[8][0] >= 100


@pytest.mark.parametrize(
    "continuous,command", [(False, "d"), (False, "q"), (True, "q")]
)
def test_robot_quit_and_stage1_discard_still_stop_client(continuous, command):
    robot, _, _ = _joint_robot()
    robot.continuous_episodes = continuous
    robot.console = SimpleNamespace(poll=lambda: command)
    operator = robot.poll_control(tick=0)
    assert operator.stop and not operator.end_episode
    assert operator.success is None
    assert operator.discard_episode == (command == "d")


@pytest.mark.parametrize("label", [None, "s", "f", "d"])
def test_joint_h_returns_home_opens_gripper_without_init_and_zeroes_before_resume(
    monkeypatch,
    label,
):
    import types

    import yaml

    from rlinf.envs.real.flexiv.gripper_command import LatestGripperCommand

    myq_root = Path("/home/qccc/Flexiv/flexiv_rdk/example_py")
    monkeypatch.syspath_prepend(str(myq_root))
    config = yaml.safe_load((myq_root / "myq/drag_collect.yaml").read_text())
    config["robots"]["left"]["gripper_open_width_m"] = 0.055
    config["recording"]["force_bias_after_home_seconds"] = 0
    robot, _, _ = _joint_robot()
    robot.config = config
    info = SimpleNamespace(
        q_min=np.full(7, -3), q_max=np.full(7, 3), dq_max=np.ones(7), K_x_nom=np.ones(6)
    )

    class Arm:
        def __init__(self):
            self.q = np.full(7, 0.4)

        def states(self):
            return SimpleNamespace(
                q=self.q.copy(),
                tcp_pose=np.r_[self.q[:3], 1, 0, 0, 0],
                ext_wrench_in_world=np.ones(6),
                ext_wrench_in_tcp=np.ones(6) * 2,
            )

        def info(self):
            return info

        def fault(self):
            return False

        def SendJointPosition(self, q, *args):
            self.q = np.asarray(q)

        def SendCartesianMotionForce(self, *args):
            pass

        def SwitchMode(self, mode):
            self.mode = mode

        def Stop(self):
            self.mode = "idle"

        def SetForceControlAxis(self, axes):
            pass

        def SetCartesianImpedance(self, stiffness):
            pass

    events = []
    robot.robots = {name: Arm() for name in ("left", "right")}
    sdk = types.ModuleType("flexivrdk")
    sdk.Mode = SimpleNamespace(
        NRT_JOINT_POSITION="joint", NRT_CARTESIAN_MOTION_FORCE="tcp"
    )
    monkeypatch.setitem(sys.modules, "flexivrdk", sdk)
    robot.gripper = SimpleNamespace(
        Init=lambda: events.append("init"),
        Move=lambda *args: events.append(args),
        states=lambda: SimpleNamespace(width=0.06, is_moving=False),
    )
    worker = LatestGripperCommand(robot.gripper, velocity_m_s=0.03, force_n=50)
    robot._gripper_worker = worker
    robot._home_joints = {"left": np.full(7, 0.1), "right": np.full(7, 0.2)}
    robot.admittance = SimpleNamespace(reset=lambda: None)
    robot.follower_filter = SimpleNamespace(reset=lambda pose: None)
    robot.aligner = SimpleNamespace(rebase=lambda a, b: None)
    clock = _Clock()
    monkeypatch.setattr(
        client,
        "time",
        SimpleNamespace(
            perf_counter=clock.monotonic, perf_counter_ns=lambda: int(clock.now * 1e9)
        ),
    )
    try:
        robot.continuous_episodes = label is not None
        robot.fine_available = True
        robot.console = SimpleNamespace(poll=lambda: label or " ")
        operator = robot.poll_control(tick=0)
        if label is None:
            _complete_pause(robot)
        if label is not None:
            assert operator.end_episode and not operator.stop
            assert operator.success == (None if label == "d" else label == "s")
            assert operator.discard_episode == (label == "d")
            assert not robot.poll_control(tick=0).end_episode
            robot.console = SimpleNamespace(poll=lambda: "r")
            assert not robot.poll_control(tick=0).start_episode
            assert robot._paused
        robot.console = SimpleNamespace(poll=lambda: "h")
        robot.poll_control(tick=1)
        clock.now += robot._home_return.duration_s
        receipt = robot.control_tick(
            tick=1, action=np.zeros(8), action_context=None
        ).receipt
        assert not receipt["schedule_valid"]
        assert robot._home_gripper_move.result(timeout=2) == 0.06
        assert events == [
            (
                0.06,
                config["gripper_control"]["velocity_m_s"],
                config["gripper_control"]["force_limit_n"],
            )
        ]
        for tick in (2, 3):
            assert not robot.control_tick(
                tick=tick, action=np.zeros(8), action_context=None
            ).receipt["schedule_valid"]
        assert robot.gripper_target == 0.06
        assert robot._paused and robot._maintenance is None and not robot._manual
        for name in robot.robots:
            np.testing.assert_allclose(
                robot.robots[name].states().q, robot._home_joints[name]
            )
            np.testing.assert_allclose(robot.force_bias[name]["world"], np.ones(6))
        robot.console = SimpleNamespace(poll=lambda: " ")
        operator = robot.poll_control(tick=4)
        if label is None:
            assert operator.reset_policy and not operator.start_episode
            assert operator.reset_to_coarse
        else:
            assert operator.start_episode and not operator.reset_policy
        assert not robot._paused
    finally:
        worker.close()


def test_manual_follower_pose_uses_myq_scale_and_orientation_limit(monkeypatch):
    import yaml
    from scipy.spatial.transform import Rotation

    myq_root = Path("/home/qccc/Flexiv/flexiv_rdk/example_py")
    monkeypatch.syspath_prepend(str(myq_root))
    from myq.control_alignment import (
        FollowerOrientationLimit,
        PoseLowPass,
        WorldDeltaAligner,
    )
    from myq.drag_runtime import SmoothedAdmittance

    config = yaml.safe_load((myq_root / "myq/drag_collect.yaml").read_text())
    control = config["control"]
    control["follower_position_scale"] = [2, 3, 1]
    robot, _, _ = _joint_robot()
    anchor = np.array([0, 0, 0, 1, 0, 0, 0], float)
    right = anchor.copy()
    right[:3] = [0.02, 0.01, 0]
    right[3:] = Rotation.from_euler("z", 45, degrees=True).as_quat(scalar_first=True)
    robot.target = {"right": right.copy(), "left": anchor.copy()}
    robot.robots["right"].states = lambda: SimpleNamespace(
        tcp_pose=right, ext_wrench_in_world=np.zeros(6)
    )
    robot.force_bias = {"right": {"world": np.zeros(6)}}
    robot.admittance = SmoothedAdmittance(1 / 30, control)
    robot.aligner = WorldDeltaAligner(
        "right",
        "left",
        anchor,
        anchor,
        xy_motion_direction="same",
        position_scale=control["follower_position_scale"],
    )
    robot.follower_filter = PoseLowPass(
        control["follower_target_filter_hz"], 1 / 30, anchor
    )
    robot._orientation_limit = FollowerOrientationLimit(anchor, [1.5, 1.5, 1.5])
    targets = robot.manual_targets()
    np.testing.assert_allclose(
        targets["left"][:3], np.array([0.04, 0.03, 0]) * robot.follower_filter.alpha
    )
    angles = Rotation.from_quat(targets["left"][3:], scalar_first=True).as_euler(
        "xyz", degrees=True
    )
    np.testing.assert_allclose(angles, [0, 0, 1.5], atol=1e-6)


@pytest.mark.parametrize("mode", ["joint_absolute", "eef_delta"])
def test_binary_close_holds_measured_pose_at_50n_until_stopped_then_resets(
    mode, monkeypatch
):
    from rlinf.envs.real.flexiv.gripper_command import LatestGripperCommand

    clock = _Clock()
    monkeypatch.setattr(
        client,
        "time",
        SimpleNamespace(
            perf_counter=clock.monotonic, perf_counter_ns=lambda: int(clock.now * 1e9)
        ),
    )
    robot, commands, _ = _joint_robot({"binary": True})
    robot.action_representation = mode
    robot.config["gripper_control"] = {"velocity_m_s": 0.03, "force_limit_n": 25}
    pose = np.array([0.2, 0, 0, 1, 0, 0, 0])
    for arm in robot.robots.values():
        arm.states = lambda: SimpleNamespace(q=np.full(7, 0.2), tcp_pose=pose.copy())
    robot.target = {name: np.r_[np.zeros(3), 1, 0, 0, 0] for name in robot.robots}
    tcp_commands = []
    robot._send = lambda targets: tcp_commands.append(targets["left"].copy())
    moves = []
    moved = threading.Event()

    def move(*args):
        moves.append(args)
        moved.set()

    state = SimpleNamespace(width=0.03, is_moving=True)
    robot.gripper = SimpleNamespace(Move=move, states=lambda: state)
    worker = LatestGripperCommand(robot.gripper, velocity_m_s=0.03, force_n=25)
    robot._gripper_worker = worker
    robot.console = SimpleNamespace(poll=lambda: None)
    dim = 8 if mode == "joint_absolute" else 7
    action = np.r_[np.full(dim - 1, 0.5), 0.04]
    context = {
        "training_reference": np.r_[np.zeros(3), 1, 0, 0, 0],
        "normalized_gripper_actions": np.full(20, 0.4),
        "action_start_frame": 0,
    }
    try:
        result = robot.control_tick(tick=0, action=action, action_context=context)
        assert not result.receipt["schedule_valid"]
        assert not result.receipt["human_mask"].any()
        assert robot._binary_close_move.result(timeout=2) == 0.003
        assert moves == [(0.003, 0.03, 50)]
        for tick in (1, 2):
            clock.now += 0.2
            result = robot.control_tick(
                tick=tick, action=action, action_context=context
            )
            assert not result.receipt["schedule_valid"]
            assert not robot.poll_control(tick=tick).reset_policy
        if mode == "joint_absolute":
            for command in commands:
                np.testing.assert_allclose(command[0], np.full(7, 0.2))
        else:
            for target in tcp_commands:
                np.testing.assert_allclose(target, pose)
        state.is_moving = False
        clock.now += 0.2
        result = robot.control_tick(tick=3, action=action, action_context=context)
        assert not result.receipt["schedule_valid"]
        assert robot.poll_control(tick=4).reset_policy
        resumed = robot.control_tick(tick=4, action=action, action_context=context)
        assert resumed.receipt.get("schedule_valid", True)
        assert moves == [
            (0.003, 0.03, 50)
        ]  # The same closed prediction does not restart.
        if mode == "joint_absolute":
            np.testing.assert_allclose(commands[-1][0], np.full(7, 0.5))
        else:
            assert tcp_commands[-1][0] == pytest.approx(0.5)
        moved.clear()
        context["normalized_gripper_actions"][:] = 0.8
        opened = robot.control_tick(tick=5, action=action, action_context=context)
        assert opened.receipt.get("schedule_valid", True)
        assert moved.wait(2)
        assert moves[-1] == (0.06, 0.03, 25)
    finally:
        worker.close()


def test_binary_close_timeout_and_command_error_never_resume_model(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(
        client,
        "time",
        SimpleNamespace(
            perf_counter=clock.monotonic, perf_counter_ns=lambda: int(clock.now * 1e9)
        ),
    )
    robot, commands, _ = _joint_robot({"binary": True})
    pose = np.r_[np.zeros(3), 1, 0, 0, 0]
    for arm in robot.robots.values():
        arm.states = lambda: SimpleNamespace(q=np.zeros(7), tcp_pose=pose)
    future = Future()
    robot._gripper_worker = SimpleNamespace(initialize=lambda callback: future)
    robot.console = SimpleNamespace(poll=lambda: None)
    context = {"normalized_gripper_actions": [0.4], "action_start_frame": 0}
    robot.control_tick(tick=0, action=np.r_[np.ones(7), 0.04], action_context=context)
    clock.now = 31
    with pytest.raises(RuntimeError, match="timed out"):
        robot.control_tick(tick=1, action=None, action_context=None)
    assert not robot.poll_control(tick=2).reset_policy
    clock.now = 1
    future.set_exception(RuntimeError("SDK Move failed"))
    with pytest.raises(RuntimeError, match="SDK Move failed"):
        robot.control_tick(tick=2, action=None, action_context=None)
    assert not robot.poll_control(tick=3).reset_policy
    for command in commands:
        np.testing.assert_array_equal(command[0], np.zeros(7))


@pytest.mark.parametrize("mode", ["joint_absolute", "eef_delta"])
def test_binary_close_can_continue_arm_motion_with_50n(mode):
    from rlinf.envs.real.flexiv.gripper_command import LatestGripperCommand

    robot, commands, _ = _joint_robot({"binary": True, "hold_arm_on_close": False})
    robot.action_representation = mode
    robot.target = {name: np.r_[np.zeros(3), 1, 0, 0, 0] for name in robot.robots}
    tcp_commands = []
    robot._send = lambda targets: tcp_commands.append(targets["left"].copy())
    entered, release, opened = threading.Event(), threading.Event(), threading.Event()
    moves = []

    class Gripper:
        def Move(self, width, velocity, force):
            moves.append((width, velocity, force))
            if width == 0.003:
                entered.set()
                assert release.wait(2)
            else:
                opened.set()

        def states(self):
            return SimpleNamespace(is_moving=True)

    worker = LatestGripperCommand(Gripper(), velocity_m_s=0.03, force_n=25)
    robot._gripper_worker = worker
    robot.console = SimpleNamespace(poll=lambda: None)
    dim = 8 if mode == "joint_absolute" else 7
    context = {
        "training_reference": np.r_[np.zeros(3), 1, 0, 0, 0],
        "normalized_gripper_actions": [0.4, 0.4, 0.8],
        "action_start_frame": 0,
    }
    try:
        for tick, position in ((0, 0.5), (1, 0.7), (2, 0.9)):
            action = np.r_[np.full(dim - 1, position), 0.04]
            receipt = robot.control_tick(
                tick=tick, action=action, action_context=context
            ).receipt
            assert receipt.get("schedule_valid", True)
            assert receipt["policy_proposal_valid"]
            if tick == 0:
                assert entered.wait(2)
            if mode == "joint_absolute":
                np.testing.assert_allclose(commands[-1][0], np.full(7, position))
            else:
                assert tcp_commands[-1][0] == pytest.approx(position)
            assert robot._maintenance is None
            assert not robot.poll_control(tick=tick).reset_policy
        assert moves == [(0.003, 0.03, 50)]
        release.set()
        assert opened.wait(2)
        assert moves == [(0.003, 0.03, 50), (0.06, 0.03, 25)]
    finally:
        release.set()
        worker.close()


@pytest.mark.parametrize("invalid", [None, 0, 1, "false"])
def test_binary_gripper_hold_switch_requires_boolean(invalid):
    with pytest.raises(ValueError, match="hold_arm_on_close must be a boolean"):
        _joint_robot({"binary": True, "hold_arm_on_close": invalid})


@pytest.mark.parametrize(
    "base_horizon,request_steps,origin", [(6, 0, False), (18, 12, True), (22, 18, True)]
)
@pytest.mark.parametrize("discard_first", [False, True])
def test_runtime_continuous_collection_restarts_coarse_and_waits_for_new_i(
    base_horizon, request_steps, origin, discard_first
):
    first_end = base_horizon + 14
    restart_request = first_end + 2 * base_horizon + 2
    restart_ready = restart_request + 2
    next_i = restart_ready + 8
    second_end = restart_ready + base_horizon + 16
    quit_tick = second_end + base_horizon + 2

    class Policy(_FakeRemote):
        def __init__(self):
            super().__init__()
            self.generation = 0
            self.events = []
            self.start_future = None

        def call(self, kind, **payload):
            self.events.append((robot.cycle, kind, dict(payload)))
            if kind == "start_episode":
                self.generation += 1
                self.start_future = Future()
                return self.start_future
            future = super().call(kind, **payload)
            if kind == "infer":
                steps = (
                    base_horizon
                    if payload.get("bootstrap")
                    or (payload["phase"] == "coarse" and not origin)
                    else request_steps
                    if payload["phase"] == "coarse"
                    else 6
                )
                actions = np.zeros((steps, 7), np.float32)
                actions[:, 0] = np.arange(steps) + self.generation * 100
                actions[:, 6] = 0.06
                future.result()["actions"] = actions
            return future

    class Robot(_FakeRobot):
        def __init__(self):
            super().__init__([])
            self.cycle = -1

        def poll_control(self, *, tick):
            self.cycle += 1
            if self.cycle == restart_ready:
                policy.start_future.set_result({"initial_phase": "coarse"})
            if self.cycle in (0, next_i):
                return RobotTickRequest(enter_fine_phase=True)
            if self.cycle in (first_end, second_end):
                return RobotTickRequest(
                    end_episode=True,
                    success=None
                    if discard_first and self.cycle == first_end
                    else self.cycle == first_end,
                    discard_episode=discard_first and self.cycle == first_end,
                )
            if self.cycle == restart_request:
                return RobotTickRequest(start_episode=True)
            if self.cycle == quit_tick:
                return RobotTickRequest(stop=True)
            return RobotTickRequest()

    clock, robot, policy = _Clock(), Robot(), Policy()
    runtime = RemoteRobotRuntime(
        robot,
        policy,
        prompt="task",
        control_hz=30,
        inference_horizon=base_horizon,
        request_steps=request_steps,
        coarse_origin_schedule=origin,
        fine_horizon=6,
        fine_available=True,
        initial_phase="coarse",
        max_ticks=quit_tick + 8,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    runtime.run()
    labels = [event[2] for event in policy.events if event[1] == "finish_episode"]
    assert labels == [
        {
            "frame": first_end - 1,
            "success": None if discard_first else True,
            **({"discard": True} if discard_first else {}),
        },
        {"frame": second_end - restart_ready - 1, "success": False},
    ]
    assert all(
        action is None for action in robot.received_actions[first_end:restart_ready]
    )
    assert all(action is None for action in robot.received_actions[second_end:])
    assert robot.received_actions[restart_ready][0] == 100
    for cycle, kind, payload in policy.events:
        if first_end <= cycle < restart_ready or cycle >= second_end:
            assert kind in {"finish_episode", "start_episode"} or (
                kind == "infer" and payload.get("preview")
            )
        if restart_ready <= cycle < next_i and kind == "infer":
            assert payload["phase"] == "coarse"
    bootstraps = [
        payload
        for _, kind, payload in policy.events
        if kind == "infer" and payload.get("bootstrap") and not payload.get("preview")
    ]
    assert len(bootstraps) == 2
    assert all(
        payload["frame"] == 0 and payload["phase"] == "coarse" for payload in bootstraps
    )
    assert any(
        cycle >= next_i and kind == "infer" and payload["phase"] == "fine"
        for cycle, kind, payload in policy.events
    )
    previews = [
        (cycle, payload)
        for cycle, kind, payload in policy.events
        if kind == "infer" and payload.get("preview")
    ]
    assert sum(first_end <= cycle < restart_request for cycle, _ in previews) >= 2
    assert sum(cycle >= second_end for cycle, _ in previews) >= 2
    assert robot.closed


@pytest.mark.parametrize("start_base", [False, True])
def test_remote_continuous_episodes_save_distinct_labels_and_reset_rtc(
    tmp_path, start_base
):
    from rlinf.algorithms.online_residual_rl.config import OnlineResidualRLConfig
    from rlinf.algorithms.online_residual_rl.replay import EpisodeReplay

    class Policy:
        actor_version = 0
        supports_fine_phase = True
        conditions_on_committed = True

        def __init__(self):
            self.calls = []

        def predict_base_action_batch(
            self, obs, *, horizon, committed, rtc_delay_steps
        ):
            self.calls.append(("coarse", committed))
            return torch.zeros(1, 32, 8), {"raw_model_actions": torch.zeros(1, 32, 32)}

        def predict_action_batch(self, obs, *, mode, committed, rtc_delay_steps):
            self.calls.append(("fine", committed))
            return torch.zeros(1, 12, 8), {
                "raw_model_actions": torch.zeros(1, 32, 32),
                "forward_inputs": {
                    "z_rl": torch.zeros(1, 4),
                    "proprio": torch.zeros(1, 8),
                    "base_ref": torch.zeros(1, 12, 8),
                    "actor_output": torch.zeros(1, 12, 8),
                },
            }

        def encode_receipts(self, values):
            return values

        def close(self):
            pass

    cfg = OnlineResidualRLConfig(
        action_representation="joint_absolute", z_dim=4, hidden_dim=8
    )
    replay = EpisodeReplay(cfg, storage_dir=tmp_path)
    policy = Policy()
    server = RemoteInferenceServer(
        policy,
        cfg,
        base_horizon=6,
        base_request_steps=0,
        start_base=start_base,
        commit_episode=lambda rows, success, terminal: replay.commit_episode(
            rows, success=success, terminal_frame=terminal
        ),
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    stop = threading.Event()
    thread = threading.Thread(target=server.serve, args=("127.0.0.1", port, stop))
    thread.start()
    remote = None
    try:
        for _ in range(20):
            try:
                remote = PackageRemotePolicy(
                    "127.0.0.1",
                    port,
                    "continuous-session",
                    input_shapes={"states": (8,)},
                    action_representation="joint_absolute",
                )
                break
            except OSError:
                time.sleep(0.02)
        assert remote is not None and remote.continuous_episodes
        ids = ["continuous-session"]
        request = {"observation": {"states": np.zeros(8)}, "prompt": "task"}
        for index, label in enumerate((True, None, False)):
            if index:
                started = remote.call("start_episode").result(timeout=2)
                assert started["initial_phase"] == "coarse"
                ids.append(started["episode_id"])
            phase = remote.initial_phase if index == 0 else "coarse"
            remote.call(
                "infer",
                frame=0,
                bootstrap=True,
                phase=phase,
                target_frame=0,
                **request,
            ).result(timeout=2)
            assert policy.calls[-1][0] == phase and policy.calls[-1][1] is None
            for frame in range(30):
                if frame % 6 == 0:
                    payload = {
                        "frame": frame,
                        "target_frame": frame + 6,
                        "phase": phase if frame == 0 else "fine",
                        **request,
                    }
                    if frame:
                        payload["accepted"] = {
                            "frame": frame - 6,
                            "accept_frame": frame,
                            "accept_tick": frame,
                            "model_version": 0,
                        }
                    remote.call("infer", **payload).result(timeout=2)
                remote.call(
                    "receipt",
                    frame=frame,
                    info={
                        "action_representation": "joint_absolute",
                        "action_receipt_physical": np.zeros(8),
                        "receipt_valid": True,
                        "schedule_valid": True,
                    },
                ).result(timeout=2)
            count_before_finish = len(replay)
            remote.call("reset_context", frame=30, phase="coarse").result(timeout=2)
            resumed = remote.call(
                "infer",
                frame=30,
                target_frame=30,
                bootstrap=True,
                phase="coarse",
                **request,
            ).result(timeout=2)
            assert resumed["phase"] == "coarse"
            assert policy.calls[-1][0] == "coarse" and policy.calls[-1][1] is None
            remote.call(
                "finish_episode", frame=29, success=label, discard=label is None
            ).result(timeout=2)
            if label is None:
                assert len(replay) == count_before_finish
                assert not replay.store.path_for(ids[index]).exists()
                assert replay.store.load_episode(ids[0])["success"] is True
            else:
                saved = replay.store.load_episode(ids[index])
                assert saved["success"] is label
                assert len(saved["transitions"]) > 0
                assert {
                    value["episode_id"] for value in saved["features"].values()
                } == {ids[index]}
            count = len(replay)
            for preview_frame in (30, 36):
                remote.call(
                    "infer",
                    frame=preview_frame,
                    target_frame=preview_frame,
                    bootstrap=True,
                    preview=True,
                    phase="coarse",
                    **request,
                ).result(timeout=2)
                assert policy.calls[-1][0] == "coarse"
            assert len(replay) == count
            if label is not None:
                assert replay.store.load_episode(ids[index])["success"] is label
        assert len(set(ids)) == 3
        assert len(list(tmp_path.glob("*.pt"))) == 2
        assert all(
            row.feature.episode_id == row.next_feature.episode_id
            for row in replay.transitions()
            if row.next_feature is not None
        )
        remote.close(29, None)
    finally:
        if remote is not None:
            remote.close(-1, None)
        stop.set()
        thread.join(timeout=3)
        server.close()
    assert not thread.is_alive()


@pytest.mark.parametrize("label", ["s", "f"])
def test_stage2_space_starts_next_episode_without_home(label):
    robot, _, _ = _joint_robot()
    robot.continuous_episodes = True
    restored = []
    robot._restore_policy_control = lambda: restored.append(True)
    robot.console = SimpleNamespace(poll=lambda: label)
    ended = robot.poll_control(tick=10)
    assert ended.end_episode and not ended.stop
    assert ended.success == (label == "s")
    assert robot._paused
    robot.console = SimpleNamespace(poll=lambda: " ")
    started = robot.poll_control(tick=11)
    assert started.start_episode and not started.stop
    assert not robot._paused and not robot._episode_ended
    assert len(restored) == 2


@pytest.mark.parametrize("delay_at", ["start_reply", "bootstrap_reply"])
def test_i_during_episode_start_survives_space_r_manual_takeover(delay_at, caplog):
    class Policy(_FakeRemote):
        def __init__(self):
            super().__init__()
            self.generation = 0
            self.start_reply = Future()
            self.bootstrap_reply = Future()
            self.bootstrap_result = None

        def call(self, kind, **payload):
            if kind == "start_episode":
                self.generation += 1
                if delay_at == "start_reply":
                    return self.start_reply
                future = Future()
                future.set_result({"initial_phase": "coarse"})
                return future
            future = super().call(kind, **payload)
            if (
                kind == "infer"
                and payload.get("bootstrap")
                and self.generation == 1
                and delay_at == "bootstrap_reply"
            ):
                self.bootstrap_result = future.result()
                return self.bootstrap_reply
            return future

    class Robot(_FakeRobot):
        def __init__(self):
            super().__init__([])
            self.cycle = -1

        def poll_control(self, *, tick):
            self.cycle += 1
            if self.cycle == 28:
                if delay_at == "start_reply":
                    policy.start_reply.set_result({"initial_phase": "coarse"})
                else:
                    policy.bootstrap_reply.set_result(policy.bootstrap_result)
            if self.cycle == 20:
                return RobotTickRequest(end_episode=True, success=True)
            if self.cycle == 22:
                return RobotTickRequest(start_episode=True)
            if self.cycle == 24:
                return RobotTickRequest(enter_fine_phase=True)
            if self.cycle == 25:
                # Space keeps the startup hold; background inference stays enabled.
                return RobotTickRequest()
            if self.cycle == 26:
                self.take_over()  # R completes force zeroing in this world fake.
            if self.cycle == 44:
                return RobotTickRequest(end_episode=True, success=True)
            if self.cycle == 48:
                return RobotTickRequest(stop=True)
            return RobotTickRequest()

    caplog.set_level(logging.INFO, logger="flexiv.status")
    clock, robot, policy = _Clock(), Robot(), Policy()
    runtime = RemoteRobotRuntime(
        robot,
        policy,
        prompt="task",
        control_hz=30,
        inference_horizon=6,
        request_steps=0,
        fine_available=True,
        initial_phase="coarse",
        max_ticks=52,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    runtime.run()
    assert "fine phase request queued" in caplog.text
    assert "queued fine phase request applied" in caplog.text
    assert "robot entered fine phase" in caplog.text
    assert any(request["phase"] == "fine" for request in policy.requests)
    assert any(accepted["frame"] == 6 for accepted in policy.accepted)
    assert robot.took_over


def test_i_while_paused_reports_why_it_was_not_applied(caplog):
    robot, _, _ = _joint_robot()
    robot.fine_available = True
    robot._paused = True
    robot.console = SimpleNamespace(poll=lambda: "i")
    assert not robot.poll_control(tick=0).enter_fine_phase
    assert "I was not applied" in caplog.text
