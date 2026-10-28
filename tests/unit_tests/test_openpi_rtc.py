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

"""Contract tests for standard PI0.5 RTC scheduling and execution."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rlinf.envs.real.flexiv.rtc_adapter import (
    FlexivRTCAdapter,
    FlexivRTCConfig,
    compose_tcp_target,
    tcp_delta_from_target,
)
from rlinf.models.embodiment.openpi.rtc.policy import PI05PolicyAdapter, PI05PolicySpec
from rlinf.models.embodiment.openpi.rtc.runtime import (
    RTCRobotRuntime,
    RuntimeConfig,
)
from rlinf.models.embodiment.openpi.rtc.scheduler import RTCChunkScheduler
from rlinf.models.embodiment.openpi.rtc.types import (
    InferenceRequest,
    PhysicalActionStep,
    PolicyObservation,
    PolicyResult,
    RTCGuidance,
)


def _observation(tick: int = 0) -> PolicyObservation:
    return PolicyObservation.snapshot(
        np.full(8, tick, dtype=np.float32),
        {
            "main": np.zeros((2, 2, 3), dtype=np.uint8),
            "wrist": np.zeros((2, 2, 3), dtype=np.uint8),
        },
        "test",
    )


class _ControlledPolicy:
    def __init__(self, horizon: int = 4) -> None:
        self.horizon = horizon
        self.requests: list[InferenceRequest] = []
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, request: InferenceRequest) -> PolicyResult:
        self.requests.append(request)
        call_index = len(self.requests) - 1
        if call_index:
            self.started.set()
            if not self.release.wait(timeout=2.0):
                raise TimeoutError("Test inference was not released.")
        raw = (
            np.arange(self.horizon * 7, dtype=np.float32).reshape(self.horizon, 7)
            + call_index * 100
        )
        physical = np.zeros((self.horizon, 7), dtype=np.float32)
        physical[:, 0] = call_index * 10 + np.arange(self.horizon)
        return PolicyResult(raw, physical, 0.01, request)


def _scheduler(policy: _ControlledPolicy) -> RTCChunkScheduler:
    return RTCChunkScheduler(
        policy,
        action_horizon=4,
        execution_horizon=2,
        initial_delay=1,
        delay_history_length=4,
        num_steps=6,
    )


def _finish(policy: _ControlledPolicy, scheduler: RTCChunkScheduler, *, tick: int):
    policy.release.set()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        event = scheduler.accept_ready(accept_tick=tick)
        if event is not None:
            return event
        time.sleep(0.001)
    raise AssertionError("Inference did not finish.")


def test_rtc_first_chunk_is_unguided_and_second_uses_raw_tail() -> None:
    policy = _ControlledPolicy()
    scheduler = _scheduler(policy)
    try:
        scheduler.bootstrap(_observation())
        assert policy.requests[0].guidance is None
        for _ in range(2):
            scheduler.next_step()
            scheduler.commit_tick()
        assert scheduler.submit_if_due(_observation(2), observation_tick=2)
        assert policy.started.wait(timeout=2.0)
        request = policy.requests[1]
        assert request.guidance is not None
        expected_raw = np.arange(28, dtype=np.float32).reshape(4, 7)
        np.testing.assert_array_equal(
            request.guidance.previous_normalized_actions[:2], expected_raw[2:]
        )
        assert request.observation_tick == request.submit_tick == 2
        assert request.submit_committed_tick == request.old_chunk_cursor == 2
        assert request.guidance.prefix_len == 1
        assert request.guidance.decay_end == 3
    finally:
        policy.release.set()
        scheduler.close()


@pytest.mark.parametrize("committed_delay", [0, 1, 3])
def test_rtc_stale_prefix_uses_committed_ticks(committed_delay: int) -> None:
    policy = _ControlledPolicy()
    scheduler = _scheduler(policy)
    try:
        scheduler.bootstrap(_observation())
        for _ in range(2):
            scheduler.next_step()
            scheduler.commit_tick()
        scheduler.submit_if_due(_observation(2), observation_tick=2)
        assert policy.started.wait(timeout=2.0)
        for _ in range(committed_delay):
            scheduler.next_step()
            scheduler.commit_tick()
        event = _finish(policy, scheduler, tick=2 + committed_delay)
        assert event.accepted
        assert event.stale_prefix == committed_delay
        assert scheduler.cursor == committed_delay
    finally:
        policy.release.set()
        scheduler.close()


def test_rtc_handoff_does_not_duplicate_delta_and_limits_inflight() -> None:
    policy = _ControlledPolicy()
    scheduler = _scheduler(policy)
    try:
        scheduler.bootstrap(_observation())
        executed = []
        for _ in range(2):
            executed.append(scheduler.next_step().tcp_delta[0])
            scheduler.commit_tick()
        assert scheduler.submit_if_due(_observation(2), observation_tick=2)
        assert not scheduler.submit_if_due(_observation(2), observation_tick=2)
        assert policy.started.wait(timeout=2.0)
        executed.append(scheduler.next_step().tcp_delta[0])
        scheduler.commit_tick()
        _finish(policy, scheduler, tick=3)
        executed.append(scheduler.next_step().tcp_delta[0])
        assert executed == [0, 1, 2, 11]
        assert len(policy.requests) == 2
    finally:
        policy.release.set()
        scheduler.close()


def test_rtc_ready_result_is_selected_in_the_same_control_tick() -> None:
    policy = _ControlledPolicy()
    scheduler = _scheduler(policy)
    try:
        scheduler.bootstrap(_observation())
        for _ in range(2):
            scheduler.next_step()
            scheduler.commit_tick()
        scheduler.submit_if_due(_observation(2), observation_tick=2)
        assert policy.started.wait(timeout=2.0)
        event = _finish(policy, scheduler, tick=2)
        assert event.stale_prefix == 0
        assert scheduler.next_step().tcp_delta[0] == 10
    finally:
        policy.release.set()
        scheduler.close()


def test_rtc_reset_rejects_late_result_and_exhaustion_never_replays() -> None:
    policy = _ControlledPolicy()
    scheduler = _scheduler(policy)
    try:
        scheduler.bootstrap(_observation())
        for _ in range(2):
            scheduler.next_step()
            scheduler.commit_tick()
        scheduler.submit_if_due(_observation(2), observation_tick=2)
        assert policy.started.wait(timeout=2.0)
        scheduler.reset(session_id=4)
        event = _finish(policy, scheduler, tick=2)
        assert not event.accepted
        assert event.reason == "stale generation"
        assert scheduler.next_step() is None
        assert scheduler.next_step() is None
    finally:
        policy.release.set()
        scheduler.close()


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleep_deadlines: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        self.sleep_deadlines.append(self.now + duration)
        self.now += duration


class _ImmediatePolicy:
    def __call__(self, request: InferenceRequest) -> PolicyResult:
        raw = np.zeros((4, 7), dtype=np.float32)
        physical = np.zeros((4, 7), dtype=np.float32)
        return PolicyResult(raw, physical, 0.0, request)


class _OpenPIPolicy:
    def __init__(self) -> None:
        self.sample_kwargs = None

    def infer(self, obs, *, sample_kwargs=None, return_raw_actions=False):
        assert set(obs) == {"observation/state", "main", "wrist", "prompt"}
        assert return_raw_actions
        self.sample_kwargs = sample_kwargs
        return {
            "raw_actions": np.full((4, 32), 2.0, dtype=np.float32),
            "actions": np.full((4, 7), 3.0, dtype=np.float32),
        }


def test_policy_adapter_keeps_raw_guidance_and_physical_output_separate() -> None:
    policy = _OpenPIPolicy()
    adapter = PI05PolicyAdapter(
        policy,
        PI05PolicySpec(
            config_name="test",
            checkpoint_path="/unused",
            policy_camera_keys=("main", "wrist"),
            action_horizon=4,
        ),
    )
    guidance = RTCGuidance(np.ones((4, 7)), 1, 5.0, 3)
    request = InferenceRequest(0, 0, 1, 2, 2, 2, 2, _observation(2), guidance, 6)
    result = adapter.infer(request)
    np.testing.assert_array_equal(result.normalized_raw_chunk, 2.0)
    np.testing.assert_array_equal(result.physical_delta_chunk, 3.0)
    np.testing.assert_array_equal(policy.sample_kwargs["prev_actions"], 1.0)
    assert policy.sample_kwargs["prefix_len"].dtype == np.int32
    assert policy.sample_kwargs["rtc_guidance_weight"].dtype == np.float32


class _FakeRobot:
    def __init__(self) -> None:
        self.applied = 0
        self.closed = 0

    def start(self) -> None:
        pass

    def read_observation(self, *, tick: int, prompt: str) -> PolicyObservation:
        del prompt
        return _observation(tick)

    def apply_policy_action(self, step) -> None:
        del step
        self.applied += 1

    def hold_target(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def close(self) -> None:
        self.closed += 1


def test_rtc_runtime_uses_absolute_deadlines_and_idempotent_close() -> None:
    clock = _Clock()
    robot = _FakeRobot()
    scheduler = RTCChunkScheduler(
        _ImmediatePolicy(), action_horizon=4, execution_horizon=2
    )
    runtime = RTCRobotRuntime(
        robot,
        scheduler,
        RuntimeConfig(prompt="test", control_hz=10.0, max_ticks=3),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    assert runtime.run() == 0
    runtime.close()
    assert clock.sleep_deadlines == pytest.approx([0.1, 0.2, 0.3])
    assert robot.applied == 3
    assert robot.closed == 1


class _OrderScheduler:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def bootstrap(self, observation, *, observation_tick):
        del observation, observation_tick
        self.events.append("bootstrap")

    def accept_ready(self, *, accept_tick):
        self.events.append(f"accept{accept_tick}")

    def submit_if_due(self, observation, *, observation_tick):
        del observation
        self.events.append(f"submit{observation_tick}")

    def next_step(self):
        self.events.append("select")
        return PhysicalActionStep(np.zeros(6), 1.0, 0, 0)

    def commit_tick(self):
        self.events.append("commit")

    def stop_accepting(self):
        pass

    def close(self, *, wait=True):
        del wait


class _OrderRobot(_FakeRobot):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    def read_observation(self, *, tick: int, prompt: str) -> PolicyObservation:
        del prompt
        self.events.append(f"read{tick}")
        return _observation(tick)

    def apply_policy_action(self, step) -> None:
        del step
        self.events.append("send")


def test_rtc_tick_polls_result_around_observation_and_submission() -> None:
    events: list[str] = []
    clock = _Clock()
    runtime = RTCRobotRuntime(
        _OrderRobot(events),
        _OrderScheduler(events),
        RuntimeConfig(prompt="test", control_hz=10.0, max_ticks=2),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    runtime.run()
    assert events == [
        "read0",
        "bootstrap",
        "accept0",
        "accept0",
        "submit0",
        "accept0",
        "select",
        "send",
        "commit",
        "accept1",
        "read1",
        "accept1",
        "submit1",
        "accept1",
        "select",
        "send",
        "commit",
    ]


def _flexiv_config(**overrides) -> FlexivRTCConfig:
    values = {
        "env": {"is_dummy": True},
        "camera_key_map": {"main": "main", "wrist": "wrist"},
        "translation_unit": "m",
        "rotation_representation": "rotvec",
        "rotation_unit": "rad",
        "tcp_delta_frame": "world",
        "translation_delta_limit_m": 0.01,
        "rotation_delta_limit_rad": 0.1,
        "gripper_state_range": (0.0, 1.0),
        "gripper_command_range": (0.0, 1.0),
        "gripper_state_open_at_max": True,
        "gripper_command_open_at_max": True,
    }
    values.update(overrides)
    return FlexivRTCConfig(**values)


def test_flexiv_delta_overwrites_target_from_one_chunk_reference() -> None:
    config = _flexiv_config()
    initial = np.array([0.5, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0])
    delta = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.2])
    first = compose_tcp_target(initial, delta, config)
    second = compose_tcp_target(initial, delta, config)
    np.testing.assert_allclose(first[:3], [0.51, 0.0, 0.25])
    np.testing.assert_allclose(second, first)
    np.testing.assert_allclose(
        Rotation.from_quat([first[4], first[5], first[6], first[3]]).as_rotvec(),
        [0.0, 0.0, 0.1],
    )


def test_flexiv_world_delta_round_trips_nonidentity_tcp_rotation() -> None:
    config = _flexiv_config(enable_tcp_delta_limits=False)
    quat = Rotation.from_euler("z", 90, degrees=True).as_quat()
    reference = np.array([0.5, 0.0, 0.25, quat[3], *quat[:3]])
    delta = np.array([0.02, -0.01, 0.003, 0.01, -0.02, 0.03])
    target = compose_tcp_target(reference, delta, config)
    np.testing.assert_allclose(
        tcp_delta_from_target(reference, target, config), delta, atol=1e-7
    )


def test_flexiv_gripper_is_independent_and_close_is_idempotent() -> None:
    adapter = FlexivRTCAdapter(_flexiv_config())
    reference = np.array([0.5, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0])
    delta = np.array([0.005, 0.0, 0.0, 0.0, 0.0, 0.0])
    adapter.start()
    try:
        adapter.apply_policy_action(PhysicalActionStep(delta, 0.0, 0, 0, reference))
        first = adapter.read_observation(tick=0, prompt="test")
        adapter.apply_policy_action(PhysicalActionStep(delta, 1.0, 0, 1, reference))
        second = adapter.read_observation(tick=1, prompt="test")
        assert first.state[-1] == 0.0
        assert second.state[-1] == 1.0
    finally:
        adapter.close()
        adapter.close()


def test_flexiv_committed_projection_matches_physical_delta_limits() -> None:
    adapter = FlexivRTCAdapter(_flexiv_config())
    projected = adapter.project_policy_actions(
        np.array([[0.02, 0.0, 0.0, 0.0, 0.0, 0.2, 1.5]])
    )
    np.testing.assert_allclose(projected[0], [0.01, 0, 0, 0, 0, 0.1, 1.0])


def test_flexiv_realtime_receipt_is_the_final_accepted_delta() -> None:
    adapter = FlexivRTCAdapter(_flexiv_config())
    reference = np.array([0.5, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0])
    adapter.start()
    try:
        receipt = adapter.apply_policy_action(
            PhysicalActionStep(np.array([0.02, 0, 0, 0, 0, 0.2]), 1.0, 0, 0, reference)
        )
        np.testing.assert_allclose(
            receipt["action_receipt_physical"], [0.01, 0, 0, 0, 0, 0.1, 1]
        )
        assert not receipt["human_mask"].any()
        np.testing.assert_allclose(receipt["tcp_target_pose"][:3], [0.51, 0, 0.25])
        np.testing.assert_allclose(adapter.read_tcp_pose()[:3], [0.51, 0, 0.25])
    finally:
        adapter.close()


def test_flexiv_realtime_intervention_receipt_uses_chunk_reference() -> None:
    config = _flexiv_config(enable_tcp_delta_limits=False)
    adapter = FlexivRTCAdapter(config)
    reference = np.array([0.5, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0])

    class Override:
        def read(self, env, command):
            del env, command
            mask = np.array([True] * 6 + [False])
            return type(
                "Sample",
                (),
                {
                    "active": True,
                    "action": np.array([0.02, 0, 0, 0, 0, 0, 1.0]),
                    "info": {"teleop_action_mask": mask},
                },
            )()

        def close(self):
            pass

    adapter.start()
    try:
        adapter.apply_policy_action(
            PhysicalActionStep(np.array([0.01, 0, 0, 0, 0, 0]), 1.0, 0, 0, reference)
        )
        adapter.read_tcp_pose()
        adapter._intervention = Override()
        receipt = adapter.apply_policy_action(
            PhysicalActionStep(np.array([0.015, 0, 0, 0, 0, 0]), 1.0, 0, 1, reference)
        )
        np.testing.assert_allclose(adapter.read_tcp_pose()[:3], [0.53, 0, 0.25])
        np.testing.assert_allclose(
            receipt["action_receipt_physical"][:6], [0.03, 0, 0, 0, 0, 0]
        )
        assert np.flatnonzero(receipt["human_mask"]).tolist() == [0]
        np.testing.assert_array_equal(receipt["pair_mask"], receipt["human_mask"])
        np.testing.assert_allclose(
            tcp_delta_from_target(reference, adapter.read_tcp_pose(), config),
            receipt["action_receipt_physical"][:6],
        )
    finally:
        adapter.close()


def test_flexiv_stop_closes_leader_before_executor_shutdown(monkeypatch) -> None:
    from rlinf.envs.real.flexiv.base import FlexivEnv
    from rlinf.envs.real.flexiv.leader_drag import LeaderDragInterventionSource

    events = []

    class Source:
        config = type("Config", (), {"control_hz": 30.0})()

        def start(self):
            pass

        def set_enabled(self, enabled):
            del enabled

        def poll(self, now_ns):
            del now_ns
            return None

        def close(self):
            events.append("leader-stop")

    source = Source()
    monkeypatch.setattr(
        LeaderDragInterventionSource,
        "connect",
        lambda *args, **kwargs: source,
    )
    monkeypatch.setattr(
        FlexivEnv, "stop_realtime", lambda self: events.append("follower-stop")
    )
    adapter = FlexivRTCAdapter(_flexiv_config(leader_serial="leader"))
    adapter.start()
    adapter.stop()
    adapter.stop()
    adapter.close()
    assert events == ["follower-stop", "leader-stop"]
