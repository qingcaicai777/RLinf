# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""SDK-independent fixed-rate runtime for split robot inference."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class RobotObservation:
    """Policy-ready inputs and opaque action context from one sensor snapshot."""

    policy_inputs: Mapping[str, Any]
    action_context: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class RobotControlResult:
    """One adapter tick's receipt and runtime control signals."""

    receipt: Mapping[str, Any]
    policy_enabled: bool = True
    stop: bool = False
    success: bool | None = None


@dataclass(frozen=True)
class RobotTickRequest:
    """Operator state sampled before a tick's inference handoff."""

    policy_enabled: bool = True
    stop: bool = False
    success: bool | None = None
    enter_fine_phase: bool = False
    discard_episode: bool = False
    reset_policy: bool = False
    reset_to_coarse: bool = False
    end_episode: bool = False
    start_episode: bool = False


class RobotAdapter(Protocol):
    """Robot operations consumed by the fixed-rate inference runtime."""

    def start(self) -> None: ...

    def read_observation(self, *, tick: int, prompt: str) -> RobotObservation | None:
        """Return mapped policy inputs and action context, or None while warming up."""
        ...

    def poll_control(self, *, tick: int) -> RobotTickRequest:
        """Read operator controls before accepting or submitting inference."""
        ...

    def control_tick(
        self,
        *,
        tick: int,
        action: np.ndarray | None,
        action_context: Mapping[str, Any] | None,
    ) -> RobotControlResult:
        """Apply an action or hold the current target and return its receipt."""
        ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


class RemotePolicy(Protocol):
    """Ordered asynchronous request channel to the policy-owning process."""

    session: str

    def call(self, kind: str, **payload: Any) -> Future: ...

    def completion_upper_bound(self, server_monotonic: float) -> float | None: ...

    def close(self, frame: int, success: bool | None) -> None: ...


class RemoteRobotRuntime:
    """Run fixed-size action blocks against any adapter implementing the protocol."""

    def __init__(
        self,
        robot: RobotAdapter,
        policy: RemotePolicy,
        *,
        prompt: str,
        control_hz: float,
        inference_horizon: int = 6,
        request_steps: int | None = None,
        coarse_origin_schedule: bool = False,
        fine_horizon: int | None = None,
        fine_available: bool = False,
        initial_phase: str = "fine",
        max_ticks: int | None = None,
        task_timeout_seconds: float | None = None,
        handoff_grace_ticks: int | None = None,
        monotonic=time.perf_counter,
        sleep=time.sleep,
    ) -> None:
        if not prompt or control_hz <= 0 or inference_horizon <= 0:
            raise ValueError(
                "Prompt, control_hz and inference_horizon must be positive."
            )
        fine_horizon = inference_horizon if fine_horizon is None else fine_horizon
        request_steps = 0 if request_steps is None else request_steps
        if fine_horizon <= 0 or initial_phase not in {"coarse", "fine"}:
            raise ValueError("Fine horizon and initial phase are invalid.")
        if not 0 <= request_steps < inference_horizon:
            raise ValueError("request_steps must be in [0, inference_horizon).")
        if coarse_origin_schedule and 2 * request_steps < inference_horizon:
            raise ValueError(
                "A coarse prediction must reach its request index at or after handoff."
            )
        if handoff_grace_ticks is not None and (
            handoff_grace_ticks < 0
            or handoff_grace_ticks >= max(inference_horizon, fine_horizon)
        ):
            raise ValueError("handoff_grace_ticks must be smaller than the horizon.")
        if (
            initial_phase == "coarse"
            and fine_available
            and inference_horizon < fine_horizon
        ):
            raise ValueError("Base block cannot provide a complete fine-phase bridge.")
        if max_ticks is not None and max_ticks <= 0:
            raise ValueError("max_ticks must be positive.")
        if task_timeout_seconds is not None and (
            not np.isfinite(task_timeout_seconds) or task_timeout_seconds <= 0
        ):
            raise ValueError("task_timeout_seconds must be positive and finite.")
        self.task_timeout_seconds = task_timeout_seconds
        self.robot = robot
        self.policy = policy
        self.prompt = prompt
        self.period = 1.0 / control_hz
        self.base_horizon = inference_horizon
        self.request_steps = request_steps
        self.coarse_origin_schedule = coarse_origin_schedule
        self.fine_horizon = fine_horizon
        self.fine_available = fine_available
        self._phase = initial_phase
        self.horizon = inference_horizon if initial_phase == "coarse" else fine_horizon
        self.max_ticks = max_ticks
        self.handoff_grace_ticks = handoff_grace_ticks
        self._monotonic = monotonic
        self._sleep = sleep
        self._episode_active = True
        self._episode_end_future: Future | None = None
        self._episode_start_future: Future | None = None
        self._next_episode_bootstrap = False
        self._preview_future: Future | None = None
        self._preview_next_tick = 0
        self._total_ticks = 0
        self._frame = 0
        self._chunk_start = 0
        self._coarse_origin = 0
        self._request_frame = 0
        self._submitted_tick = 0
        self._request_target_frame = 0
        self._accepted_pending: dict[str, int] | None = None
        self._fine_requested = False
        self._fine_requested_during_start = False
        self._success: bool | None = None
        self._discard_episode = False
        self._future: Future | None = None
        self._submitted_at: float | None = None
        self._actions: np.ndarray | None = None
        self._action_context: Any = None
        self._waiting_boundary: int | None = None
        self._waiting_observation: RobotObservation | None = None
        self._holding_for_inference = False
        self._recovery_request = False
        self._next_hold_warning_frame: int | None = None
        self._bootstrap_observation: RobotObservation | None = None
        self._policy_enabled = True

    def _reset_episode_schedule(self) -> None:
        """Begin the next episode with an empty coarse inference context."""
        self._phase = "coarse"
        self.horizon = self.base_horizon
        self._frame = self._chunk_start = self._coarse_origin = 0
        self._request_frame = self._submitted_tick = self._request_target_frame = 0
        self._fine_requested = False
        self._success = None
        self._discard_episode = False
        self._future = None
        self._accepted_pending = None
        self._actions = None
        self._action_context = None
        self._waiting_boundary = None
        self._waiting_observation = None
        self._bootstrap_observation = None
        self._holding_for_inference = False
        self._recovery_request = False
        self._next_hold_warning_frame = None
        self._policy_enabled = True

    def _preview_while_paused(self) -> None:
        """Keep coarse inference running without recording or executing a proposal."""
        if self._preview_future is not None:
            if not self._preview_future.done():
                return
            result = self._preview_future.result()
            self._parse_block(result, horizon=self.base_horizon)
            self._preview_future = None
        if self._total_ticks < self._preview_next_tick:
            return
        observation = self.robot.read_observation(tick=self._frame, prompt=self.prompt)
        if observation is None:
            raise RuntimeError("Observation missing during paused inference.")
        self._preview_future = self.policy.call(
            "infer",
            frame=self._frame,
            target_frame=self._frame,
            bootstrap=True,
            preview=True,
            phase="coarse",
            observation=dict(observation.policy_inputs),
            action_context=None
            if observation.action_context is None
            else dict(observation.action_context),
            prompt=self.prompt,
        )
        interval = (
            self.request_steps if self.coarse_origin_schedule else self.base_horizon
        )
        self._preview_next_tick = self._total_ticks + max(1, interval)

    def _hold_warning_interval_ticks(self) -> int:
        """Return how often to report a handoff that remains pending."""
        if self.handoff_grace_ticks is None:
            return max(1, self.horizon - 1)
        return max(1, self.handoff_grace_ticks)

    def _request_budget_ms(self) -> float:
        """Return the time from this request to its planned handoff."""
        return (
            (self._request_target_frame - self._submitted_tick) * self.period * 1000.0
        )

    def _next_boundary(self) -> int:
        """Return the next planned handoff tick."""
        if self._phase == "coarse" and self.coarse_origin_schedule:
            return self._coarse_origin + self.base_horizon
        return self._chunk_start + self.horizon

    def _submit(
        self,
        tick: int,
        observation: RobotObservation,
        accepted: Mapping[str, int] | None,
        *,
        bootstrap: bool = False,
        warmup: bool = False,
    ) -> None:
        self._submitted_at = self._monotonic()
        self._future = self.policy.call(
            "infer",
            frame=tick,
            bootstrap=bootstrap,
            warmup=warmup,
            phase=self._phase,
            target_frame=(tick if bootstrap else self._next_boundary()),
            accepted=accepted if accepted is not None else self._accepted_pending,
            observation=dict(observation.policy_inputs),
            action_context=(
                None
                if observation.action_context is None
                else dict(observation.action_context)
            ),
            prompt=self.prompt,
        )
        self._request_frame = tick
        self._submitted_tick = self._frame
        self._request_target_frame = tick if bootstrap else self._next_boundary()
        if accepted is not None or self._accepted_pending is not None:
            self._accepted_pending = None

    def _parse_block(self, result: Mapping[str, Any], *, horizon: int) -> np.ndarray:
        actions = np.asarray(result["actions"], dtype=np.float32)
        representation = getattr(self.robot, "action_representation", None)
        expected_dim = (
            8
            if representation == "joint_absolute"
            else 7
            if representation == "eef_delta"
            else None
        )
        if (
            actions.ndim != 2
            or actions.shape[0] != horizon
            or actions.shape[1] == 0
            or (expected_dim is not None and actions.shape[1] != expected_dim)
            or not np.isfinite(actions).all()
        ):
            raise ValueError(f"Policy block must be finite [{horizon}, action_dim].")
        return actions

    def run(self) -> int:
        """Run control ticks and consecutive Stage2 episodes until stop or tick limit."""
        try:
            self.robot.start()
            prompt_deadline = self._monotonic() + 10.0
            while True:
                observation = self.robot.read_observation(tick=0, prompt=self.prompt)
                if observation is not None:
                    break
                if self._monotonic() >= prompt_deadline:
                    raise TimeoutError("Initial robot observation was not ready.")
                self._sleep(0.01)
            self._submit(0, observation, None, bootstrap=True)
            result = self._future.result(timeout=30.0)
            if result.get("phase", self._phase) != self._phase:
                raise ValueError("Bootstrap response phase does not match the robot.")
            self._actions = self._parse_block(result, horizon=self.horizon)
            active_request_frame = 0
            self._action_context = result.get("action_context")
            self._future = None
            self._bootstrap_observation = observation

            # Match OpenPI's continuation warmup: the first request that
            # includes committed actions can trigger a distinct compiled or
            # CUDA-graph path.  Warm it before the robot's fixed-rate loop so
            # compilation latency cannot consume the first handoff budget.
            if getattr(self.policy, "rtc_enabled", False):
                self._submit(0, observation, None, bootstrap=True, warmup=True)
                self._future.result(timeout=30.0)
                self._future = None
            deadline = self._monotonic()
            episode_started = deadline
            previous_joint_target = None
            previous_joint_step = None

            while self.max_ticks is None or self._total_ticks < self.max_ticks:
                if (
                    self._episode_active
                    and self.task_timeout_seconds is not None
                    and self._monotonic() - episode_started >= self.task_timeout_seconds
                ):
                    self._success = False
                    logging.warning(
                        "Task timed out after %.2f seconds", self.task_timeout_seconds
                    )
                    break
                deadline += self.period
                ready_at_tick_start = self._future is not None and self._future.done()
                operator = self.robot.poll_control(tick=self._frame)
                if operator.stop:
                    self._success = operator.success
                    self._discard_episode = operator.discard_episode
                    break
                if operator.end_episode:
                    if (
                        not self._episode_active
                        and self._episode_start_future is None
                        and not self._next_episode_bootstrap
                    ):
                        raise RuntimeError("No active episode to label.")
                    finish_options = (
                        {"discard": True} if operator.discard_episode else {}
                    )
                    self._episode_end_future = self.policy.call(
                        "finish_episode",
                        frame=self._frame - 1 if self._episode_active else -1,
                        success=operator.success,
                        **finish_options,
                    )
                    self._episode_active = False
                    self._fine_requested_during_start = False
                    self._episode_start_future = None
                    self._next_episode_bootstrap = False
                    self._future = None
                    self._actions = None
                    self._accepted_pending = None
                    self._action_context = None
                if (
                    self._episode_end_future is not None
                    and self._episode_end_future.done()
                ):
                    self._episode_end_future.result()
                    self._episode_end_future = None
                    self._preview_next_tick = self._total_ticks
                if operator.start_episode:
                    if (
                        self._episode_active
                        or self._episode_start_future is not None
                        or self._next_episode_bootstrap
                    ):
                        raise RuntimeError("An episode is already active or starting.")
                    self._fine_requested_during_start = False
                    self._episode_start_future = self.policy.call("start_episode")
                if (
                    operator.enter_fine_phase
                    and not self._episode_active
                    and (
                        self._episode_start_future is not None
                        or self._next_episode_bootstrap
                    )
                ):
                    if not self.fine_available:
                        raise RuntimeError(
                            "This inference server has no fine-phase Actor."
                        )
                    self._fine_requested_during_start = True
                    logging.getLogger("flexiv.status").info(
                        "fine phase request queued while episode bootstrap is pending"
                    )
                if (
                    self._episode_start_future is not None
                    and self._episode_start_future.done()
                ):
                    started = self._episode_start_future.result()
                    if started.get("initial_phase") != "coarse":
                        raise ValueError(
                            "Continuous collection must restart in the coarse phase."
                        )
                    self._episode_start_future = None
                    logging.getLogger("flexiv.status").info(
                        "episode started id=%s phase=coarse", started.get("episode_id")
                    )
                    self._preview_future = None
                    self._reset_episode_schedule()
                    previous_joint_target = previous_joint_step = None
                    observation = self.robot.read_observation(
                        tick=0, prompt=self.prompt
                    )
                    if observation is None:
                        raise RuntimeError("Observation missing at episode restart.")
                    self._submit(0, observation, None, bootstrap=True)
                    self._bootstrap_observation = observation
                    self._next_episode_bootstrap = True
                if self._next_episode_bootstrap and self._future.done():
                    result = self._future.result()
                    if (
                        result.get("phase", "coarse") != "coarse"
                        or int(result["frame"]) != 0
                    ):
                        raise ValueError(
                            "Episode bootstrap does not match its coarse origin."
                        )
                    self._actions = self._parse_block(result, horizon=self.base_horizon)
                    self._action_context = result.get("action_context")
                    self._future = None
                    self._next_episode_bootstrap = False
                    self._episode_active = True
                    if self._fine_requested_during_start:
                        self._fine_requested = True
                        self._fine_requested_during_start = False
                        logging.getLogger("flexiv.status").info(
                            "queued fine phase request applied; waiting for base boundary=%d",
                            self._next_boundary(),
                        )
                    active_request_frame = 0
                    episode_started = self._monotonic()
                    ready_at_tick_start = False
                if not self._episode_active:
                    if (
                        self._episode_end_future is None
                        and self._episode_start_future is None
                        and not self._next_episode_bootstrap
                    ):
                        self._preview_while_paused()
                    self.robot.control_tick(
                        tick=self._frame,
                        action=None,
                        action_context={"control_phase": "coarse"},
                    )
                    self._total_ticks += 1
                    remaining = deadline - self._monotonic()
                    if remaining > 0:
                        self._sleep(remaining)
                    continue
                if operator.reset_policy:
                    reset_options = {}
                    if operator.reset_to_coarse:
                        self._phase = "coarse"
                        self.horizon = self.base_horizon
                        self._fine_requested = False
                        self._fine_requested_during_start = False
                        reset_options["phase"] = "coarse"
                        logging.getLogger("flexiv.status").info(
                            "Home recovery resumed coarse phase; configured joint motion limits restored"
                        )
                    self.policy.call(
                        "reset_context", frame=self._frame, **reset_options
                    )
                    self._future = None
                    self._actions = None
                    self._accepted_pending = None
                    self._chunk_start = self._frame
                    self._coarse_origin = (
                        self._frame - self.request_steps
                        if self.coarse_origin_schedule
                        else self._frame
                    )
                    self._waiting_boundary = None
                    self._waiting_observation = None
                    self._bootstrap_observation = None
                    self._holding_for_inference = True
                    self._recovery_request = True
                    ready_at_tick_start = False
                if not operator.policy_enabled:
                    self._policy_enabled = False
                if operator.enter_fine_phase and self._phase == "coarse":
                    if not self.fine_available:
                        raise RuntimeError(
                            "This inference server has no fine-phase Actor."
                        )
                    if not self._fine_requested:
                        self._fine_requested = True
                        logging.getLogger("flexiv.status").info(
                            "fine phase requested tick=%d; waiting for base boundary=%d",
                            self._frame,
                            self._next_boundary(),
                        )
                boundary = (
                    self._waiting_boundary
                    if self._waiting_boundary is not None
                    else self._next_boundary()
                )
                handoff = self._policy_enabled and (
                    self._waiting_boundary is not None or self._frame >= boundary
                )
                accepted: dict[str, int] | None = None
                accepted_late = False
                if handoff:
                    ready = self._future is not None and (
                        ready_at_tick_start or self._future.done()
                    )
                    if not ready and self._waiting_boundary is None:
                        self._waiting_boundary = self._frame
                        self._waiting_observation = self.robot.read_observation(
                            tick=self._frame, prompt=self.prompt
                        )
                        if self._waiting_observation is None:
                            raise RuntimeError(
                                "Observation missing at handoff boundary."
                            )
                        logging.warning(
                            "inference pending at handoff tick=%d request_tick=%d budget_ms=%.2f; holding target until a usable action block arrives",
                            self._frame,
                            self._submitted_tick,
                            self._request_budget_ms(),
                        )
                        self._next_hold_warning_frame = (
                            self._frame + self._hold_warning_interval_ticks()
                        )
                    elif (
                        not ready
                        and self._next_hold_warning_frame is not None
                        and self._frame >= self._next_hold_warning_frame
                    ):
                        elapsed_ms = (
                            (self._monotonic() - self._submitted_at) * 1000.0
                            if self._submitted_at is not None
                            else float("nan")
                        )
                        logging.warning(
                            "inference still pending at tick=%d request_tick=%d elapsed_ms=%.1f; continuing to hold target",
                            self._frame,
                            self._submitted_tick,
                            elapsed_ms,
                        )
                        self._next_hold_warning_frame = (
                            self._frame + self._hold_warning_interval_ticks()
                        )

                if handoff and ready:
                    result = self._future.result()
                    if int(result["frame"]) != self._request_frame:
                        raise ValueError("Inference response does not match handoff.")
                    if result.get("phase", self._phase) != self._phase:
                        raise ValueError(
                            "Inference response phase does not match handoff."
                        )
                    action_count = (
                        self.base_horizon - (boundary - self._request_frame)
                        if self._phase == "coarse" and self.coarse_origin_schedule
                        else self.horizon
                    )
                    invalid_block = False
                    try:
                        actions = self._parse_block(result, horizon=action_count)
                    except (ValueError, KeyError) as exc:
                        logging.warning(
                            "Invalid inference block frame=%d: %s",
                            self._request_frame,
                            exc,
                        )
                        actions = None
                        invalid_block = True
                    expired = actions is not None and self._frame >= boundary + len(
                        actions
                    )
                    fine_bridge_expired = (
                        self._fine_requested
                        and self._frame - boundary >= self.fine_horizon
                    )
                    fine_late = self._phase == "fine" and self._frame > boundary
                    if expired or fine_bridge_expired or fine_late or invalid_block:
                        discard = self.policy.call(
                            "discard",
                            frame=self._request_frame,
                            reason=(
                                "invalid_action_block"
                                if invalid_block
                                else "late_fine_block"
                                if fine_late
                                else "stale_fine_bridge"
                                if fine_bridge_expired
                                else "stale_action_block"
                            ),
                        )
                        # Discard is ordered before the replacement inference by
                        # RemotePolicy, so the server can release this decision.
                        del discard
                        logging.warning(
                            "discarding stale inference frame=%d tick=%d target_boundary=%d; holding target and requesting a fresh block",
                            self._request_frame,
                            self._frame,
                            boundary,
                        )
                        self._future = None
                        self._waiting_boundary = None
                        self._waiting_observation = None
                        self._chunk_start = self._frame
                        if self.coarse_origin_schedule:
                            self._coarse_origin = self._frame - self.request_steps
                        self._holding_for_inference = True
                        self._recovery_request = True
                        self._next_hold_warning_frame = None
                        actions = None
                    if actions is not None and self._fine_requested:
                        actions = actions[: self.fine_horizon]
                        self.horizon = self.fine_horizon
                        self._phase = "fine"
                        self._fine_requested = False
                        logging.getLogger("flexiv.status").info(
                            "robot entered fine phase at tick=%d", boundary
                        )
                    if actions is not None:
                        self._actions = actions
                        active_request_frame = self._request_frame
                        self._chunk_start = boundary
                        if self._phase == "coarse" and self.coarse_origin_schedule:
                            self._coarse_origin = self._request_frame
                        self._action_context = result.get("action_context")
                        self._holding_for_inference = False
                        self._next_hold_warning_frame = None
                    model_ms = float(result.get("model_latency_ms", float("nan")))
                    received_at = result.get("_client_received_at")
                    if received_at is None:
                        received_at = self._monotonic()
                    received_at = float(received_at)
                    response_ms = (
                        (received_at - self._submitted_at) * 1000.0
                        if self._submitted_at is not None
                        else float("nan")
                    )
                    if actions is not None:
                        logging.getLogger("flexiv.status").info(
                            "robot handoff session=%s tick=%d phase=%s model_version=%d completion_hz=%s model_hz=%.2f model_ms=%.2f response_hz=%.2f response_ms=%.2f handoff_ms=%.2f budget_ms=%.2f slack_ms=%.2f",
                            getattr(self.policy, "session", "fake"),
                            self._frame,
                            self._phase,
                            int(result["model_version"]),
                            f"{result['inference_hz']:.2f}"
                            if result.get("inference_hz") is not None
                            else "warming up",
                            1000.0 / model_ms if model_ms > 0 else float("nan"),
                            model_ms,
                            1000.0 / response_ms if response_ms > 0 else float("nan"),
                            response_ms,
                            (self._monotonic() - self._submitted_at) * 1000.0
                            if self._submitted_at is not None
                            else float("nan"),
                            self._request_budget_ms(),
                            (deadline - self._monotonic()) * 1000.0,
                        )
                    if actions is not None:
                        if self._phase == "coarse" and self.coarse_origin_schedule:
                            logging.info(
                                "coarse chunk entry tick=%d request_frame=%d planned_boundary=%d first_prediction_index=%d held_ticks=%d",
                                self._frame,
                                self._request_frame,
                                boundary,
                                self._frame - self._request_frame,
                                self._frame - boundary,
                            )
                        accepted = {
                            "frame": self._request_frame,
                            "accept_frame": boundary,
                            "accept_tick": self._frame,
                            "inference_delay_steps": max(
                                0,
                                int(np.ceil(response_ms / (self.period * 1000.0))),
                            ),
                            "model_version": int(result["model_version"]),
                        }
                        self._accepted_pending = accepted
                        self._future = None
                        accepted_late = self._waiting_boundary is not None
                        self._waiting_boundary = None

                if self._recovery_request:
                    request_tick = self._frame
                elif self._phase == "fine":
                    request_tick = self._chunk_start
                else:
                    origin = (
                        self._coarse_origin
                        if self.coarse_origin_schedule
                        else self._chunk_start
                    )
                    request_tick = origin + self.request_steps
                should_submit = (
                    self._policy_enabled
                    and self._future is None
                    and self._frame >= request_tick
                )
                if should_submit:
                    reuse_boundary_observation = accepted_late and (
                        self._phase == "fine" or self.request_steps == 0
                    )
                    observation = (
                        self._waiting_observation
                        if reuse_boundary_observation
                        else self._bootstrap_observation
                        if self._frame == 0
                        else self.robot.read_observation(
                            tick=self._frame, prompt=self.prompt
                        )
                    )
                    if observation is None:
                        raise RuntimeError("Observation missing at inference request.")
                    if (
                        previous_joint_target is not None
                        and logging.getLogger().isEnabledFor(logging.INFO)
                    ):
                        joint_state = np.asarray(
                            observation.policy_inputs.get("states", []),
                            dtype=np.float64,
                        )
                        if joint_state.shape == (8,):
                            logging.info(
                                "joint request tracking tick=%d max_command_tracking_error_deg=%.4f measured_joint_rad=%s last_target_joint_rad=%s",
                                self._frame,
                                float(
                                    np.rad2deg(
                                        np.abs(previous_joint_target - joint_state[:7])
                                    ).max()
                                ),
                                joint_state[:7].tolist(),
                                previous_joint_target.tolist(),
                            )
                    self._submit(
                        boundary if reuse_boundary_observation else self._frame,
                        observation,
                        None,
                    )
                    self._recovery_request = False
                    self._waiting_observation = None
                    self._bootstrap_observation = None

                action = None
                if (
                    self._policy_enabled
                    and self._waiting_boundary is None
                    and not self._holding_for_inference
                ):
                    assert self._actions is not None
                    action = self._actions[self._frame - self._chunk_start]
                receipt = self.robot.control_tick(
                    tick=self._frame,
                    action=action,
                    action_context={
                        **(self._action_context or {}),
                        "control_phase": self._phase,
                    },
                )
                joint_target = receipt.receipt.get("joint_target_rad")
                if joint_target is not None and logging.getLogger().isEnabledFor(
                    logging.INFO
                ):
                    joint_target = np.asarray(joint_target, dtype=np.float64)
                    if previous_joint_target is not None:
                        joint_step = joint_target - previous_joint_target
                        if handoff and action is not None:
                            step_change = (
                                joint_step - previous_joint_step
                                if previous_joint_step is not None
                                else joint_step
                            )
                            logging.info(
                                "joint handoff tick=%d late_ticks=%d max_target_step_deg=%.4f max_target_step_change_deg=%.4f",
                                self._frame,
                                self._frame - boundary,
                                float(np.rad2deg(np.abs(joint_step).max())),
                                float(np.rad2deg(np.abs(step_change).max())),
                            )
                        previous_joint_step = joint_step
                    previous_joint_target = joint_target.copy()
                if receipt.stop:
                    self._success = receipt.success
                    break
                if not receipt.policy_enabled:
                    self._policy_enabled = False
                receipt_info = dict(receipt.receipt)
                if getattr(self.policy, "diagnostics_enabled", False):
                    receipt_info["execution_alignment"] = {
                        "source_request_frame": active_request_frame,
                        "chunk_start_frame": self._chunk_start,
                        "prediction_index": (
                            self._frame - active_request_frame
                            if action is not None
                            else None
                        ),
                        "phase": self._phase,
                    }
                receipt_info.setdefault(
                    "policy_proposal_physical",
                    None if action is None else np.asarray(action).copy(),
                )
                receipt_info["policy_proposal_valid"] = action is not None and bool(
                    receipt_info.get("policy_proposal_valid", True)
                )
                remaining = deadline - self._monotonic()
                receipt_info["schedule_valid"] = bool(
                    receipt_info.get("schedule_valid", True)
                ) and (action is not None and remaining >= 0)
                self.policy.call("receipt", frame=self._frame, info=receipt_info)
                self._frame += 1
                self._total_ticks += 1
                remaining = deadline - self._monotonic()
                if remaining > 0:
                    self._sleep(remaining)
                else:
                    logging.warning(
                        "robot deadline miss tick=%d ms=%.2f",
                        self._frame,
                        -remaining * 1000.0,
                    )
            return 0
        finally:
            try:
                self.robot.stop()
            finally:
                try:
                    self.robot.close()
                finally:
                    if self._discard_episode:
                        self.policy.close(self._frame - 1, self._success, discard=True)
                    else:
                        self.policy.close(self._frame - 1, self._success)
