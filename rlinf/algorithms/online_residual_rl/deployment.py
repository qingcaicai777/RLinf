# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Fixed six-frame rollout handoff for Stage2 residual inference."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch

from rlinf.models.embodiment.openpi.rtc.types import (
    PhysicalActionStep,
    PolicyObservation,
)
from rlinf.utils.logging import get_logger

from .config import OnlineResidualRLConfig
from .online import ActorMailbox, EpisodeCollector, OperatorConsole
from .scheduling import HandoffSchedule


class ResidualPolicy(Protocol):
    """Stage1 PyTorch or APX base with the same Stage2 output contract."""

    def predict_action_batch(
        self, env_obs: dict[str, Any], *, mode: str, committed: torch.Tensor | None
    ) -> tuple[torch.Tensor, dict[str, Any]]: ...

    def encode_receipts(self, physical_actions: torch.Tensor) -> torch.Tensor: ...

    def close(self) -> None: ...


class ResidualRobot(Protocol):
    """Single control-thread owner for physical commands and observations."""

    def start(self) -> None: ...

    def read_observation(self, *, tick: int, prompt: str) -> PolicyObservation: ...

    def read_tcp_pose(self) -> np.ndarray: ...

    def apply_policy_action(
        self, step: PhysicalActionStep
    ) -> dict[str, Any] | None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class ResidualDeploymentConfig:
    """Timing and task settings for the fixed handoff executor."""

    prompt: str
    control_hz: float = 30.0
    max_ticks: int | None = None
    task_timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if not self.prompt or self.control_hz <= 0:
            raise ValueError("A prompt and positive control_hz are required.")
        if self.task_timeout_seconds is not None and (
            not np.isfinite(self.task_timeout_seconds) or self.task_timeout_seconds <= 0
        ):
            raise ValueError("task_timeout_seconds must be positive and finite.")
        if self.max_ticks is not None and self.max_ticks <= 0:
            raise ValueError("max_ticks must be positive.")


def _policy_input(observation: PolicyObservation) -> dict[str, Any]:
    images = observation.images
    return {
        "main_images": np.expand_dims(images["observation/image"], 0),
        "wrist_images": np.expand_dims(images["observation/wrist_image"], 0),
        "states": np.expand_dims(observation.state, 0),
        "task_descriptions": [observation.prompt],
    }


class ResidualHandoffRuntime:
    """Execute only the new half of each Actor block at absolute deadlines.

    A supplied single-worker executor also owns thread-bound policy loading and
    bootstrap; the caller keeps it alive across episodes and closes the policy
    on that worker after the last runtime stops.
    """

    def __init__(
        self,
        robot: ResidualRobot,
        policy: ResidualPolicy,
        contract: OnlineResidualRLConfig,
        config: ResidualDeploymentConfig,
        *,
        project_actions: Callable[[np.ndarray], np.ndarray],
        collector: EpisodeCollector | None = None,
        actor_mailbox: ActorMailbox | None = None,
        operator: OperatorConsole | None = None,
        terminal_label: bool | None = None,
        own_policy: bool = True,
        inference_executor: ThreadPoolExecutor | None = None,
        monotonic=time.monotonic,
        sleep=time.sleep,
    ) -> None:
        self.robot = robot
        self.policy = policy
        self.contract = contract
        self.config = config
        self._project_actions = project_actions
        self.collector = collector
        self.actor_mailbox = actor_mailbox
        self.operator = operator
        self.terminal_label = terminal_label
        self.abort_requested = False
        self.own_policy = own_policy
        self.schedule = HandoffSchedule(contract)
        self._monotonic = monotonic
        self._sleep = sleep
        self._logger = get_logger()
        self._owns_executor = inference_executor is None
        self._executor = inference_executor or ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="residual-infer"
        )
        self._future: Future | None = None
        self._submit_time = 0.0
        self._ready_physical: np.ndarray | None = None
        self._raw_queue: torch.Tensor | None = None
        self._ready_raw: torch.Tensor | None = None
        self._ready_reference_pose: np.ndarray | None = None
        self._queue_reference_pose: np.ndarray | None = None
        self._queue_physical: np.ndarray | None = None
        self._stop_event = threading.Event()
        self._closed = False

    @staticmethod
    def _snapshot_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Detach decision features before handing them to the collector."""

        def snapshot(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().clone()
            if isinstance(value, dict):
                return {key: snapshot(item) for key, item in value.items()}
            return value

        return snapshot(metadata)

    def request_stop(self) -> None:
        """Ask the control owner to exit after its current tick."""
        self._stop_event.set()

    def abort(self) -> None:
        """Stop without publishing a possibly incomplete human label."""
        self.terminal_label = None
        self.abort_requested = True
        self.request_stop()

    def _project(self, physical: torch.Tensor) -> tuple[np.ndarray, torch.Tensor]:
        horizon = self.contract.inference_horizon
        array = physical.detach().cpu().numpy()
        if array.shape != (1, horizon, self.contract.action_dim):
            raise ValueError(f"Execution block must have shape [1, {horizon}, 7].")
        projected = self._project_actions(array[0])
        normalized = self.policy.encode_receipts(torch.from_numpy(projected[None]))
        if normalized.shape != (1, horizon, self.contract.action_dim):
            raise ValueError("Encoded command queue has an invalid shape.")
        if not torch.isfinite(normalized).all():
            raise ValueError("Encoded command queue is non-finite.")
        return projected, normalized[0].detach().cpu()

    def _infer(
        self,
        observation: PolicyObservation,
        committed: torch.Tensor | None,
        actor_update: tuple[int, dict[str, torch.Tensor]] | None,
    ) -> tuple[
        float, np.ndarray, torch.Tensor, torch.Tensor, np.ndarray, dict[str, Any]
    ]:
        """Infer off-thread using the prior raw model-normalized chunk.

        The scheduler separately snapshots the final accepted queue for replay.
        Only the raw proposal is passed to a policy that implements RTC
        guidance; receipt normalization must not silently replace it.
        """
        if actor_update is not None:
            self.policy.install_actor(actor_update[1], actor_update[0])
        physical, metadata = self.policy.predict_action_batch(
            _policy_input(observation),
            mode="eval",
            committed=None if committed is None else committed[None],
        )
        n = self.contract.inference_horizon
        block, normalized = self._project(physical[:, n : 2 * n])
        raw = metadata.get("raw_model_actions")
        if isinstance(raw, torch.Tensor):
            raw = raw.detach().to(dtype=torch.float32)
            if (
                raw.ndim == 3
                and raw.shape[0] == 1
                and raw.shape[1] >= self.contract.actor_horizon
                and raw.shape[2] > 0
                and torch.isfinite(raw).all()
            ):
                raw = raw[:, n:][0].cpu()
            else:
                raw = None
        if raw is None:
            if getattr(self.policy, "conditions_on_committed", False):
                raise ValueError("RTC guidance requires raw PI0.5 model actions.")
            # APX does not condition the VLA flow on previous actions.
            raw = normalized.clone()
        if observation.tcp_reference_pose is None:
            raise ValueError("Residual inference requires an observation TCP pose.")
        return (
            self._monotonic(),
            block,
            normalized,
            raw,
            observation.tcp_reference_pose.copy(),
            self._snapshot_metadata(metadata),
        )

    def _submit(self, observation: PolicyObservation, tick: int) -> None:
        if self._future is not None:
            raise RuntimeError("Only one policy inference may be in flight.")
        actor_update = self.actor_mailbox.take() if self.actor_mailbox else None
        version = (
            actor_update[0]
            if actor_update
            else (self.actor_mailbox.version if self.actor_mailbox else 0)
        )
        request = self.schedule.begin(tick, model_version=version)
        self._request_frame = request.frame
        self._request_sequence = request.sequence
        self._request_version = request.model_version
        self._submit_time = self._monotonic()
        guidance = (
            self._raw_queue.detach().clone()
            if getattr(self.policy, "conditions_on_committed", False)
            else request.committed.clone()
        )
        future = self._executor.submit(self._infer, observation, guidance, actor_update)
        self._future = future
        self._logger.info("residual submit frame=%s request=%s", tick, request.sequence)

    def _accept(self, tick: int) -> None:
        future = self._future
        if future is None or not future.done():
            return
        self._future = None
        finished_at, block, normalized, raw, reference_pose, metadata = future.result()
        completed_frame = (
            self._request_frame
            + (finished_at - self._submit_time) * self.config.control_hz
        )
        accepted = self.schedule.submit(
            self._request_sequence, normalized, completed_frame=completed_frame
        )
        latency_ms = (finished_at - self._submit_time) * 1000.0
        if accepted:
            self._ready_physical = block
            self._ready_raw = raw
            self._ready_reference_pose = reference_pose
            if self.collector is not None:
                self.collector.decision(
                    self._request_frame,
                    metadata,
                    self._request_version,
                    tcp_reference_pose=self._ready_reference_pose,
                )
            self._logger.info(
                "residual ready frame=%s request=%s latency_ms=%.2f",
                tick,
                self._request_sequence,
                latency_ms,
            )
        else:
            self._logger.warning(
                "residual late result discarded frame=%s request=%s latency_ms=%.2f",
                tick,
                self._request_sequence,
                latency_ms,
            )

    def run(self) -> int:
        """Bootstrap once, then never wait for normal inference inside a tick."""
        n = self.contract.inference_horizon
        tick = 0
        try:
            self.robot.start()
            observation = self.robot.read_observation(tick=0, prompt=self.config.prompt)
            if self._owns_executor:
                initial, initial_metadata = self.policy.predict_action_batch(
                    _policy_input(observation), mode="eval", committed=None
                )
            else:
                # The APX Python model is thread-bound: load, bootstrap and
                # subsequent inference must all use its dedicated worker.
                initial, initial_metadata = self._executor.submit(
                    self.policy.predict_action_batch,
                    _policy_input(observation),
                    mode="eval",
                    committed=None,
                ).result()
            self._queue_physical, normalized = self._project(initial[:, :n])
            initial_raw = initial_metadata.get("raw_model_actions")
            if isinstance(initial_raw, torch.Tensor):
                initial_raw = initial_raw.detach().to(dtype=torch.float32)
                if (
                    initial_raw.ndim == 3
                    and initial_raw.shape[0] == 1
                    and initial_raw.shape[1] >= n
                    and torch.isfinite(initial_raw).all()
                ):
                    initial_raw = initial_raw[0].cpu()
                else:
                    initial_raw = None
            if initial_raw is None and getattr(
                self.policy, "conditions_on_committed", False
            ):
                raise ValueError("RTC guidance requires raw PI0.5 model actions.")
            self._raw_queue = (
                initial_raw if initial_raw is not None else normalized.clone()
            )
            self._queue_reference_pose = observation.tcp_reference_pose.copy()
            self.schedule.recover(normalized)
            period = 1.0 / self.config.control_hz
            deadline = self._monotonic()
            episode_started = deadline
            while not self._stop_event.is_set():
                if (
                    self.config.task_timeout_seconds is not None
                    and self._monotonic() - episode_started
                    >= self.config.task_timeout_seconds
                ):
                    self.terminal_label = False
                    self._logger.warning(
                        "Task timed out after %.2f seconds",
                        self.config.task_timeout_seconds,
                    )
                    break
                if self.operator is not None:
                    for command in self.operator.poll():
                        if command in {"on", "off"}:
                            self.robot.set_intervention_enabled(command == "on")
                        elif command in {"success", "failure"}:
                            self.terminal_label = command == "success"
                            self.request_stop()
                        elif command == "stop":
                            self.abort()
                if self._stop_event.is_set():
                    break
                self._accept(tick)
                if self.config.max_ticks is not None and tick >= self.config.max_ticks:
                    break
                deadline += period
                if tick and tick % n == 0:
                    # A future may complete between the regular poll above
                    # and this boundary.  Poll once more without waiting so a
                    # result that completed in this control tick is staged
                    # before the fixed handoff decision.
                    self._accept(tick)
                    next_queue = self.schedule.handoff(tick)
                    if next_queue is None or self._ready_physical is None:
                        self._logger.error(
                            "residual handoff missed frame=%s; stopping", tick
                        )
                        break
                    self._queue_physical = self._ready_physical
                    self._raw_queue = self._ready_raw
                    self._queue_reference_pose = self._ready_reference_pose
                    self._ready_physical = None
                    self._ready_raw = None
                    self._ready_reference_pose = None
                if tick % n == 0:
                    if tick:
                        observation = self.robot.read_observation(
                            tick=tick, prompt=self.config.prompt
                        )
                    self._submit(observation, tick)
                else:
                    self.robot.read_tcp_pose()
                assert self._queue_physical is not None
                values = self._queue_physical[tick % n]
                receipt = self.robot.apply_policy_action(
                    PhysicalActionStep(
                        tcp_delta=values[:6],
                        gripper=float(values[6]),
                        chunk_id=tick // n,
                        chunk_index=tick % n,
                        tcp_reference_pose=self._queue_reference_pose,
                    )
                )
                if self.collector is not None:
                    if receipt is None:
                        raise RuntimeError(
                            "Online collection requires final command receipts."
                        )
                    self.collector.receipt(
                        tick,
                        {
                            **receipt,
                            "schedule_valid": deadline - self._monotonic() >= 0,
                        },
                    )
                if receipt is not None:
                    accepted = self.policy.encode_receipts(
                        torch.as_tensor(
                            receipt["action_receipt_physical"], dtype=torch.float32
                        )[None]
                    )[0]
                    self.schedule.record_execution(tick, accepted)
                tick += 1
                remaining = deadline - self._monotonic()
                if remaining > 0:
                    self._sleep(remaining)
                else:
                    self._logger.warning(
                        "residual deadline miss frame=%s ms=%.2f",
                        tick,
                        -remaining * 1000,
                    )
            return 0 if not self.schedule.recovering else 1
        finally:
            try:
                self.robot.stop()
                if self.collector is not None:
                    self.collector.finish(tick - 1, self.terminal_label)
            finally:
                self.close()

    def close(self) -> None:
        """Stop robot before waiting for any in-flight policy request."""
        if self._closed:
            return
        self._closed = True
        self.request_stop()
        try:
            self.robot.stop()
        finally:
            if self._owns_executor:
                self._executor.shutdown(wait=True, cancel_futures=True)
            else:
                # Drain the request before closing cameras/SDK. The caller
                # retains this worker for the next episode and policy.close().
                self._executor.submit(lambda: None).result()
            try:
                self.robot.close()
            finally:
                if self.own_policy:
                    self.policy.close()
