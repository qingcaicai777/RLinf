# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Sequential policy service for a separate robot control process."""

from __future__ import annotations

import logging
import socket
import time
import uuid
from collections import Counter, deque
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import numpy as np
import torch

from .online import EpisodeCollector
from .remote_wire import PROTOCOL_VERSION, receive_message, send_message
from .storage import PendingEpisodeStore
from .transition import OnlineResidualTransition


class RemoteInferenceServer:
    """Own the policy and optional episode collector for one client at a time."""

    def __init__(
        self,
        policy: Any,
        contract: Any,
        trainer: Any = None,
        mailbox: Any = None,
        expected_inputs: dict[str, tuple[int, ...]] | None = None,
        observation_mapper: Any = None,
        backend: str | None = None,
        base_horizon: int | None = None,
        base_request_steps: int | None = None,
        start_base: bool = False,
        rtc_delay_buffer_size: int = 8,
        binary_gripper: bool = False,
        gripper_threshold: float = 0.5,
        commit_episode: Callable[[Sequence[OnlineResidualTransition], bool, int], None]
        | None = None,
        pending_store: PendingEpisodeStore | None = None,
    ):
        if backend not in {None, "torch", "apx"}:
            raise ValueError(f"Unsupported inference backend: {backend!r}")
        if rtc_delay_buffer_size <= 0:
            raise ValueError("RTC delay buffer size must be positive.")
        if not isinstance(binary_gripper, bool):
            raise ValueError("binary_gripper must be a boolean.")
        self.binary_gripper = binary_gripper
        self.gripper_threshold = float(gripper_threshold)
        if not np.isfinite(self.gripper_threshold):
            raise ValueError("Gripper threshold must be finite.")
        self.pending_store = pending_store or getattr(trainer, "pending_store", None)
        self.policy = policy
        self.contract = contract
        self.fine_horizon = int(contract.inference_horizon)
        self.base_request_steps = int(
            0 if base_request_steps is None else base_request_steps
        )
        self.base_horizon = int(
            base_horizon
            if base_horizon is not None
            else getattr(policy, "inference_horizon", self.fine_horizon)
        )
        if (
            self.fine_horizon <= 0
            or not 1 <= self.base_horizon <= 32
            or not 0 <= self.base_request_steps < self.base_horizon
        ):
            raise ValueError("Base horizon must be 1–32 and Actor horizon positive.")
        self.fine_available = bool(getattr(policy, "supports_fine_phase", True))
        self.initial_phase = (
            "coarse" if start_base or not self.fine_available else "fine"
        )
        self.coarse_origin_schedule = (
            self.initial_phase == "coarse" and self.base_request_steps > 0
        )
        if (
            self.coarse_origin_schedule
            and 2 * self.base_request_steps < self.base_horizon
        ):
            raise ValueError(
                "Prediction-origin coarse scheduling requires 2 * request_steps >= "
                "base_horizon so the next request does not precede the handoff."
            )
        if (
            self.initial_phase == "coarse"
            and not self.coarse_origin_schedule
            and 2 * self.base_horizon - self.base_request_steps > 32
        ):
            raise ValueError(
                "Coarse handoff exceeds the 32-step Stage1 prediction; request "
                "closer to the boundary or shorten the base chunk."
            )
        if (
            self.initial_phase == "coarse"
            and self.fine_available
            and self.base_horizon < self.fine_horizon
        ):
            raise ValueError("Base horizon must cover the six-frame fine bridge.")
        if (
            self.coarse_origin_schedule
            and self.fine_available
            and self.base_request_steps < self.fine_horizon
        ):
            raise ValueError("Coarse request interval cannot cover the fine bridge.")
        self._phase = self.initial_phase
        if trainer is not None and commit_episode is not None:
            raise ValueError("Choose either a learner or an episode storage callback.")
        self.commit_episode = trainer.submit if trainer is not None else commit_episode
        self.trainer = trainer
        self.mailbox = mailbox
        self.expected_inputs = expected_inputs
        self.observation_mapper = observation_mapper or self._default_observation_mapper
        self.backend = backend
        self.rtc_delay_buffer_size = rtc_delay_buffer_size
        self.rtc_delay_history: deque[int] = deque(maxlen=rtc_delay_buffer_size)
        self.collector: EpisodeCollector | None = None
        self.session: str | None = None
        self._pending_decisions: dict[
            int,
            tuple[
                dict[str, Any],
                int,
                np.ndarray | None,
                torch.Tensor | None,
                str,
                int,
                np.ndarray,
            ],
        ] = {}
        self._accepted_raw: torch.Tensor | None = None
        self._accepted_raw_frame: int | None = None
        self._accepted_raw_reference: np.ndarray | None = None
        self._accepted_physical: np.ndarray | None = None
        self._inference_completions: deque[float] = deque(maxlen=10)
        self._episode_active = False
        self._schedule_counts: Counter[str] = Counter()
        self._closed = False

    @staticmethod
    def _default_observation_mapper(
        observation: dict[str, Any], prompt: str
    ) -> dict[str, Any]:
        """Batch adapter-mapped model inputs and add the task description."""
        if not isinstance(observation, dict) or not observation:
            raise ValueError("Robot observation must be a nonempty mapping.")
        result = {name: np.asarray(value)[None] for name, value in observation.items()}
        result["task_descriptions"] = [prompt]
        return result

    def _new_collector(self, episode_id: str) -> EpisodeCollector | None:
        if self.commit_episode is None:
            return None
        record_metrics = getattr(self.trainer, "record_metrics", None)
        return EpisodeCollector(
            self.contract,
            self.policy.encode_receipts,
            self.commit_episode,
            episode_id=episode_id,
            pending_store=self.pending_store,
            record_metrics=partial(record_metrics, episode_id=episode_id)
            if record_metrics
            else None,
        )

    def _finish(
        self, frame: int, success: bool | None, *, discard: bool = False
    ) -> None:
        collector = self.collector
        try:
            if collector is not None:
                if discard:
                    collector.finish(frame, success, discard=True)
                else:
                    collector.finish(frame, success)
                logging.getLogger("flexiv.status").info(
                    "Episode %s scheduling: coarse_requests=%d fine_requests=%d fine_accepted=%d "
                    "fine_discarded=%d coarse_bridge_discarded=%d",
                    getattr(collector, "episode_id", self.session),
                    self._schedule_counts["coarse_requests"],
                    self._schedule_counts["fine_requests"],
                    self._schedule_counts["fine_accepted"],
                    self._schedule_counts["fine_discarded"],
                    self._schedule_counts["coarse_bridge_discarded"],
                )
        finally:
            self.collector = None
            self._episode_active = False
            self._pending_decisions.clear()
            self._accepted_raw = None
            self._accepted_raw_frame = None
            self._accepted_raw_reference = None
            self._accepted_physical = None
            self._inference_completions.clear()
            self.rtc_delay_history.clear()
            self._phase = self.initial_phase
            reset = getattr(self.policy, "reset", None)
            if callable(reset):
                reset()
            self.session = None

    def handle(self, message: dict[str, Any]) -> dict[str, Any]:
        """Process one ordered request on the policy-owning thread."""
        kind = message.get("type")
        session = message.get("session")
        if kind == "start":
            if self.session is not None:
                raise RuntimeError("Finish the active inference session first.")
            if not isinstance(session, str) or not session:
                raise ValueError("A nonempty session id is required.")
            if self.backend is not None and message.get("backend") != self.backend:
                raise ValueError(
                    "Robot and inference server backends differ: "
                    f"robot={message.get('backend')!r}, server={self.backend!r}."
                )
            if message.get("binary_gripper", False) != self.binary_gripper:
                raise ValueError(
                    "Robot and inference server binary gripper settings differ."
                )
            if (
                self.binary_gripper
                and message.get("gripper_threshold", 0.5) != self.gripper_threshold
            ):
                raise ValueError(
                    "Robot and learner gripper thresholds differ: "
                    f"robot={message.get('gripper_threshold', 0.5)!r}, "
                    f"learner={self.gripper_threshold!r}."
                )
            representation = getattr(
                self.contract, "action_representation", "eef_delta"
            )
            representation = getattr(representation, "value", representation)
            if message.get("action_representation", representation) != representation:
                raise ValueError(
                    "Robot and inference server action representations differ."
                )
            if self.expected_inputs is not None:
                announced = message.get("input_shapes")
                if (
                    not isinstance(announced, dict)
                    or {name: tuple(shape) for name, shape in announced.items()}
                    != self.expected_inputs
                ):
                    raise ValueError(
                        "Robot observation shapes differ from server config."
                    )
            self._pending_decisions.clear()
            self._accepted_raw = None
            self._accepted_raw_frame = None
            self._accepted_raw_reference = None
            self._accepted_physical = None
            self._inference_completions.clear()
            self._phase = self.initial_phase
            reset = getattr(self.policy, "reset", None)
            if callable(reset):
                reset()
            self._schedule_counts.clear()
            self.collector = self._new_collector(session)
            self._episode_active = True
            self.session = session
            return {
                "action_representation": representation,
                "binary_gripper": self.binary_gripper,
                "gripper_threshold": self.gripper_threshold,
                "action_dim": int(getattr(self.contract, "action_dim", 7)),
                "type": "started",
                "supports_diagnostics": True,
                "continuous_episodes": self.fine_available,
                "session": session,
                "inference_horizon": (
                    self.base_horizon if self._phase == "coarse" else self.fine_horizon
                ),
                "base_horizon": self.base_horizon,
                "base_request_steps": self.base_request_steps,
                "coarse_origin_schedule": self.coarse_origin_schedule,
                "fine_horizon": self.fine_horizon,
                "fine_available": self.fine_available,
                "initial_phase": self._phase,
                "rtc_enabled": bool(
                    getattr(self.policy, "conditions_on_committed", False)
                ),
            }
        if session != self.session or self.session is None:
            raise ValueError("Unknown or expired inference session.")
        if kind == "start_episode":
            if self._episode_active or not self.fine_available:
                raise RuntimeError(
                    "Finish the current Stage2 episode before starting another."
                )
            self._pending_decisions.clear()
            self._accepted_raw = None
            self._accepted_raw_frame = None
            self._accepted_raw_reference = None
            self._accepted_physical = None
            self._inference_completions.clear()
            self.rtc_delay_history.clear()
            reset = getattr(self.policy, "reset", None)
            if callable(reset):
                reset()
            self._schedule_counts.clear()
            episode_id = str(uuid.uuid4())
            self.collector = self._new_collector(episode_id)
            self._episode_active = True
            self._phase = "coarse"
            return {
                "type": "episode_started",
                "session": session,
                "episode_id": episode_id,
                "initial_phase": "coarse",
            }
        if kind in {"finish", "finish_episode"}:
            success = message.get("success")
            if success is not None and not isinstance(success, bool):
                raise ValueError("Terminal success must be true, false or null.")
            discard = message.get("discard", False)
            if not isinstance(discard, bool):
                raise ValueError("Discard must be a boolean.")
            if kind == "finish_episode" and (
                not self._episode_active or not self.fine_available
            ):
                raise RuntimeError("No active Stage2 episode to finish.")
            self._finish(int(message["frame"]), success, discard=discard)
            if kind == "finish_episode":
                self._phase = "coarse"
                self.session = session
                return {"type": "episode_finished", "session": session}
            return {"type": "finished", "session": session}
        preview = message.get("preview", False)
        if not isinstance(preview, bool):
            raise ValueError("Preview must be a boolean.")
        if preview and (
            self._episode_active
            or kind != "infer"
            or not message.get("bootstrap")
            or message.get("phase") != "coarse"
        ):
            raise ValueError(
                "Paused previews require a coarse bootstrap outside an active episode."
            )
        if not self._episode_active and not preview:
            raise RuntimeError(
                "Start another episode before sending inference or receipts."
            )
        if kind == "reset_context":
            phase = message.get("phase")
            if phase is not None:
                if phase != "coarse" or not self.fine_available:
                    raise ValueError("Context phase reset requires Stage2 coarse mode.")
                self._phase = "coarse"
            self._pending_decisions.clear()
            self._accepted_raw = None
            self._accepted_raw_frame = None
            self._accepted_raw_reference = None
            self._accepted_physical = None
            self.rtc_delay_history.clear()
            reset = getattr(self.policy, "reset", None)
            if callable(reset):
                reset()
            return {"type": "context_reset", "session": session}
        if kind == "receipt":
            if self.collector is not None:
                self.collector.receipt(int(message["frame"]), message["info"])
            return {"type": "received", "session": session}
        if kind == "accept":
            frame = int(message["frame"])
            accept_tick = int(message.get("accept_tick", message["accept_frame"]))
            if accept_tick < int(message["accept_frame"]):
                raise ValueError("Inference cannot be accepted before its boundary.")
            try:
                metadata, version, reference, raw, phase, target_frame, physical = (
                    self._pending_decisions[frame]
                )
            except KeyError as exc:
                raise ValueError(
                    f"No pending inference for accepted frame {frame}."
                ) from exc
            if version != int(message["model_version"]):
                raise ValueError(
                    "Accepted Actor version differs from the inference result."
                )
            if int(message["accept_frame"]) != target_frame:
                raise ValueError("Accepted inference switched at the wrong boundary.")
            del self._pending_decisions[frame]
            measured_delay = int(
                message.get(
                    "inference_delay_steps", accept_tick - int(message["accept_frame"])
                )
            )
            if measured_delay < 0:
                raise ValueError("RTC inference delay must be nonnegative.")
            self.rtc_delay_history.append(measured_delay)
            # Absolute joints keep the prediction's original time origin after
            # a late switch: the robot already skips its expired prefix.
            keep_raw = accept_tick == target_frame or (
                getattr(self.contract, "action_representation", "eef_delta")
                == "joint_absolute"
                and phase == "coarse"
            )
            self._accepted_raw = raw.clone() if raw is not None and keep_raw else None
            self._accepted_raw_frame = frame if self._accepted_raw is not None else None
            self._accepted_physical = (
                physical if self._accepted_raw is not None else None
            )
            self._accepted_raw_reference = (
                reference.copy()
                if self._accepted_raw is not None and reference is not None
                else None
            )
            if raw is not None and not keep_raw:
                logging.warning(
                    "RTC context cleared after late handoff session=%s frame=%d late_ticks=%d",
                    session,
                    frame,
                    accept_tick - target_frame,
                )
            elif raw is not None and accept_tick != target_frame:
                logging.info(
                    "RTC joint context retained after late handoff session=%s frame=%d late_ticks=%d origin_frame=%d",
                    session,
                    frame,
                    accept_tick - target_frame,
                    frame,
                )
            if (
                self.collector is not None
                and phase == "fine"
                and accept_tick == int(message["accept_frame"])
            ):
                self._schedule_counts["fine_accepted"] += 1
                self.collector.decision(
                    frame,
                    metadata,
                    version,
                    tcp_reference_pose=reference,
                )
            return {"type": "accepted", "session": session, "frame": frame}
        if kind == "discard":
            frame = int(message["frame"])
            pending = self._pending_decisions.pop(frame, None)
            if pending is None:
                raise ValueError(f"No pending inference for discarded frame {frame}.")
            self._schedule_counts[f"{pending[4]}_discarded"] += 1
            if message.get("reason") == "stale_fine_bridge":
                self._schedule_counts["coarse_bridge_discarded"] += 1
            return {"type": "discarded", "session": session, "frame": frame}
        if kind != "infer":
            raise ValueError(f"Unsupported inference request type: {kind!r}")
        accepted = message.get("accepted")
        if accepted is not None:
            if not isinstance(accepted, dict):
                raise ValueError("Piggybacked handoff must be a mapping.")
            self.handle({**accepted, "type": "accept", "session": session})
        frame = int(message["frame"])
        bootstrap = bool(message.get("bootstrap", False))
        warmup = bool(message.get("warmup", False))
        if warmup and not bootstrap:
            raise ValueError("RTC warmup must be a bootstrap inference.")
        phase = str(message.get("phase", self._phase))
        if phase not in {"coarse", "fine"}:
            raise ValueError(f"Unsupported inference phase {phase!r}.")
        if phase != self._phase:
            if (
                self._phase != "coarse"
                or phase != "fine"
                or not self.fine_available
                or bootstrap
                or accepted is None
            ):
                raise ValueError("Fine phase requires an accepted coarse handoff.")
            self._phase = "fine"
            logging.getLogger("flexiv.status").info(
                "inference entered fine phase session=%s frame=%d", session, frame
            )
        n = self.base_horizon if phase == "coarse" else self.fine_horizon
        target_frame = int(message.get("target_frame", frame + n))
        action_offset = target_frame - frame
        if action_offset < 0 or action_offset > n:
            raise ValueError("Inference request must precede its chunk boundary.")
        if (
            phase == "coarse"
            and not bootstrap
            and self.coarse_origin_schedule
            and action_offset == n
        ):
            raise ValueError("Coarse handoff has no action before the next boundary.")
        if (
            phase == "coarse"
            and not bootstrap
            and not self.coarse_origin_schedule
            and action_offset + n > 32
        ):
            raise ValueError(
                "Coarse handoff exceeds the 32-step Stage1 prediction; request "
                "closer to the boundary or shorten the base chunk."
            )
        if not bootstrap and self._pending_decisions:
            raise RuntimeError("Previous inference has not been accepted or discarded.")
        observation = message.get("observation")
        if not isinstance(observation, dict) or not observation:
            raise ValueError("Inference requires a mapped robot observation.")
        if self.expected_inputs is not None and set(observation) != set(
            self.expected_inputs
        ):
            raise ValueError("Robot observation inputs differ from server config.")
        for name, value in observation.items():
            array = np.asarray(value)
            if not np.isfinite(array).all():
                raise ValueError(f"Non-finite observation input {name!r}.")
            if (
                self.expected_inputs is not None
                and array.shape != self.expected_inputs[name]
            ):
                raise ValueError(
                    f"Observation input {name!r} has shape {array.shape}, expected {self.expected_inputs[name]}."
                )
        action_context = message.get("action_context")
        reference = (
            np.asarray(action_context["training_reference"], dtype=np.float64)
            if isinstance(action_context, dict)
            and action_context.get("training_reference") is not None
            else None
        )
        if reference is not None and (
            reference.shape != (7,) or not np.isfinite(reference).all()
        ):
            raise ValueError("Action reference must be a finite TCP pose [7].")
        update = self.mailbox.take() if self.mailbox is not None else None
        if update is not None:
            self.policy.install_actor(update[1], update[0])
            logging.getLogger("flexiv.status").info(
                "Actor weights installed session=%s model_version=%d",
                session,
                update[0],
            )
        version = int(getattr(self.policy, "actor_version", 0))
        # The current block continues until target_frame even when inference
        # completes earlier, so its overlap is hard RTC context.
        rtc_delay_steps = max(
            action_offset if not bootstrap else 0,
            max(self.rtc_delay_history, default=0),
        )
        inference_started = time.perf_counter()
        logging.info(
            "inference started session=%s frame=%d phase=%s bootstrap=%s rtc_delay_steps=%d",
            session,
            frame,
            phase,
            bootstrap,
            rtc_delay_steps,
        )
        rollout_mode = getattr(self.trainer, "rollout_mode", "eval")
        if callable(rollout_mode):
            rollout_mode = rollout_mode()
        if rollout_mode not in {"eval", "base", "rollout"}:
            raise ValueError(f"Unsupported rollout mode {rollout_mode!r}.")
        mapped = self.observation_mapper(observation, str(message["prompt"]))
        committed = None
        if (
            getattr(self.policy, "conditions_on_committed", False)
            and self._accepted_raw is not None
            and self._accepted_raw_frame is not None
        ):
            executed = frame - self._accepted_raw_frame
            if 0 <= executed < self._accepted_raw.shape[1]:
                committed = self._accepted_raw[:, executed:]
                rebase = getattr(self.policy, "rebase_committed_actions", None)
                if (
                    callable(rebase)
                    and getattr(self.contract, "action_representation", "eef_delta")
                    != "joint_absolute"
                ):
                    if self._accepted_raw_reference is None or reference is None:
                        raise ValueError(
                            "RTC rebasing requires both TCP reference poses."
                        )
                    committed = rebase(
                        committed, self._accepted_raw_reference, reference
                    )
        base_predict = getattr(self.policy, "predict_base_action_batch", None)
        logging.info(
            "inference RTC session=%s frame=%d conditioned=%s overlap_steps=%d handoff_offset=%d",
            session,
            frame,
            committed is not None,
            0 if committed is None else committed.shape[1],
            action_offset,
        )
        diagnostic_rng = {}
        if message.get("diagnostics", False) and self.backend == "torch":
            diagnostic_rng["torch_cpu_rng_state"] = torch.random.get_rng_state().numpy()
            if torch.cuda.is_initialized():
                diagnostic_rng["torch_cuda_rng_states"] = [
                    state.numpy() for state in torch.cuda.get_rng_state_all()
                ]
        if phase == "coarse" and callable(base_predict):
            base_kwargs = {"horizon": n, "committed": committed}
            if getattr(self.policy, "conditions_on_committed", False):
                base_kwargs["rtc_delay_steps"] = rtc_delay_steps
            values, metadata = base_predict(mapped, **base_kwargs)
        else:
            predict_kwargs = {
                "mode": "eval" if phase == "coarse" else rollout_mode,
                "committed": committed,
            }
            if getattr(self.policy, "conditions_on_committed", False):
                predict_kwargs["rtc_delay_steps"] = rtc_delay_steps
            values, metadata = self.policy.predict_action_batch(
                mapped, **predict_kwargs
            )
        if self._episode_active and not bootstrap and not warmup:
            self._schedule_counts[f"{phase}_requests"] += 1
        values = values.detach()
        action_dim = int(getattr(self.contract, "action_dim", values.shape[-1]))
        expected_steps = 32 if phase == "coarse" else 2 * n
        if values.shape != (1, expected_steps, action_dim) or not bool(
            torch.isfinite(values).all().item()
        ):
            raise ValueError(
                f"Policy output must be finite [1,{expected_steps},{action_dim}]."
            )
        normalized_gripper = None
        if self.binary_gripper:
            normalized = metadata.get("normalized_actions")
            if normalized is None:
                normalized = metadata.get("raw_model_actions")
            if (
                not torch.is_tensor(normalized)
                or normalized.ndim != 3
                or normalized.shape[0] != 1
                or normalized.shape[1] < expected_steps
                or normalized.shape[2] < action_dim
            ):
                raise ValueError(
                    "Binary control requires pre-unnormalization model actions for every physical step."
                )
            normalized_gripper = normalized[0, :expected_steps, action_dim - 1].detach()
            if not torch.isfinite(normalized_gripper).all():
                raise ValueError("Normalized gripper actions must be finite.")
        raw = None
        if getattr(self.policy, "conditions_on_committed", False):
            raw = metadata.get("raw_model_actions")
            if not torch.is_tensor(raw):
                raise ValueError("Torch policy must return raw model actions.")
            raw = raw.detach()
            if raw.shape != (1, 32, 32) or not bool(torch.isfinite(raw).all().item()):
                raise ValueError(
                    "Torch raw model actions have invalid shape or values."
                )
            # Copy the physical output and RTC raw actions in one host transfer
            # to avoid two independent CUDA synchronization points.
            value_count = values.numel()
            raw_count = raw.numel()
            parts = [values.reshape(-1), raw.reshape(-1)]
            if normalized_gripper is not None:
                parts.append(normalized_gripper)
            packed = torch.cat(parts).cpu()
            values = packed[:value_count].reshape(values.shape).numpy()
            raw = (
                packed[value_count : value_count + raw_count].reshape(raw.shape).clone()
            )
            if normalized_gripper is not None:
                normalized_gripper = packed[value_count + raw_count :].numpy()
        else:
            values = values.cpu().numpy()
            if normalized_gripper is not None:
                normalized_gripper = normalized_gripper.cpu().numpy()
        model_completed_at = time.perf_counter()
        model_latency_ms = (model_completed_at - inference_started) * 1000.0
        if (
            logging.getLogger().isEnabledFor(logging.INFO)
            and getattr(self.contract, "action_representation", None)
            == "joint_absolute"
            and committed is not None
            and self._accepted_physical is not None
        ):
            old_index = frame - self._accepted_raw_frame + action_offset
            if old_index < self._accepted_physical.shape[1]:
                change = (
                    values[0, action_offset, :7]
                    - self._accepted_physical[0, old_index, :7]
                )
                logging.info(
                    "joint prediction continuity frame=%d target_frame=%d planned_first_index=%d max_same_time_change_deg=%.4f",
                    frame,
                    target_frame,
                    action_offset,
                    float(np.rad2deg(np.abs(change).max())),
                )
        inference_hz = None
        if not bootstrap:
            self._inference_completions.append(model_completed_at)
            if len(self._inference_completions) > 1:
                elapsed = (
                    self._inference_completions[-1] - self._inference_completions[0]
                )
                if elapsed > 0:
                    inference_hz = (len(self._inference_completions) - 1) / elapsed
            logging.getLogger("flexiv.status").info(
                "inference session=%s frame=%d phase=%s completion_hz=%s model_hz=%.2f model_ms=%.2f rtc_delay_steps=%d rtc_weight=%.2f rollout_mode=%s model_version=%d",
                session,
                frame,
                phase,
                f"{inference_hz:.2f}" if inference_hz is not None else "warming up",
                1000.0 / model_latency_ms if model_latency_ms > 0 else float("nan"),
                model_latency_ms,
                rtc_delay_steps,
                float(
                    getattr(
                        getattr(self.policy, "model", self.policy),
                        "rtc_guidance_clip",
                        0.0,
                    )
                ),
                rollout_mode,
                version,
            )
            # Only the robot control owner can confirm a fixed-frame handoff.
            # A completed model call is not evidence that its block executed.
            self._pending_decisions[frame] = (
                metadata,
                version,
                None if reference is None else reference.copy(),
                raw,
                phase,
                target_frame,
                values.copy(),
            )
        if bootstrap and not warmup and raw is not None:
            self._accepted_raw = raw.clone()
            self._accepted_raw_frame = frame
            self._accepted_raw_reference = (
                reference.copy() if reference is not None else None
            )
            self._accepted_physical = values.copy()
        if bootstrap:
            selection = slice(0, n)
        elif phase == "coarse":
            end = n if self.coarse_origin_schedule else action_offset + n
            selection = slice(action_offset, end)
        else:
            selection = slice(n, 2 * n)
        actions = values[0, selection]
        if normalized_gripper is not None:
            action_context = dict(action_context or {})
            action_context["normalized_gripper_actions"] = normalized_gripper[
                selection
            ].copy()
            action_context["action_start_frame"] = frame if bootstrap else target_frame
        diagnostics = {}
        if message.get("diagnostics", False):
            raw_output = metadata.get("raw_model_actions")
            diagnostics["diagnostics"] = {
                **diagnostic_rng,
                "full_actions_physical": values.copy(),
                "raw_model_actions": (
                    raw_output.detach().float().cpu().numpy()
                    if torch.is_tensor(raw_output)
                    else None
                ),
                "committed_raw_actions": (
                    committed.detach().float().cpu().numpy()
                    if torch.is_tensor(committed)
                    else None
                ),
                "rtc_delay_steps": rtc_delay_steps,
                "rtc_guidance_mode": getattr(self.policy, "rtc_guidance_mode", None),
                "rtc_guidance_clip": getattr(
                    getattr(self.policy, "model", None), "rtc_guidance_clip", None
                ),
                "num_flow_steps": getattr(
                    getattr(self.policy, "model", None), "num_steps", None
                ),
                "selected_prediction_start": selection.start,
                "selected_prediction_stop": selection.stop,
                "target_frame": frame if bootstrap else target_frame,
            }
        return {
            "type": "inferred",
            "session": session,
            "frame": frame,
            "actions": actions,
            "action_context": action_context,
            "model_version": version,
            "phase": phase,
            "model_latency_ms": model_latency_ms,
            "inference_hz": inference_hz,
            "model_completed_at": model_completed_at,
            **diagnostics,
        }

    def serve(self, host: str, port: int, stop_event: Any = None) -> None:
        """Serve one client at a time; disconnect discards an unlabelled episode."""
        with socket.create_server((host, port), backlog=1) as listener:
            listener.settimeout(0.5)
            logging.getLogger("flexiv.startup").info(
                "Robot inference listening on %s:%d", host, port
            )
            if getattr(self.policy, "conditions_on_committed", False):
                model = getattr(self.policy, "model", self.policy)
                logging.getLogger("flexiv.startup").info(
                    "Torch RTC guidance_weight=%.2f delay_buffer_size=%d mode=%s",
                    float(getattr(model, "rtc_guidance_clip", 10.0)),
                    self.rtc_delay_buffer_size,
                    str(getattr(model, "rtc_guidance_mode", "approx")),
                )
            while stop_event is None or not stop_event.is_set():
                try:
                    connection, _ = listener.accept()
                except socket.timeout:
                    continue
                with connection:
                    try:
                        while stop_event is None or not stop_event.is_set():
                            request = receive_message(connection)
                            try:
                                reply = self.handle(request)
                            except Exception as exc:
                                logging.exception("Robot inference request failed")
                                reply = {"type": "error", "detail": str(exc)}
                            send_message(
                                connection, {"version": PROTOCOL_VERSION, **reply}
                            )
                    except (ConnectionError, OSError):
                        pass
                    finally:
                        if self.session is not None:
                            self._finish(-1, None)

    def close(self) -> None:
        """Discard an unfinished episode and close owned policy resources."""
        if self._closed:
            return
        self._closed = True
        if self.session is not None:
            self._finish(-1, None)
        if self.trainer is not None:
            self.trainer.close()
        self.policy.close()
