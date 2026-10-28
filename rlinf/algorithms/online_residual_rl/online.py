# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Non-blocking rollout collection and learner-to-Actor publication."""

from __future__ import annotations

import logging
import queue
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from rlinf.utils.logging import get_logger

from .config import ActionRepresentation, OnlineResidualRLConfig
from .storage import PendingEpisodeStore
from .training import OnlineTraining
from .transition import FeatureRecord, OnlineResidualTransition, TransitionAssembler


class ActorMailbox:
    """Keep only the newest complete Actor snapshot until a request boundary."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: tuple[int, dict[str, torch.Tensor]] | None = None
        self.version = 0

    def publish(self, version: int, state: Mapping[str, torch.Tensor]) -> None:
        """Publish a detached CPU snapshot without touching the live policy."""
        snapshot = {key: value.detach().cpu().clone() for key, value in state.items()}
        with self._lock:
            if version <= self.version or (
                self._pending is not None and version <= self._pending[0]
            ):
                raise ValueError("Actor versions must increase monotonically.")
            self._pending = (version, snapshot)

    def take(self) -> tuple[int, dict[str, torch.Tensor]] | None:
        """Reserve an update for one inference request without copying its tensors."""
        with self._lock:
            value, self._pending = self._pending, None
            if value is not None:
                self.version = value[0]
            return value


@dataclass(frozen=True)
class _Decision:
    frame: int
    version: int
    metadata: Mapping[str, Any]
    tcp_reference_pose: np.ndarray | None = None


def _receipt_in_decision_frame(
    info: Mapping[str, Any], reference: np.ndarray | None
) -> Mapping[str, Any]:
    """Re-express a final TCP target relative to one decision observation."""
    if reference is None:
        return info

    def target_delta(value: Any, name: str) -> np.ndarray:
        target = np.asarray(value, dtype=np.float64)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError(f"{name} must be finite [7].")
        target_rotation = Rotation.from_quat(
            [target[4], target[5], target[6], target[3]]
        )
        return np.concatenate(
            (
                target[:3] - reference[:3],
                (target_rotation * reference_rotation.inv()).as_rotvec(),
            )
        )

    reference_rotation = Rotation.from_quat(
        [reference[4], reference[5], reference[6], reference[3]]
    )
    receipt = np.asarray(info["action_receipt_physical"], dtype=np.float32).copy()
    if receipt.shape != (7,) or not np.isfinite(receipt).all():
        raise ValueError("Final physical receipt must be finite [7].")
    receipt[:6] = target_delta(info["tcp_target_pose"], "Final TCP target")
    result = {**info, "action_receipt_physical": receipt}
    proposal_target = info.get("policy_proposal_tcp_target_pose")
    proposal_width = info.get("policy_proposal_gripper_width_m")
    if proposal_target is not None and proposal_width is not None:
        try:
            proposal_delta = np.concatenate(
                (
                    target_delta(proposal_target, "Policy proposal target"),
                    [float(proposal_width)],
                )
            ).astype(np.float32)
            result["policy_proposal_physical"] = proposal_delta
        except (ValueError, TypeError):
            result["policy_proposal_valid"] = False
            result["policy_proposal_physical"] = None
    return result


class EpisodeCollector:
    """Assemble finalized 12-frame windows away from the control thread.

    The control owner only enqueues immutable decision and command snapshots.
    The worker waits for real receipts and the t+12 feature, then sends one
    labelled episode to the learner. Unknown or mismatched receipts are dropped.
    """

    def __init__(
        self,
        config: OnlineResidualRLConfig,
        encode_actions: Callable[[torch.Tensor], torch.Tensor],
        commit_episode: Callable[[Sequence[OnlineResidualTransition], bool, int], None],
        *,
        episode_id: str,
        config_version: str | None = None,
        queue_capacity: int = 1024,
        pending_store: PendingEpisodeStore | None = None,
        record_metrics: Callable[[Mapping[str, float]], None] | None = None,
    ) -> None:
        self.config = config
        self.pending_store = pending_store
        self.record_metrics = record_metrics
        self._pending_episode = None
        self._counts = {
            "decisions": 0,
            "receipts": 0,
            "invalid_committed": 0,
            "invalid_features": 0,
            "invalid_windows": 0,
            "no_execution": 0,
        }
        self.encode_actions = encode_actions
        self.commit_episode = commit_episode
        self.episode_id = episode_id
        representation = ActionRepresentation(
            getattr(config, "action_representation", "eef_delta")
        )
        self.config_version = (
            config_version or f"flexiv-{representation.value.replace('_', '-')}-v1"
        )
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue(queue_capacity)
        self._thread = threading.Thread(target=self._run, name="residual-collector")
        self._thread.start()
        self._finished = False
        self._error: BaseException | None = None
        self._error_event = threading.Event()
        self._logger = get_logger()

    def decision(
        self,
        frame: int,
        metadata: Mapping[str, Any],
        version: int,
        *,
        tcp_reference_pose: np.ndarray | None = None,
    ) -> None:
        """Queue a policy decision after its fixed-frame handoff is accepted."""
        reference = None
        if (
            tcp_reference_pose is not None
            and self.config.action_representation is ActionRepresentation.EEF_DELTA
        ):
            reference = np.asarray(tcp_reference_pose, dtype=np.float64).copy()
            if reference.shape != (7,) or not np.isfinite(reference).all():
                raise ValueError("Decision TCP reference must be finite [7].")
        self._put(
            "decision",
            _Decision(
                frame,
                version,
                metadata,
                reference,
            ),
        )

    def receipt(self, frame: int, info: Mapping[str, Any]) -> None:
        """Queue the final accepted physical command and intervention masks."""
        copied = {
            key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
            for key, value in info.items()
        }
        copied["frame_index"] = frame
        self._put("receipt", copied)

    def _put(self, kind: str, value: Any) -> None:
        if self._finished:
            raise RuntimeError("Episode collector is already finished.")
        if self._error_event.is_set():
            raise RuntimeError("Episode collection failed.") from self._error
        try:
            self._queue.put_nowait((kind, value))
        except queue.Full as exc:
            raise RuntimeError(
                "Episode collector fell behind the control loop."
            ) from exc

    def finish(
        self, terminal_frame: int, success: bool | None, *, discard: bool = False
    ) -> None:
        """Flush stopped commands; an unknown label stages a pending episode."""
        if self._finished:
            return
        self._finished = True
        while True:
            if self._error_event.is_set():
                break
            try:
                self._queue.put(
                    ("finish", (terminal_frame, success, discard)), timeout=0.1
                )
                break
            except queue.Full:
                continue
        self._thread.join()
        if self._error is not None:
            raise RuntimeError("Episode collection failed.") from self._error

    def label(self, success: bool) -> None:
        """Confirm a stopped draft; the learner discovers persisted confirmations."""
        if not isinstance(success, bool) or not self._finished:
            raise ValueError("Finish the episode before supplying a boolean label.")
        if self._pending_episode is None:
            raise RuntimeError("No pending episode remains to label.")
        rows, terminal_frame = self._pending_episode
        if self.pending_store is not None:
            self.pending_store.label(self.episode_id, success)
        else:
            self.commit_episode(rows, success, terminal_frame)
        self._pending_episode = None

    def _run(self) -> None:
        assembler = TransitionAssembler(self.config)
        decisions: dict[int, _Decision] = {}
        features: dict[int, FeatureRecord] = {}
        receipts: dict[int, Mapping[str, Any]] = {}
        rows: list[OnlineResidualTransition] = []
        try:
            while True:
                kind, value = self._queue.get()
                if kind == "receipt":
                    self._counts["receipts"] += 1
                    receipts[int(value["frame_index"])] = value
                elif kind == "decision":
                    self._counts["decisions"] += 1
                    decisions[value.frame] = value
                else:
                    terminal_frame, success, discard = value
                    if not discard and terminal_frame >= 0:
                        self._advance(
                            assembler,
                            decisions,
                            features,
                            receipts,
                            rows,
                            terminal_frame=terminal_frame,
                        )
                        if rows:
                            if success is not None:
                                self.commit_episode(rows, success, terminal_frame)
                            else:
                                self._pending_episode = (tuple(rows), terminal_frame)
                                if self.pending_store is not None:
                                    self.pending_store.stage(rows, terminal_frame)
                                logging.getLogger("flexiv.status").info(
                                    "Episode %s awaits terminal labelling",
                                    self.episode_id,
                                )
                    metrics = {
                        "collection/" + key: value
                        for key, value in self._counts.items()
                    }
                    metrics.update(
                        {
                            "collection/valid_windows": len(rows),
                            "collection/unresolved_windows": len(decisions),
                            "collection/pending_label": int(
                                success is None and not discard and bool(rows)
                            ),
                            "collection/discarded": int(discard),
                        }
                    )
                    logging.getLogger("flexiv.status").info(
                        "Episode %s collection: %s", self.episode_id, metrics
                    )
                    if not rows and not discard:
                        logging.getLogger("flexiv.status").warning(
                            "No RL transitions saved for episode %s: decisions=%d receipts=%d. "
                            "Check Fine entry/handoffs and the scheduling summary; this episode does not count toward warmup.",
                            self.episode_id,
                            self._counts["decisions"],
                            self._counts["receipts"],
                        )
                    if self.record_metrics is not None:
                        self.record_metrics(metrics)
                    return
                self._advance(assembler, decisions, features, receipts, rows)
        except BaseException as exc:
            self._error = exc
            self._error_event.set()

    def _advance(
        self,
        assembler: TransitionAssembler,
        decisions: dict[int, _Decision],
        features: dict[int, FeatureRecord],
        receipts: dict[int, Mapping[str, Any]],
        rows: list[OnlineResidualTransition],
        *,
        terminal_frame: int | None = None,
    ) -> None:
        n = self.config.inference_horizon
        for frame, decision in list(decisions.items()):
            if frame in features or any(frame + i not in receipts for i in range(n)):
                continue
            if any(
                not receipts[frame + i].get("schedule_valid", True)
                or not receipts[frame + i].get("receipt_valid", True)
                for i in range(n)
            ):
                self._counts["invalid_committed"] += 1
                self._logger.warning(
                    "Discarding decision %s: invalid committed prefix", frame
                )
                del decisions[frame]
                continue
            inputs = decision.metadata.get("forward_inputs")
            if not isinstance(inputs, Mapping):
                self._counts["invalid_features"] += 1
                self._logger.warning("Missing residual features at frame %s", frame)
                del decisions[frame]
                continue
            try:
                physical = torch.stack(
                    [
                        torch.as_tensor(
                            _receipt_in_decision_frame(
                                receipts[frame + i], decision.tcp_reference_pose
                            )["action_receipt_physical"],
                            dtype=torch.float32,
                        )
                        for i in range(n)
                    ]
                )
                committed = (
                    self.encode_actions(physical[None])[0].detach().cpu().float()
                )
                if (
                    physical.shape != (n, self.config.action_dim)
                    or not torch.isfinite(physical).all()
                ):
                    raise ValueError(
                        "Committed receipts must be finite with the configured shape."
                    )
            except (ValueError, KeyError, RuntimeError) as exc:
                self._counts["invalid_committed"] += 1
                self._logger.warning("Discarding decision %s: %s", frame, exc)
                del decisions[frame]
                continue

            def one(name: str) -> torch.Tensor:
                return torch.as_tensor(inputs[name])[0].detach().cpu().clone()

            try:
                feature = FeatureRecord(
                    obs_id=f"{self.episode_id}:{frame}",
                    z=one("z_rl"),
                    proprio=one("proprio"),
                    base_reference=one("base_ref"),
                    committed=committed,
                    episode_id=self.episode_id,
                    frame_index=frame,
                    model_version=decision.version,
                    config_version=self.config_version,
                )
                # Keep the Actor's complete proposal. The real executed prefix is
                # stored independently in FeatureRecord.committed.
                proposal = one("actor_output")
                assembler.begin(feature, proposal)
                features[frame] = feature
            except (ValueError, KeyError, RuntimeError, IndexError) as exc:
                self._counts["invalid_features"] += 1
                self._logger.warning(
                    "Discarding invalid features at frame %s: %s", frame, exc
                )
                assembler.discard(f"{self.episode_id}:{frame}")
                del decisions[frame]

        for frame, feature in list(features.items()):
            if frame not in decisions:
                continue
            try:
                available = [
                    _receipt_in_decision_frame(
                        receipts[frame + i], decisions[frame].tcp_reference_pose
                    )
                    for i in range(2 * n)
                    if frame + i in receipts
                ]
            except (ValueError, KeyError, RuntimeError) as exc:
                self._counts["invalid_windows"] += 1
                self._logger.warning(
                    "Discarding invalid receipt window at frame %s: %s", frame, exc
                )
                assembler.discard(feature.obs_id)
                del decisions[frame]
                continue
            if len(available) != 2 * n and terminal_frame is None:
                continue
            if terminal_frame is not None:
                available = available[: max(0, terminal_frame - frame + 1)]
            if len(available) <= n:
                self._counts["no_execution"] += 1
                assembler.discard(feature.obs_id)
                del decisions[frame]
                continue
            end = frame + len(available) - 1
            terminal = terminal_frame is not None and end == terminal_frame
            successor = None if terminal else features.get(frame + 2 * n)
            if not terminal and successor is None:
                continue
            try:
                row = assembler.complete_from_receipts(
                    feature.obs_id,
                    step_infos=available,
                    encode_actions=lambda actions: self.encode_actions(actions[None])[
                        0
                    ],
                    terminal=terminal,
                    next_feature=successor,
                )
            except (ValueError, KeyError, RuntimeError) as exc:
                self._counts["invalid_windows"] += 1
                self._logger.warning("Discarding invalid frame %s: %s", frame, exc)
                assembler.discard(feature.obs_id)
            else:
                rows.append(row)
            del decisions[frame]


class OnlineLearnerThread:
    """Own all optimizer and replay mutations outside the control loop."""

    def __init__(
        self,
        training: OnlineTraining,
        *,
        checkpoint_dir: Path,
        checkpoint_interval: int = 250,
    ) -> None:
        self.training = training
        self.pending_store = (
            PendingEpisodeStore(
                training.replay.store.directory, training.learner.config
            )
            if training.replay.store is not None
            else None
        )
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self._queue: queue.Queue[Any] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="residual-learner")
        self._closed = False
        self._error: BaseException | None = None
        self._error_event = threading.Event()
        self._thread.start()

    def submit(
        self,
        rows: Sequence[OnlineResidualTransition],
        success: bool,
        terminal_frame: int,
    ) -> None:
        """Submit one finalized, human-labelled episode without waiting."""
        if self._closed:
            raise RuntimeError("Online learner is closed.")
        if self._error_event.is_set():
            raise RuntimeError("Online learner failed.") from self._error
        from .learner_process import _serialize

        payload = _serialize((tuple(rows), success, terminal_frame))
        self._queue.put_nowait(("episode", payload))

    def record_metrics(
        self, metrics: Mapping[str, float], *, episode_id: str | None = None
    ) -> None:
        """Queue rollout diagnostics for the learner's persistent metric journal."""
        self._queue.put_nowait(("metrics", (dict(metrics), episode_id)))

    @property
    def rollout_mode(self) -> str:
        """Return the training phase without waiting for learner work."""
        return self.training.rollout_mode

    def _run(self) -> None:
        from .learner_process import run_learner

        try:
            run_learner(
                self.training,
                self._queue,
                self.checkpoint_dir,
                self.checkpoint_interval,
            )
        except BaseException as exc:
            self._error = exc
            self._error_event.set()

    def close(self) -> None:
        """Drain accepted episodes and stop the learner exactly once."""
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join()
        if self._error is not None:
            raise RuntimeError("Online learner failed.") from self._error


class OperatorConsole:
    """Read explicit Enter-terminated operator switches without blocking ticks."""

    def __init__(self) -> None:
        self._events: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._read, name="operator-console", daemon=True
        )
        self._thread.start()

    def _read(self) -> None:
        for line in sys.stdin:
            command = line.strip().lower()
            if command in {"on", "off", "success", "failure", "stop", "next"}:
                self._events.put(command)

    def poll(self) -> list[str]:
        """Return all pending switch and terminal-label events in order."""
        result = []
        while True:
            try:
                result.append(self._events.get_nowait())
            except queue.Empty:
                return result

    def wait_next(self, stop_event: threading.Event | None = None) -> bool:
        """Wait between stopped episodes for explicit operator re-arming."""
        while True:
            if stop_event is not None and stop_event.is_set():
                return False
            try:
                command = self._events.get(timeout=0.1)
            except queue.Empty:
                continue
            if command in {"next", "stop"}:
                return command == "next"
