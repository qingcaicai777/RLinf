# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Independent learner ownership, metric journals and Actor snapshot delivery."""

from __future__ import annotations

import io
import json
import multiprocessing
import os
import queue
import threading
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from rlinf.utils.logging import get_logger

from .learner import OnlineResidualLearner
from .replay import EpisodeReplay
from .storage import PendingEpisodeStore
from .training import OnlineTraining
from .transition import OnlineResidualTransition


def _serialize(value: Any) -> bytes:
    stream = io.BytesIO()
    torch.save(value, stream)
    return stream.getvalue()


def run_learner(
    training: OnlineTraining,
    events: Any,
    checkpoint_dir: Path,
    checkpoint_interval: int,
    notify_mode: Callable[[str], None] | None = None,
) -> None:
    """Drain labelled episodes and finite update credits in their owning worker."""
    if checkpoint_interval <= 0:
        raise ValueError("Checkpoint interval must be positive.")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger()
    store = training.replay.store
    seen_paths: set[Path] = set()
    last_saved = training.actor_updates
    last_checkpoint = time.monotonic()
    dirty = False
    updates_dirty = False
    stopping = False

    with (checkpoint_dir / "metrics.jsonl").open("a", encoding="utf-8") as journal:

        def record(
            metrics: Mapping[str, float], *, episode_id: str | None = None
        ) -> None:
            payload = {
                "time": time.time(),
                "learner_pid": os.getpid(),
                "critic_updates": training.critic_updates,
                "actor_updates": training.actor_updates,
                "rollout_mode": training.rollout_mode,
                "episode_id": episode_id,
                "metrics": dict(metrics),
            }
            journal.write(json.dumps(payload, allow_nan=False) + "\n")
            journal.flush()

        def admit(rows, success, terminal_frame, *, persist=True):
            nonlocal dirty
            count = training.commit_episode(
                rows, success=success, terminal_frame=terminal_frame, persist=persist
            )
            if count:
                dirty = True
                record(
                    {
                        **training.replay.statistics(),
                        "episode/new_windows": count,
                        "episode/success": int(success),
                        "episode/terminal_covered": int(
                            any(row.terminal for row in rows)
                        ),
                    },
                    episode_id=rows[0].feature.episode_id,
                )
                logger.info(
                    "Learner admitted episode %s: %d windows, success=%s",
                    rows[0].feature.episode_id,
                    count,
                    success,
                )

        def scan_confirmed() -> None:
            if store is not None:
                for path in sorted(store.directory.glob("*.pt")):
                    if path in seen_paths:
                        continue
                    payload = torch.load(path, map_location="cpu", weights_only=True)
                    payload = store.load_episode(payload["episode_id"])
                    if not isinstance(payload.get("success"), bool):
                        raise ValueError("Only confirmed episodes may enter Replay.")
                    admit(
                        store.rows_from_payload(payload),
                        payload["success"],
                        payload["terminal_frame"],
                        persist=False,
                    )
                    seen_paths.add(path)

        while True:
            # Atomic offline confirmations also survive a server restart.
            scan_confirmed()
            if notify_mode is not None:
                notify_mode(training.rollout_mode)
            can_train = (
                training.episodes >= training.initial_episodes
                and training.budget.pending_critic > 0
            )
            if stopping:
                event = ("idle", None)
            else:
                try:
                    event = events.get(timeout=0 if can_train else 0.2)
                except queue.Empty:
                    event = ("idle", None)
            stopping = stopping or event is None
            if stopping:
                scan_confirmed()
            kind, payload = ("idle", None) if stopping else event
            if kind == "metrics":
                metrics, episode_id = payload
                record(metrics, episode_id=episode_id)
            elif kind == "episode":
                rows, success, terminal_frame = torch.load(
                    io.BytesIO(payload), map_location="cpu", weights_only=False
                )
                path = (
                    store.path_for(rows[0].feature.episode_id)
                    if rows and store
                    else None
                )
                persist = path is None or not path.exists()
                if not persist:
                    saved = store.load_episode(rows[0].feature.episode_id)
                    if (
                        saved["success"] != success
                        or saved["terminal_frame"] != terminal_frame
                    ):
                        raise ValueError(
                            "Submitted episode conflicts with its stored label."
                        )
                admit(rows, success, terminal_frame, persist=persist)
                if path is not None:
                    seen_paths.add(path)
            # Service incoming episodes between optimizer steps, including while
            # consuming the credit accumulated before the warmup threshold.
            if dirty or kind == "episode":
                training.save(checkpoint_dir / "latest.pt")
                last_checkpoint = time.monotonic()
                dirty = False
                updates_dirty = False
            before = training.budget.pending_critic
            if before > 0:
                metrics = training.step()
                if training.budget.pending_critic != before:
                    updates_dirty = True
                    record(metrics)
                count = training.actor_updates
                if count and count != last_saved and count % checkpoint_interval == 0:
                    training.save(checkpoint_dir / f"actor_{count}.pt")
                    last_saved = count
            if notify_mode is not None:
                notify_mode(training.rollout_mode)
            drained = (
                training.budget.pending_critic == 0
                or training.budget.pending_critic == before
            )
            if (updates_dirty and time.monotonic() - last_checkpoint >= 10) or (
                stopping and drained
            ):
                training.save(checkpoint_dir / "latest.pt")
                last_checkpoint = time.monotonic()
                updates_dirty = False
            if stopping and drained:
                return


def _worker(
    bootstrap: bytes,
    events: Any,
    updates: Any,
    mode: Any,
    checkpoint_dir: Path,
    checkpoint_interval: int,
    device: str,
    cpu_threads: int,
) -> None:
    try:
        torch.set_num_threads(cpu_threads)
        spec = torch.load(io.BytesIO(bootstrap), map_location="cpu", weights_only=False)
        learner = OnlineResidualLearner(
            spec["config"], action_projector=spec["projector"], **spec["optimizer"]
        ).to(device)
        replay = EpisodeReplay(spec["config"], **spec["replay"])

        def publish(version, state):
            updates.put(("actor", (version, _serialize(state))))

        training = OnlineTraining(learner, replay, publish=publish, **spec["training"])
        training.load_state(spec["state"])
        # torch.load maps optimizer tensors to CPU; move slots with their parameters.
        for optimizer in (learner.actor_optimizer, learner.critic_optimizer):
            for slots in optimizer.state.values():
                for key, value in slots.items():
                    if isinstance(value, torch.Tensor) and key != "step":
                        slots[key] = value.to(device)
        updates.put(("ready", None))
        run_learner(
            training,
            events,
            checkpoint_dir,
            checkpoint_interval,
            lambda value: setattr(mode, "value", int(value == "rollout")),
        )
    except BaseException:
        updates.put(("error", traceback.format_exc()))
    finally:
        updates.put(("closed", None))


class OnlineLearnerProcess:
    """Own optimization in a spawn process; install snapshots through a mailbox.

    Construction transfers a CPU bootstrap snapshot. The parent keeps no live
    optimizer or Replay copy. Close drains accepted events and joins the child;
    worker failures surface to submit, phase reads and close.
    """

    def __init__(
        self,
        training: OnlineTraining,
        *,
        checkpoint_dir: Path,
        checkpoint_interval: int = 250,
        device: str = "cpu",
        cpu_threads: int = 1,
    ) -> None:
        if checkpoint_interval <= 0 or cpu_threads <= 0:
            raise ValueError("Learner intervals and CPU thread count must be positive.")
        self.checkpoint_dir = Path(checkpoint_dir)
        self._publish = training.publish
        self.pending_store = (
            PendingEpisodeStore(
                training.replay.store.directory, training.learner.config
            )
            if training.replay.store is not None
            else None
        )
        spec = {
            "config": training.learner.config,
            "projector": training.learner.action_projector,
            "optimizer": {
                "actor_lr": training.learner.actor_optimizer.param_groups[0]["lr"],
                "critic_lr": training.learner.critic_optimizer.param_groups[0]["lr"],
                "max_grad_norm": training.learner.max_grad_norm,
            },
            "replay": {
                "capacity": training.replay.capacity,
                "recent_window": training.replay.recent_window,
                "storage_dir": training.replay.store.directory
                if training.replay.store
                else None,
            },
            "training": {
                "initial_episodes": training.initial_episodes,
                "warmup_updates": training.warmup_remaining,
                "td_batch_size": training.td_batch_size,
                "supervision_batch_size": training.supervision_batch_size,
                "publish_interval": training.publish_interval,
            },
            "state": training.checkpoint_state(),
        }
        context = multiprocessing.get_context("spawn")
        self._events = context.Queue()
        self._updates = context.Queue()
        self._mode = context.Value(
            "b", int(training.rollout_mode == "rollout"), lock=False
        )
        self._ready = threading.Event()
        self._closed = False
        self._error: str | None = None
        self._process = context.Process(
            target=_worker,
            name="residual-learner",
            args=(
                _serialize(spec),
                self._events,
                self._updates,
                self._mode,
                self.checkpoint_dir,
                checkpoint_interval,
                device,
                cpu_threads,
            ),
        )
        self._process.start()
        self._listener = threading.Thread(
            target=self._receive, name="actor-publication"
        )
        self._listener.start()
        if not self._ready.wait(60):
            self._process.terminate()
            self._process.join()
            self._updates.put(("closed", None))
            self._listener.join()
            raise RuntimeError("Learner process did not initialize within 60 seconds.")
        if self._error is not None:
            self._process.join()
            self._listener.join()
            self._events.close()
            self._updates.close()
        self._check()

    @property
    def pid(self) -> int | None:
        """Return the optimizer-owning process ID."""
        return self._process.pid

    def _receive(self) -> None:
        try:
            while True:
                try:
                    kind, payload = self._updates.get(timeout=0.2)
                except queue.Empty:
                    if not self._process.is_alive():
                        raise RuntimeError(
                            f"Learner process {self.pid} exited without completing "
                            f"shutdown (exitcode={self._process.exitcode})."
                        )
                    continue
                if kind == "ready":
                    self._ready.set()
                elif kind == "actor" and self._publish is not None:
                    version, snapshot = payload
                    try:
                        self._publish(
                            version, torch.load(io.BytesIO(snapshot), weights_only=True)
                        )
                    except BaseException:
                        self._record_failure(traceback.format_exc())
                elif kind == "error":
                    self._record_failure(payload)
                    self._ready.set()
                elif kind == "closed":
                    if not self._closed and self._error is None:
                        self._record_failure(
                            "Learner closed without a shutdown request."
                        )
                    return
        except BaseException:
            self._record_failure(traceback.format_exc())
            self._ready.set()

    def _record_failure(self, detail: str) -> None:
        if self._error is not None:
            return
        self._error = detail
        get_logger().error("Online learner failed (pid=%s):\n%s", self.pid, detail)
        try:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            with (self.checkpoint_dir / "learner_error.log").open("a") as stream:
                stream.write(f"time={time.time()} pid={self.pid}\n{detail}\n")
        except OSError:
            get_logger().exception("Could not persist the learner failure log.")

    def _check(self) -> None:
        if (
            self._error is None
            and not self._closed
            and self._process.exitcode is not None
        ):
            self._record_failure(
                f"Learner process {self.pid} exited (exitcode={self._process.exitcode})."
            )
        if self._error is not None:
            raise RuntimeError(f"Online learner failed:\n{self._error}")

    @property
    def rollout_mode(self) -> str:
        """Read the last completed learner phase without waiting for optimizer work."""
        self._check()
        return "rollout" if self._mode.value else "base"

    def submit(
        self,
        rows: Sequence[OnlineResidualTransition],
        success: bool,
        terminal_frame: int,
    ) -> None:
        """Persist a labelled episode before enqueueing it for optimization."""
        if self._closed:
            raise RuntimeError("Online learner is closed.")
        if self.pending_store is not None and rows:
            # Collection owns this temporary replay only to assign terminal
            # rewards and atomically publish the file; it retains no rows.
            replay = EpisodeReplay(
                self.pending_store.config, storage_dir=self.pending_store.directory
            )
            path = replay.store.path_for(rows[0].feature.episode_id)
            if path.exists():
                saved = replay.store.load_episode(rows[0].feature.episode_id)
                if (
                    saved["success"] != success
                    or saved["terminal_frame"] != terminal_frame
                ):
                    raise ValueError(
                        "Submitted episode conflicts with its stored label."
                    )
            else:
                replay.commit_episode(
                    rows, success=success, terminal_frame=terminal_frame
                )
        self._check()
        # Tensor queue reduction opens one shared-memory descriptor per storage.
        # Serialize before enqueueing so long episodes use bytes and failures
        # reach the collector instead of being lost in the queue feeder thread.
        payload = _serialize((tuple(rows), success, terminal_frame))
        self._events.put_nowait(("episode", payload))

    def record_metrics(
        self, metrics: Mapping[str, float], *, episode_id: str | None = None
    ) -> None:
        """Enqueue collection diagnostics for persistent monitoring."""
        self._check()
        self._events.put_nowait(("metrics", (dict(metrics), episode_id)))

    def close(self) -> None:
        """Drain submitted episodes, persist state and join the worker once."""
        if self._closed:
            return
        self._closed = True
        self._events.put(None)
        self._process.join()
        self._listener.join()
        self._events.close()
        self._updates.close()
        self._check()
