# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Episode-atomic replay with separate supervision sampling pools."""

from __future__ import annotations

import copy
import random
import threading
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import torch

from .config import OnlineResidualRLConfig
from .losses import discounted_return
from .storage import TransitionStore
from .transition import OnlineResidualTransition, validate_transition


class EpisodeReplay:
    """Publish an episode only after its terminal label and receipts are final."""

    def __init__(
        self,
        config: OnlineResidualRLConfig,
        *,
        capacity: int = 200000,
        recent_window: int = 10000,
        seed: int = 0,
        storage_dir: str | Path | None = None,
    ) -> None:
        if capacity <= 0 or recent_window <= 0:
            raise ValueError("Replay capacity and recent window must be positive.")
        self.config = config
        self.capacity = capacity
        self.recent_window = recent_window
        self._rows: list[OnlineResidualTransition] = []
        self._episodes: set[str] = set()
        self._lock = threading.RLock()
        self._random = random.Random(seed)
        self.store = TransitionStore(storage_dir, config) if storage_dir else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._rows)

    def state_dict(self) -> dict:
        """Capture replay, deduplication and sampler state together."""
        with self._lock:
            return copy.deepcopy(
                {
                    "rows": self._rows,
                    "episodes": self._episodes,
                    "rng": self._random.getstate(),
                }
            )

    def load_state_dict(self, state: dict) -> None:
        """Validate restored rows before replacing the replay."""
        restored = copy.deepcopy(state)
        for row in restored["rows"]:
            validate_transition(row, self.config)
        with self._lock:
            self._random.setstate(restored["rng"])
            self._rows = restored["rows"][-self.capacity :]
            self._episodes = restored["episodes"]

    def commit_episode(
        self,
        rows: Sequence[OnlineResidualTransition],
        *,
        success: bool,
        terminal_frame: int,
    ) -> int:
        """Validate all rows, assign the single physical terminal reward, then publish.

        ``terminal_frame`` is the zero-based index of the last executed frame.
        Overlapping windows each include that frame's reward in their return;
        they do not create additional physical reward events.
        """
        if not isinstance(success, bool):
            raise ValueError("An explicit boolean human terminal label is required.")
        if not rows:
            return 0
        episode = rows[0].feature.episode_id
        prepared = []
        ids = set()
        frames = set()
        for source in rows:
            row = copy.deepcopy(source)
            feature = row.feature
            if (
                feature.episode_id != episode
                or feature.obs_id in ids
                or feature.frame_index in frames
            ):
                raise ValueError(
                    "Episode rows must have unique decisions from one episode."
                )
            ids.add(feature.obs_id)
            frames.add(feature.frame_index)
            end = feature.frame_index + row.valid_len - 1
            if end > terminal_frame or row.terminal != (end == terminal_frame):
                raise ValueError(
                    "Window termination must agree with the physical terminal frame."
                )
            reward = torch.zeros(self.config.td_horizon, dtype=torch.float32)
            if row.terminal and success:
                reward[row.valid_len - 1] = 1.0
            valid = torch.arange(self.config.td_horizon) < row.valid_len
            returns = discounted_return(
                reward[None], valid[None], gamma=self.config.gamma
            )[0]
            row = replace(row, rewards=reward, discounted_return=returns)
            validate_transition(row, self.config)
            prepared.append(row)
        with self._lock:
            if episode in self._episodes:
                return 0
            if self.store is not None:
                self.store.save_episode(
                    prepared, success=success, terminal_frame=terminal_frame
                )
            self._rows = (self._rows + prepared)[-self.capacity :]
            self._episodes.add(episode)
        return len(prepared)

    def sample(self, count: int, *, kind: str = "td") -> list[OnlineResidualTransition]:
        """Sample supervision independently, spreading draws across episode/time bins."""
        if count <= 0:
            raise ValueError("Sample count must be positive.")
        if kind not in {"td", "rank", "relative", "ordinary", "human"}:
            raise ValueError(f"Unknown replay pool: {kind}")
        with self._lock:
            groups = {}
            for row in self._rows[-self.recent_window :]:
                human = row.correction is not None and bool(
                    row.correction.human_mask.any()
                )
                pair = row.correction is not None and bool(
                    row.correction.pair_mask.any()
                )
                if (
                    (kind == "rank" and not pair)
                    or (kind == "relative" and not row.relative_q_eligible)
                    or (kind == "human" and not human)
                    or (kind == "ordinary" and human)
                ):
                    continue
                key = (
                    row.feature.episode_id,
                    row.feature.frame_index // self.config.td_horizon,
                )
                groups.setdefault(key, []).append(row)
            keys = list(groups)
            if not keys:
                return []
            result = []
            while len(result) < count:
                self._random.shuffle(keys)
                for key in keys:
                    result.append(self._random.choice(groups[key]))
                    if len(result) == count:
                        break
            return copy.deepcopy(result)
