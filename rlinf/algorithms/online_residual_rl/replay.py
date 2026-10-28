# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Episode-atomic replay with separate supervision sampling pools."""

from __future__ import annotations

import copy
import random
import threading
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Sequence

import torch

from .config import OnlineResidualRLConfig
from .losses import discounted_return
from .storage import TransitionStore
from .transition import (
    FeatureRecord,
    OnlineResidualTransition,
    SparseCorrection,
    validate_transition,
)


@dataclass(frozen=True)
class _ReplayRow:
    obs_id: str
    next_obs_id: str | None
    correction_id: str | None
    executed: torch.Tensor
    rewards: torch.Tensor
    discounted_return: torch.Tensor
    terminal: bool
    valid_len: int
    schedule_valid: bool
    receipts_valid: bool


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
        self._rows: list[_ReplayRow] = []
        self._features: dict[str, FeatureRecord] = {}
        self._corrections: dict[str, SparseCorrection] = {}
        self._episodes: set[str] = set()
        self._lock = threading.RLock()
        self._random = random.Random(seed)
        self.store = TransitionStore(storage_dir, config) if storage_dir else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._rows)

    def state_dict(self) -> dict:
        """Capture ID tables, deduplication and sampler state together."""
        with self._lock:
            return copy.deepcopy(
                {
                    "schema_version": 2,
                    "config": asdict(self.config),
                    "rows": self._rows,
                    "features": self._features,
                    "corrections": self._corrections,
                    "episodes": self._episodes,
                    "rng": self._random.getstate(),
                }
            )

    def load_state_dict(self, state: dict) -> None:
        """Restore indexed replay; accept legacy checkpoints with full rows."""
        restored = copy.deepcopy(state)
        if (
            "config" in restored
            and OnlineResidualRLConfig(**restored["config"]) != self.config
        ):
            raise ValueError("Replay action contract differs from this learner.")
        candidate = EpisodeReplay(
            self.config, capacity=self.capacity, recent_window=self.recent_window
        )
        if restored.get("schema_version") == 2:
            candidate._features = restored["features"]
            candidate._corrections = restored["corrections"]
            candidate._rows = restored["rows"][-self.capacity :]
            for indexed in candidate._rows:
                validate_transition(candidate._materialize(indexed), self.config)
            candidate._trim()
        else:
            for row in restored["rows"][-self.capacity :]:
                validate_transition(row, self.config)
                candidate._rows.append(candidate._index(row))
        with self._lock:
            self._random.setstate(restored["rng"])
            self._rows, self._features, self._corrections = (
                candidate._rows,
                candidate._features,
                candidate._corrections,
            )
            self._episodes = restored["episodes"]

    def _index(self, row: OnlineResidualTransition) -> _ReplayRow:
        for feature in (row.feature, row.next_feature):
            if feature is not None:
                existing = self._features.get(feature.obs_id)
                if existing is not None:
                    for field in fields(feature):
                        old, new = (
                            getattr(existing, field.name),
                            getattr(feature, field.name),
                        )
                        same = (
                            torch.equal(old, new)
                            if isinstance(old, torch.Tensor)
                            else old == new
                        )
                        if not same:
                            raise ValueError(
                                "Conflicting payloads for one observation ID."
                            )
                else:
                    self._features[feature.obs_id] = feature
        correction_id = row.feature.obs_id if row.correction is not None else None
        if correction_id is not None:
            self._corrections[correction_id] = row.correction
        return _ReplayRow(
            obs_id=row.feature.obs_id,
            next_obs_id=row.next_feature.obs_id if row.next_feature else None,
            correction_id=correction_id,
            executed=row.executed,
            rewards=row.rewards,
            discounted_return=row.discounted_return,
            terminal=row.terminal,
            valid_len=row.valid_len,
            schedule_valid=row.schedule_valid,
            receipts_valid=row.receipts_valid,
        )

    def _materialize(self, row: _ReplayRow) -> OnlineResidualTransition:
        return OnlineResidualTransition(
            feature=self._features[row.obs_id],
            next_feature=self._features[row.next_obs_id] if row.next_obs_id else None,
            correction=self._corrections[row.correction_id]
            if row.correction_id
            else None,
            executed=row.executed,
            rewards=row.rewards,
            discounted_return=row.discounted_return,
            terminal=row.terminal,
            valid_len=row.valid_len,
            schedule_valid=row.schedule_valid,
            receipts_valid=row.receipts_valid,
        )

    def _trim(self) -> None:
        self._rows = self._rows[-self.capacity :]
        feature_ids = {
            key for row in self._rows for key in (row.obs_id, row.next_obs_id) if key
        }
        correction_ids = {row.correction_id for row in self._rows if row.correction_id}
        self._features = {
            key: value for key, value in self._features.items() if key in feature_ids
        }
        self._corrections = {
            key: value
            for key, value in self._corrections.items()
            if key in correction_ids
        }

    def transitions(self) -> list[OnlineResidualTransition]:
        """Return a detached resident snapshot, sharing feature copies by ID."""
        with self._lock:
            return copy.deepcopy([self._materialize(row) for row in self._rows])

    def statistics(self) -> dict[str, float]:
        """Return resident window, feature and correction counts."""
        with self._lock:
            rows = [self._materialize(row) for row in self._rows]
            return {
                "replay/windows": len(rows),
                "replay/features": len(self._features),
                "replay/episodes": len(self._episodes),
                "replay/relative_windows": sum(row.relative_q_eligible for row in rows),
                "replay/terminal_windows": sum(row.terminal for row in rows),
                "replay/human_windows": sum(
                    row.correction is not None and bool(row.correction.human_mask.any())
                    for row in rows
                ),
                "replay/pair_windows": sum(
                    row.correction is not None and bool(row.correction.pair_mask.any())
                    for row in rows
                ),
                "replay/human_coordinates": sum(
                    int(value.human_mask.sum()) for value in self._corrections.values()
                ),
                "replay/pair_coordinates": sum(
                    int(value.pair_mask.sum()) for value in self._corrections.values()
                ),
            }

    def commit_episode(
        self,
        rows: Sequence[OnlineResidualTransition],
        *,
        success: bool,
        terminal_frame: int,
        persist: bool = True,
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
            # Build tables privately so conflicting IDs cannot partially publish.
            candidate = EpisodeReplay(
                self.config, capacity=self.capacity, recent_window=self.recent_window
            )
            candidate._features = self._features.copy()
            candidate._corrections = self._corrections.copy()
            indexed = [candidate._index(row) for row in prepared]
            if self.store is not None and persist:
                self.store.save_episode(
                    prepared, success=success, terminal_frame=terminal_frame
                )
            self._features, self._corrections = (
                candidate._features,
                candidate._corrections,
            )
            self._rows.extend(indexed)
            self._trim()
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
            for indexed in self._rows[-self.recent_window :]:
                row = self._materialize(indexed)
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
