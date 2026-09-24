# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Atomic, episode-scoped storage for finalized residual RL transitions."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import torch

from .config import OnlineResidualRLConfig
from .transition import FeatureRecord, OnlineResidualTransition, validate_transition


def _cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().clone()


def _feature_payload(feature: FeatureRecord) -> dict:
    return {
        "z": _cpu_tensor(feature.z),
        "p": _cpu_tensor(feature.proprio),
        "base_ref": _cpu_tensor(feature.base_reference),
        "committed": _cpu_tensor(feature.committed),
        "episode_id": feature.episode_id,
        "frame_index": feature.frame_index,
        "model_version": feature.model_version,
        "config_version": feature.config_version,
    }


class TransitionStore:
    """Write each labelled episode once, with features and corrections by ID.

    Files are plain tensor/dict ``torch.save`` payloads. A file is published
    atomically only after all rows pass the transition contract; incomplete
    episodes and unknown command receipts never become visible to readers.
    """

    def __init__(self, directory: str | Path, config: OnlineResidualRLConfig) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config = config

    def path_for(self, episode_id: str) -> Path:
        """Return the stable path without using an episode ID as a filename."""
        digest = hashlib.sha256(episode_id.encode("utf-8")).hexdigest()
        return self.directory / f"{digest}.pt"

    def save_episode(
        self,
        rows: Sequence[OnlineResidualTransition],
        *,
        success: bool,
        terminal_frame: int,
    ) -> Path:
        """Save validated, final command receipts in the transition schema."""
        if not rows:
            raise ValueError("An episode must contain at least one transition.")
        episode_id = rows[0].feature.episode_id
        features: dict[str, dict] = {}
        corrections: dict[str, dict] = {}
        transitions: list[dict] = []
        for row in rows:
            validate_transition(row, self.config)
            if row.feature.episode_id != episode_id:
                raise ValueError("All transitions must belong to one episode.")
            for feature in (row.feature, row.next_feature):
                if feature is not None and feature.obs_id not in features:
                    features[feature.obs_id] = _feature_payload(feature)
            correction_id = None
            if row.correction is not None:
                correction_id = row.feature.obs_id
                corrections[correction_id] = {
                    "human_mask": _cpu_tensor(row.correction.human_mask),
                    "pair_mask": _cpu_tensor(row.correction.pair_mask),
                    "proposal_values": _cpu_tensor(row.correction.proposal_values),
                }
            transitions.append(
                {
                    "obs_id": row.feature.obs_id,
                    "next_obs_id": row.next_feature.obs_id
                    if row.next_feature
                    else None,
                    "exec": _cpu_tensor(row.executed),
                    "return": _cpu_tensor(row.discounted_return),
                    "rewards": _cpu_tensor(row.rewards),
                    "terminal": row.terminal,
                    "valid_len": row.valid_len,
                    "correction_id": correction_id,
                    "schedule_valid": row.schedule_valid,
                    "receipts_valid": row.receipts_valid,
                }
            )
        payload = {
            "schema_version": 1,
            "episode_id": episode_id,
            "success": bool(success),
            "terminal_frame": int(terminal_frame),
            "features": features,
            "transitions": transitions,
            "corrections": corrections,
        }
        destination = self.path_for(episode_id)
        if destination.exists():
            raise FileExistsError(f"Episode {episode_id!r} is already on disk.")
        with tempfile.NamedTemporaryFile(dir=self.directory, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            os.link(temporary, destination)  # Publish without replacing an episode.
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def load_episode(self, episode_id: str) -> dict:
        """Read one finalized episode as feature, transition and correction tables."""
        payload = torch.load(
            self.path_for(episode_id), map_location="cpu", weights_only=True
        )
        if (
            payload.get("schema_version") != 1
            or payload.get("episode_id") != episode_id
        ):
            raise ValueError("Stored episode does not match the transition schema.")
        return payload
