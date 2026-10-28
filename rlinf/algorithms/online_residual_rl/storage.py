# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Atomic, episode-scoped storage for finalized residual RL transitions."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import torch

from .config import OnlineResidualRLConfig
from .transition import (
    FeatureRecord,
    OnlineResidualTransition,
    SparseCorrection,
    validate_transition,
)


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
        success: bool | None,
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
                    "exec_physical": None
                    if row.executed_physical is None
                    else _cpu_tensor(row.executed_physical),
                    "proposal_trace": (
                        None
                        if row.proposal_trace is None
                        else _cpu_tensor(row.proposal_trace)
                    ),
                    "return": _cpu_tensor(row.discounted_return),
                    "rewards": _cpu_tensor(row.rewards),
                    "terminal": row.terminal,
                    "valid_len": row.valid_len,
                    "correction_id": correction_id,
                    "schedule_valid": row.schedule_valid,
                    "receipts_valid": row.receipts_valid,
                }
            )
        contract = asdict(self.config)
        contract["action_representation"] = self.config.action_representation.value
        payload = {
            "config": contract,
            "label_status": "pending" if success is None else "confirmed",
            "schema_version": 3,
            "proprio_normalization": self.config.proprio_normalization,
            "normalization_id": self.config.normalization_id,
            "terminal_covered": any(row.terminal for row in rows),
            "action_representation": self.config.action_representation.value,
            "action_dim": self.config.action_dim,
            "physical_action_units": (
                ["rad"] * 7 + ["m"]
                if self.config.action_dim == 8
                else ["m"] * 3 + ["rad"] * 3 + ["m"]
            ),
            "episode_id": episode_id,
            "success": success,
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
            payload.get("schema_version") not in {1, 2, 3}
            or payload.get("episode_id") != episode_id
        ):
            raise ValueError("Stored episode does not match the transition schema.")
        if payload.get("schema_version") == 3 and (
            payload.get("action_representation")
            != self.config.action_representation.value
            or payload.get("action_dim") != self.config.action_dim
        ):
            raise ValueError(
                "Stored action representation differs from transition contract."
            )
        if (
            payload.get("proprio_normalization", "raw")
            != self.config.proprio_normalization
            or payload.get("normalization_id") != self.config.normalization_id
        ):
            raise ValueError(
                "Stored proprio normalization differs from transition contract."
            )
        for row in payload["transitions"]:
            if row["exec"].shape != (self.config.td_horizon, self.config.action_dim):
                raise ValueError(
                    "Stored action shape differs from transition contract."
                )
        return payload

    def rows_from_payload(self, payload: dict) -> list[OnlineResidualTransition]:
        """Reconstruct shared features and validate stored rows before admission."""
        features = {
            key: FeatureRecord(
                obs_id=key,
                z=value["z"],
                proprio=value["p"],
                base_reference=value["base_ref"],
                committed=value["committed"],
                episode_id=value["episode_id"],
                frame_index=value["frame_index"],
                model_version=value["model_version"],
                config_version=value["config_version"],
            )
            for key, value in payload["features"].items()
        }
        corrections = {
            key: SparseCorrection(**value)
            for key, value in payload["corrections"].items()
        }
        rows = []
        for value in payload["transitions"]:
            row = OnlineResidualTransition(
                feature=features[value["obs_id"]],
                next_feature=features[value["next_obs_id"]]
                if value["next_obs_id"]
                else None,
                executed=value["exec"],
                rewards=value["rewards"],
                discounted_return=value["return"],
                terminal=value["terminal"],
                valid_len=value["valid_len"],
                correction=corrections.get(value["correction_id"]),
                proposal_trace=value.get("proposal_trace"),
                executed_physical=value.get("exec_physical"),
                schedule_valid=value.get("schedule_valid", True),
                receipts_valid=value.get("receipts_valid", True),
            )
            validate_transition(row, self.config)
            rows.append(row)
        return rows


class PendingEpisodeStore:
    """Keep stopped, unlabelled episodes outside Replay until explicit labelling."""

    def __init__(self, directory: str | Path, config: OnlineResidualRLConfig) -> None:
        self.directory = Path(directory)
        self.pending = TransitionStore(self.directory / "pending", config)
        self.config = config

    def stage(
        self, rows: Sequence[OnlineResidualTransition], terminal_frame: int
    ) -> Path:
        """Atomically retain trustworthy windows without assigning terminal rewards."""
        return self.pending.save_episode(
            rows, success=None, terminal_frame=terminal_frame
        )

    def label(self, episode_id: str, success: bool) -> Path:
        """Confirm a pending episode, publish final rewards, then remove its draft.

        A repeated confirmation with the same label is safe after a crash between
        publication and draft removal. A conflicting label is rejected.
        """
        if not isinstance(success, bool):
            raise ValueError("A boolean terminal label is required.")
        from .replay import EpisodeReplay

        draft = self.pending.load_episode(episode_id)
        replay = EpisodeReplay(self.config, storage_dir=self.directory)
        destination = replay.store.path_for(episode_id)
        if destination.exists():
            existing = replay.store.load_episode(episode_id)
            if (
                existing["success"] != success
                or existing["terminal_frame"] != draft["terminal_frame"]
            ):
                raise ValueError("Episode already has a conflicting terminal label.")
        else:
            replay.commit_episode(
                self.pending.rows_from_payload(draft),
                success=success,
                terminal_frame=draft["terminal_frame"],
            )
        self.pending.path_for(episode_id).unlink()
        return destination

    def discard(self, episode_id: str) -> None:
        """Explicitly delete a pending draft without publishing it."""
        self.pending.path_for(episode_id).unlink()
