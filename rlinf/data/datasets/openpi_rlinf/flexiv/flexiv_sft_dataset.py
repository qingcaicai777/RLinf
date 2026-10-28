# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Load compact single-arm Flexiv demonstrations for OpenPI SFT.

Absolute TCP recordings become deltas from the current measured TCP. Absolute
joint recordings retain commanded joint targets and gripper width unchanged.
Both modes construct episode-local future chunks before OpenPI transforms.
"""

from __future__ import annotations

import bisect
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from rlinf.data.storage.lerobot import resolve_lerobot_dataset_root


def absolute_tcp_chunk_to_control_delta(
    measured_tcp: np.ndarray, absolute_actions: np.ndarray
) -> np.ndarray:
    """Convert absolute targets to start-state deltas in the TCP control frame.

    Args:
        measured_tcp: Current measured TCP pose with final dimension six.
        absolute_actions: Target xyz+rotvec plus absolute gripper width, with
            final dimension seven. Any leading dimensions are supported.

    Returns:
        Actions with ``target_xyz - measured_xyz`` and the rotation vector of
        ``R_target @ R_measured.inv()``. Gripper width is passed through. The
        Every row in the chunk uses the same measured start pose as its
        reference; this is not a chain of consecutive deltas. The source and
        robot command frame must be the same; this function does not perform a
        separate world/base/tool frame conversion.
    """
    reference = np.array(measured_tcp, dtype=np.float64, copy=True)
    targets = np.array(absolute_actions, dtype=np.float64, copy=True)
    if reference.shape[-1:] != (6,):
        raise ValueError(
            f"measured_tcp must end in dimension 6, got {reference.shape}."
        )
    if targets.shape[-1:] != (7,):
        raise ValueError(
            f"absolute_actions must end in dimension 7, got {targets.shape}."
        )
    try:
        reference = np.broadcast_to(reference, targets.shape[:-1] + (6,)).copy()
    except ValueError as exc:
        raise ValueError(
            "measured_tcp leading dimensions must broadcast to the action chunk."
        ) from exc
    if not np.isfinite(reference).all() or not np.isfinite(targets).all():
        raise ValueError("TCP poses and actions must contain only finite values.")

    result = targets.copy()
    result[..., :3] -= reference[..., :3]
    reference_rotation = Rotation.from_rotvec(reference[..., 3:6].reshape(-1, 3))
    target_rotation = Rotation.from_rotvec(targets[..., 3:6].reshape(-1, 3))
    result[..., 3:6] = (
        (target_rotation * reference_rotation.inv())
        .as_rotvec()
        .reshape(targets.shape[:-1] + (3,))
    )
    return result


def absolute_tcp_chunk_to_world_delta(
    measured_tcp: np.ndarray, absolute_actions: np.ndarray
) -> np.ndarray:
    """Backward-compatible alias for the control-frame delta conversion."""
    return absolute_tcp_chunk_to_control_delta(measured_tcp, absolute_actions)


def future_action_indices(length: int, horizon: int) -> np.ndarray:
    """Return episode-local future indices with terminal-frame padding."""
    if length <= 0:
        raise ValueError("Episode length must be positive.")
    if horizon <= 0:
        raise ValueError("Action horizon must be positive.")
    starts = np.arange(length, dtype=np.int64)[:, None]
    offsets = np.arange(horizon, dtype=np.int64)[None, :]
    return np.minimum(starts + offsets, length - 1)


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _decode_rgb(encoded: bytes) -> np.ndarray:
    from PIL import Image

    if not isinstance(encoded, (bytes, bytearray, memoryview)):
        raise TypeError(f"Expected encoded image bytes, got {type(encoded).__name__}.")
    with Image.open(io.BytesIO(encoded)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def validate_flexiv_metadata(
    metadata: dict[str, Any],
    action_representation: str = "eef_delta",
    *,
    require_images: bool = True,
) -> None:
    """Validate the physical recording contract before transforms or statistics."""
    expected = {
        "observation.eef_state": [6],
        "observation.joint_state": [7],
        "observation.gripper_state": [1],
        "action": [8 if action_representation == "joint_absolute" else 7],
    }
    features = metadata.get("features", {})
    for key, shape in expected.items():
        if features.get(key, {}).get("shape") != shape:
            raise ValueError(f"Flexiv SFT {key} must have shape {shape}.")
    source = (
        "joint_absolute"
        if action_representation == "joint_absolute"
        else "eef_absolute"
    )
    if metadata.get("action_representation") != source:
        raise ValueError(f"Flexiv SFT expects source action_representation={source!r}.")
    image_keys = set(metadata.get("image_keys", []))
    required_images = {"observation.image", "observation.wrist_image"}
    if require_images and not required_images.issubset(image_keys):
        raise ValueError("Flexiv SFT requires main and wrist RGB observations.")


class FlexivSftDataset(torch.utils.data.Dataset):
    """Expose compact Flexiv frames in the canonical OpenPI SFT schema."""

    def __init__(
        self,
        *,
        data_path: str,
        action_horizon: int,
        cache_dir: str | None = None,
        action_representation: str = "eef_delta",
    ) -> None:
        from datasets import load_dataset

        root = resolve_lerobot_dataset_root(data_path)
        info_path = root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Flexiv metadata not found at {info_path}.")
        with info_path.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        if action_representation not in {"eef_delta", "joint_absolute"}:
            raise ValueError("Unsupported Flexiv training action representation.")
        self._action_representation = action_representation
        validate_flexiv_metadata(metadata, action_representation)

        episode_rows = _read_json_lines(root / "meta" / "episodes.jsonl")
        episode_rows.sort(key=lambda row: int(row["episode_index"]))
        lengths = [int(row["length"]) for row in episode_rows]
        if any(length <= 0 for length in lengths):
            raise ValueError(
                "Every Flexiv SFT episode must contain at least one frame."
            )
        self._episode_ends = np.cumsum(lengths, dtype=np.int64).tolist()
        self._episode_starts = [0, *self._episode_ends[:-1]]

        task_rows = _read_json_lines(root / "meta" / "tasks.jsonl")
        self._tasks = {int(row["task_index"]): str(row["task"]) for row in task_rows}
        parquet_paths = sorted((root / "data").glob("episode_*.parquet"))
        if len(parquet_paths) != len(lengths):
            raise ValueError(
                f"Found {len(parquet_paths)} Parquet files for {len(lengths)} episodes."
            )
        self._dataset = load_dataset(
            "parquet",
            data_files={"train": [str(path) for path in parquet_paths]},
            split="train",
            cache_dir=cache_dir,
        )
        if len(self._dataset) != self._episode_ends[-1]:
            raise ValueError(
                f"Loaded {len(self._dataset)} frames, but episode metadata declares "
                f"{self._episode_ends[-1]}."
            )
        self._root = root
        self._action_horizon = int(action_horizon)
        if self._action_horizon <= 0:
            raise ValueError("action_horizon must be positive.")

    @property
    def dataset_root(self) -> Path:
        """Return the source dataset root."""
        return self._root

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode = bisect.bisect_right(self._episode_ends, index)
        start = self._episode_starts[episode]
        end = self._episode_ends[episode]
        local_index = index - start
        future = future_action_indices(end - start, self._action_horizon)[local_index]
        future_rows = self._dataset[[start + int(offset) for offset in future]]
        row = self._dataset[index]

        measured_tcp = np.asarray(row["observation.eef_state"], dtype=np.float64)
        absolute_actions = np.asarray(future_rows["action"], dtype=np.float64)
        actions = (
            absolute_actions.copy()
            if self._action_representation == "joint_absolute"
            else absolute_tcp_chunk_to_control_delta(measured_tcp, absolute_actions)
        )
        if (
            actions.shape
            != (
                self._action_horizon,
                8 if self._action_representation == "joint_absolute" else 7,
            )
            or not np.isfinite(actions).all()
        ):
            raise ValueError(
                "Flexiv action chunk has invalid shape or non-finite commands."
            )
        state = np.concatenate(
            (
                np.asarray(row["observation.joint_state"], dtype=np.float32),
                np.asarray(row["observation.gripper_state"], dtype=np.float32),
            )
        )
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("Flexiv proprio must contain eight finite values.")
        task_index = int(row["task_index"])
        if task_index not in self._tasks:
            raise KeyError(f"No prompt found for Flexiv task_index={task_index}.")
        return {
            "observation/image": _decode_rgb(row["observation.image"]),
            "observation/wrist_image": _decode_rgb(row["observation.wrist_image"]),
            "observation/state": state,
            "actions": actions.astype(np.float32),
            "prompt": self._tasks[task_index],
        }
