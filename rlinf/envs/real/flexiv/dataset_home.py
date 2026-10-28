# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Read measured joint Homes from compact Flexiv demonstrations."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_dataset_joint_home(dataset_path: str | Path) -> tuple[Path, np.ndarray]:
    """Read seven measured joint angles from the first episode's first frame.

    Args:
        dataset_path: Dataset root, data directory, or an episode Parquet file.
            Directories select the first Parquet file in sorted path order.

    Returns:
        Selected episode path and a finite seven-element joint vector in radians.

    Raises:
        FileNotFoundError: The path or its Parquet episodes are missing.
        ValueError: The episode has no valid first-frame measured joint state.
    """
    import pyarrow.parquet as pq

    path = Path(dataset_path).expanduser()
    if path.is_dir():
        directory = path / "data" if (path / "data").is_dir() else path
        episodes = sorted(directory.rglob("*.parquet"))
        if not episodes:
            raise FileNotFoundError(f"No Parquet episodes found under {directory}.")
        path = episodes[0]
    elif not path.is_file():
        raise FileNotFoundError(f"Dataset Home path does not exist: {path}.")
    elif path.suffix != ".parquet":
        raise ValueError("Dataset Home requires a Parquet episode or data directory.")

    column = "observation.joint_state"
    with pq.ParquetFile(path) as episode:
        if column not in episode.schema_arrow.names:
            raise ValueError(f"Dataset Home episode {path} has no {column} column.")
        if episode.metadata.num_rows == 0:
            raise ValueError(f"Dataset Home episode {path} is empty.")
        columns = [column]
        if "frame_index" in episode.schema_arrow.names:
            columns.append("frame_index")
        first = episode.read_row_group(0, columns=columns).slice(0, 1)
    if "frame_index" in columns and first["frame_index"][0].as_py() != 0:
        raise ValueError(f"Dataset Home episode {path} must start at frame_index 0.")
    joints = np.asarray(first[column][0].as_py(), dtype=np.float64)
    if joints.shape != (7,) or not np.isfinite(joints).all():
        raise ValueError(f"Dataset Home episode {path} requires seven finite joints.")
    return path, joints.copy()
