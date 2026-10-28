# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Compute OpenPI statistics for compact single-arm Flexiv demonstrations.

The script builds absolute joint-position action chunks from future recorded
observations, with the same terminal padding as the Flexiv SFT dataset adapter.
It reads only numeric Parquet columns and never rewrites the source recordings.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from rlinf.data.datasets.openpi_rlinf.flexiv.flexiv_sft_dataset import (
    future_action_chunks,
    future_action_indices,
)

DEFAULT_DATASET = Path("/home/qccc/Flexiv/flexiv_rdk/example_py/myq/peg_insertion_ok")


def read_metadata(dataset: Path) -> dict:
    """Read the recording contract without importing the model environment."""
    with (dataset / "meta" / "info.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if len(metadata.get("recorded_arm_names", [])) != 1:
        raise ValueError("Statistics require a single recorded arm.")
    expected = {
        "observation.joint_state": [7],
        "action": [8],
    }
    for key, shape in expected.items():
        if metadata.get("features", {}).get(key, {}).get("shape") != shape:
            raise ValueError(f"{key} must have shape {shape}.")
    return metadata


def iter_numeric_batches(
    dataset: Path,
    *,
    batch_size: int = 4096,
    action_horizon: int = 32,
    model_action_dim: int = 8,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield 8D joint-plus-gripper states and absolute 8D action chunks."""
    import pyarrow.parquet as pq

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if model_action_dim != 8:
        raise ValueError("Flexiv norm stats must use the physical 8D dimension.")
    read_metadata(dataset)
    paths = sorted((dataset / "data").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No Parquet episodes found under {dataset / 'data'}.")
    columns = ["observation.joint_state", "action"]
    columns.insert(1, "observation.gripper_state")
    for path in paths:
        table = pq.read_table(path, columns=columns)
        values = {
            key: np.asarray(table[key].to_pylist(), dtype=np.float64) for key in columns
        }
        for key, width in zip(columns, (7, 1, 8), strict=True):
            if (
                values[key].shape != (table.num_rows, width)
                or not np.isfinite(values[key]).all()
            ):
                raise ValueError(
                    f"Invalid numeric values or shape for {key} in {path}."
                )
        chunk_indices = future_action_indices(table.num_rows, action_horizon)
        for start in range(0, table.num_rows, batch_size):
            stop = min(start + batch_size, table.num_rows)
            action_chunks = future_action_chunks(values["action"], chunk_indices[start:stop])
            padded_actions = action_chunks
            padded_state = np.concatenate(
                (
                    values["observation.joint_state"][start:stop],
                    values["observation.gripper_state"][start:stop],
                ),
                axis=-1,
            )
            yield {
                "state": padded_state,
                "actions": padded_actions,
            }


def compute_stats(
    dataset: Path,
    output: Path,
    *,
    batch_size: int = 4096,
    action_horizon: int = 32,
    model_action_dim: int = 8,
) -> None:
    """Save the same norm_stats.json structure consumed by OpenPI transforms."""
    batches = iter_numeric_batches(
        dataset,
        batch_size=batch_size,
        action_horizon=action_horizon,
        model_action_dim=model_action_dim,
    )
    first = next(batches, None)  # Validate representation before loading OpenPI.
    if first is None:
        raise ValueError("Cannot compute statistics from an empty dataset.")
    import itertools

    import openpi.shared.normalize as normalize

    if (output / "norm_stats.json").exists():
        raise FileExistsError(
            f"Statistics already exist at {output}; choose a new output directory."
        )
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    frames = 0
    for batch in itertools.chain((first,), batches):
        for key, accumulator in stats.items():
            accumulator.update(batch[key])
        frames += len(batch["state"])
    metadata = read_metadata(dataset)
    if frames != int(metadata["total_frames"]):
        raise ValueError(
            f"Read {frames} frames, but metadata declares {metadata['total_frames']}."
        )
    normalize.save(
        output,
        {key: accumulator.get_statistics() for key, accumulator in stats.items()},
    )


def main() -> None:
    """Inspect the source now, or compute statistics with the SFT transform."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--model-action-dim", type=int, default=8)
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Check the recording schema without computing statistics.",
    )
    args = parser.parse_args()
    if args.inspect:
        metadata = read_metadata(args.dataset)
        print(
            json.dumps(
                {
                    "dataset": str(args.dataset),
                    "state_dim": 8,
                    "action_dim": 8,
                    "action_representation": "joint_absolute",
                    "frames": metadata["total_frames"],
                    "episodes": metadata["total_episodes"],
                },
                indent=2,
            )
        )
        return
    if args.output is None:
        parser.error("--output is required when computing statistics")
    compute_stats(
        args.dataset,
        args.output,
        batch_size=args.batch_size,
        action_horizon=args.action_horizon,
        model_action_dim=args.model_action_dim,
    )


if __name__ == "__main__":
    main()
