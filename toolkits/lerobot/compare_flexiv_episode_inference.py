#!/usr/bin/env python3
"""Compare one full Flexiv episode's predicted and recorded actions."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from rlinf.models.embodiment.openpi_rlinf.inference import (
    OpenPIRLInfStage1Adapter,
    load_stage1_model,
)


def parse_args() -> argparse.Namespace:
    """Parse model, dataset, and episode settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inference-config",
        type=Path,
        default=Path("examples/embodiment/config/flexiv_joint_absolute_inference.yaml"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "/home/qccc/Flexiv/flexiv_rdk/example_py/myq/"
            "peg_insertion_joint_absolute_1008"
        ),
    )
    parser.add_argument("--episode", type=int, required=True, help="Episode index")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/flexiv_episode_compare")
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", help="Override the task prompt stored in the dataset")
    return parser.parse_args()


def load_prompt(dataset: Path, override: str | None) -> str:
    """Read the episode task text unless the caller supplied a prompt."""
    if override:
        return override
    tasks_path = dataset / "meta" / "tasks.jsonl"
    for line in tasks_path.read_text().splitlines():
        task = json.loads(line)
        if int(task.get("task_index", 0)) == 0:
            return str(task["task"])
    raise ValueError(f"No task_index=0 prompt found in {tasks_path}")


def decode_rgb(encoded: bytes) -> np.ndarray:
    """Decode one parquet JPEG/PNG field into an RGB array."""
    with Image.open(io.BytesIO(encoded)) as image:
        return np.asarray(image.convert("RGB"))


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    from omegaconf import OmegaConf

    config = OmegaConf.to_container(
        OmegaConf.load(args.inference_config), resolve=True
    )
    checkpoint = Path(config["checkpoint"])
    stats = Path(config["norm_stats"])
    recipe = Path(config["training_config"])
    steps = int(config.get("num_flow_steps", 10))
    representation = str(config.get("action_representation", "joint_absolute"))
    horizon = int(config.get("base_request_steps", 16))
    episode_path = args.dataset / "data" / f"episode_{args.episode:06d}.parquet"
    if not episode_path.is_file():
        raise FileNotFoundError(episode_path)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    prompt = load_prompt(args.dataset, args.prompt)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"episode_{args.episode:06d}"

    model = load_stage1_model(
        recipe,
        checkpoint,
        args.device,
        norm_stats_path=stats,
        num_flow_steps=steps,
        use_cuda_graph=False,
        rtc_guidance_mode="approx",
        action_representation=representation,
    )
    policy = OpenPIRLInfStage1Adapter(
        model,
        horizon,
        use_rtc=False,
        rtc_guidance_mode="approx",
    )

    required = [
        "frame_index",
        "observation.joint_state",
        "observation.gripper_state",
        "observation.image",
        "observation.wrist_image",
        "action",
    ]
    predicted: list[np.ndarray] = []
    actual: list[np.ndarray] = []
    frame_indices: list[int] = []
    try:
        parquet = pq.ParquetFile(episode_path)
        total = parquet.metadata.num_rows
        for record_batch in parquet.iter_batches(batch_size=args.batch_size, columns=required):
            rows = record_batch.to_pylist()
            states = np.asarray(
                [row["observation.joint_state"] + row["observation.gripper_state"] for row in rows],
                dtype=np.float32,
            )
            main_images = np.stack([decode_rgb(row["observation.image"]) for row in rows])
            wrist_images = np.stack(
                [decode_rgb(row["observation.wrist_image"]) for row in rows]
            )
            observation: dict[str, Any] = {
                "states": states,
                "main_images": main_images,
                "wrist_images": wrist_images,
                "task_descriptions": [prompt] * len(rows),
            }
            with torch.inference_mode():
                actions, _ = policy.predict_action_batch(observation)
            # Each parquet row is evaluated as the current observation; compare
            # its immediate prediction with that row's recorded action.
            predicted.extend(actions[:, 0].detach().float().cpu().numpy())
            actual.extend(np.asarray([row["action"] for row in rows], dtype=np.float32))
            frame_indices.extend(int(row["frame_index"]) for row in rows)
            done = len(frame_indices)
            print(f"inferred {done}/{total} frames", flush=True)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
    finally:
        policy.close()

    prediction = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(actual, dtype=np.float64)
    frames = np.asarray(frame_indices, dtype=np.int64)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError(f"Prediction/target shape mismatch: {prediction.shape}, {target.shape}")
    error = prediction - target
    is_joint = representation == "joint_absolute"
    action_dim = 8 if is_joint else prediction.shape[1]
    labels = [f"q{i + 1}" for i in range(7)] + ["gripper"] if is_joint else [
        f"action_{i}" for i in range(action_dim)
    ]

    csv_path = args.output_dir / f"{stem}_actions.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_index", *[f"pred_{x}" for x in labels], *[f"actual_{x}" for x in labels], *[f"error_{x}" for x in labels]])
        for index, pred, truth, delta in zip(frames, prediction, target, error):
            writer.writerow([index, *pred.tolist(), *truth.tolist(), *delta.tolist()])

    npz_path = args.output_dir / f"{stem}_actions.npz"
    np.savez_compressed(
        npz_path,
        frame_index=frames,
        predicted=prediction,
        actual=target,
        error=error,
        prompt=prompt,
        inference_mode="independent_frame_no_rtc",
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(action_dim, 1, figsize=(14, max(2.3 * action_dim, 8)), sharex=True)
    axes = np.atleast_1d(axes)
    for dim, axis in enumerate(axes):
        scale = 180.0 / np.pi if is_joint and dim < 7 else (1000.0 if is_joint and dim == 7 else 1.0)
        unit = "deg" if is_joint and dim < 7 else ("mm" if is_joint else "action")
        axis.plot(frames, prediction[:, dim] * scale, label="model", linewidth=1.1)
        axis.plot(frames, target[:, dim] * scale, label="dataset", linewidth=1.0, alpha=0.8)
        axis.set_ylabel(f"{labels[dim]} ({unit})")
        axis.grid(True, alpha=0.25)
        if dim == 0:
            axis.legend(loc="upper right")
    axes[-1].set_xlabel("Frame")
    figure.suptitle(
        f"{stem}: teacher-forced per-frame prediction vs dataset action\n"
        f"RTC=off; prompt={prompt}"
    )
    figure.tight_layout()
    plot_path = args.output_dir / f"{stem}_actions.png"
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)

    metrics: dict[str, Any] = {
        "episode": args.episode,
        "frames": len(frames),
        "prompt": prompt,
        "inference_mode": "independent_frame_no_rtc",
        "mae": {},
        "rmse": {},
    }
    for dim, label in enumerate(labels):
        scale = 180.0 / np.pi if is_joint and dim < 7 else (1000.0 if is_joint and dim == 7 else 1.0)
        unit = "deg" if is_joint and dim < 7 else ("mm" if is_joint and dim == 7 else "action")
        metrics["mae"][label] = {"value": float(np.mean(np.abs(error[:, dim])) * scale), "unit": unit}
        metrics["rmse"][label] = {"value": float(np.sqrt(np.mean(error[:, dim] ** 2)) * scale), "unit": unit}
    metrics_path = args.output_dir / f"{stem}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(f"saved plot: {plot_path}")
    print(f"saved actions: {csv_path}")
    print(f"saved metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
