#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Control both Flexiv arms from the Flexiv conda environment."""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from pathlib import Path

from flexiv_remote_robot import (
    FlexivRobotAdapter,
    RemotePolicy,
    load_myq_config,
    run_episode,
)


def main() -> int:
    """Connect to a prevalidated inference server before activating either arm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--myq-root", type=Path, default=Path("/home/qccc/Flexiv/flexiv_rdk/example_py")
    )
    parser.add_argument(
        "--myq-config",
        type=Path,
        default=Path("/home/qccc/Flexiv/flexiv_rdk/example_py/myq/drag_collect.yaml"),
    )
    parser.add_argument("--inference-config", type=Path)
    parser.add_argument(
        "--disable-debug",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Disable trace/image recording and detailed diagnostics; keep inference metrics, prompt and phase changes",
    )
    parser.add_argument(
        "--binary-gripper", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--backend", choices=("torch", "apx"), default="torch")
    parser.add_argument(
        "--skip-gripper-init",
        action="store_true",
        help="Keep the current follower gripper opening; skip startup Init and open",
    )
    parser.add_argument("--max-ticks", type=int)
    parser.add_argument(
        "--diagnostic-dir",
        type=Path,
        help="Save all inference observations/chunks and every control receipt for offline replay",
    )
    parser.add_argument(
        "--task-timeout-seconds",
        type=float,
        help="Stop and label failure when the task exceeds this duration",
    )
    parser.add_argument(
        "--handoff-grace-ticks",
        type=int,
        help="Ticks between warnings while holding for a late inference (default: chunk length minus one)",
    )
    parser.add_argument(
        "--capture-dir",
        type=Path,
        help="Save the first processed RGB observation pairs under this directory",
    )
    parser.add_argument(
        "--capture-count",
        type=int,
        default=7,
        help="Number of inference observations to save when --capture-dir is set",
    )
    parser.add_argument(
        "--capture-interval-seconds",
        type=float,
        default=15.0,
        help="Minimum spacing between saved RGB observation pairs",
    )
    args = parser.parse_args()
    if args.capture_count <= 0:
        parser.error("--capture-count must be positive")
    if args.capture_interval_seconds <= 0:
        parser.error("--capture-interval-seconds must be positive")
    import yaml

    inference_config = (
        yaml.safe_load(args.inference_config.read_text())
        if args.inference_config is not None
        else {}
    )
    disable_debug = (
        bool(inference_config.get("disable_debug", False))
        if args.disable_debug is None
        else args.disable_debug
    )
    logging.basicConfig(level=logging.WARNING if disable_debug else logging.INFO)
    if disable_debug:
        logging.getLogger().setLevel(logging.WARNING)
        args.diagnostic_dir = None
        args.capture_dir = None
    logging.getLogger("flexiv.startup").setLevel(logging.INFO)
    logging.getLogger("flexiv.status").setLevel(logging.INFO)
    gripper_policy = dict(inference_config.get("gripper_policy", {}))
    if args.binary_gripper is not None:
        gripper_policy["binary"] = args.binary_gripper
    representation = inference_config.get("action_representation", "eef_delta")
    config = load_myq_config(args.myq_config)
    logging.getLogger("flexiv.startup").info(
        "Policy prompt: %s", config["recording"]["task"]
    )
    sys.path.insert(0, str(args.myq_root))
    camera_shapes = {
        name: (
            int(
                camera.get("resize_height", camera.get("crop_height", camera["height"]))
            ),
            int(camera.get("resize_width", camera.get("crop_width", camera["width"]))),
            3,
        )
        for name, camera in config["cameras"].items()
    }
    shapes = {
        "states": (8,),
        "main_images": camera_shapes["main"],
        "wrist_images": camera_shapes["wrist"],
    }
    robot = FlexivRobotAdapter(
        config,
        myq_path=args.myq_config,
        capture_dir=args.capture_dir,
        capture_count=args.capture_count,
        capture_interval_seconds=args.capture_interval_seconds,
        initialize_gripper=False if args.skip_gripper_init else None,
        action_representation=representation,
        joint_control=inference_config.get("joint_control"),
        home_dataset_path=inference_config.get("home_dataset_path"),
        gripper_policy=gripper_policy,
        auto_fine=inference_config.get("auto_fine"),
        record_diagnostics=args.diagnostic_dir is not None,
    )
    policy = RemotePolicy(
        args.host,
        args.port,
        str(uuid.uuid4()),
        input_shapes=shapes,
        backend=args.backend,
        action_representation=representation,
        binary_gripper=robot.binary_gripper,
        gripper_threshold=robot.gripper_threshold,
        diagnostic_dir=args.diagnostic_dir,
        diagnostic_metadata={
            "inference_config": inference_config,
            "myq_config": config,
            "backend": args.backend,
            "host": args.host,
            "port": args.port,
        },
    )
    robot.fine_available = policy.fine_available
    robot.continuous_episodes = getattr(policy, "continuous_episodes", False)
    logging.getLogger("flexiv.startup").info(
        "Inference schedule: phase=%s request_index=%d handoff_index=%d origin_schedule=%s fine_steps=%d fine_available=%s",
        policy.initial_phase,
        policy.base_request_steps,
        policy.base_horizon,
        policy.coarse_origin_schedule,
        policy.fine_horizon,
        policy.fine_available,
    )
    if policy.initial_phase == "coarse" and policy.fine_available:
        logging.getLogger("flexiv.status").info(
            "Fine trigger config: %s; press I to request Fine manually.",
            inference_config.get("auto_fine", "manual only"),
        )
    try:
        return run_episode(
            robot,
            policy,
            max_ticks=args.max_ticks,
            task_timeout_seconds=args.task_timeout_seconds
            if args.task_timeout_seconds is not None
            else inference_config.get("task_timeout_seconds"),
            handoff_grace_ticks=args.handoff_grace_ticks,
        )
    except Exception as exc:
        print(f"Flexiv rollout stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
