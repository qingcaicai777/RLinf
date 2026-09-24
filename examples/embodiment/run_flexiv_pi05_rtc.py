#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run standard OpenPI/JAX PI0.5 RTC inference on one Flexiv arm."""

from __future__ import annotations

import argparse
import logging
import signal
from pathlib import Path

import yaml

from rlinf.envs.real.flexiv.rtc_adapter import FlexivRTCAdapter, FlexivRTCConfig
from rlinf.models.embodiment.openpi.rtc import (
    LateChunkBehavior,
    PI05PolicyAdapter,
    PI05PolicySpec,
    RTCChunkScheduler,
    RTCRobotRuntime,
    RuntimeConfig,
)


def parse_args() -> argparse.Namespace:
    """Parse the deployment config path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/embodiment/config/flexiv_pi05_rtc.yaml"),
    )
    return parser.parse_args()


def _pair(values: list[float], name: str) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values.")
    return float(values[0]), float(values[1])


def main() -> int:
    """Load policy first, then activate hardware and run the RTC loop."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    with args.config.expanduser().open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    policy_config = config["policy"]
    rtc_config = config["rtc"]
    control_config = config["control"]
    flexiv_config = dict(config["flexiv"])
    arm = flexiv_config.pop("arm", None)
    if arm not in {"left", "right"}:
        raise ValueError("flexiv.arm must explicitly be left or right.")
    if float(flexiv_config["control_hz"]) != float(control_config["frequency_hz"]):
        raise ValueError("flexiv.control_hz must equal control.frequency_hz.")

    camera_key_map = dict(policy_config["camera_keys"])
    policy = PI05PolicyAdapter.load(
        PI05PolicySpec(
            config_name=str(policy_config["config_name"]),
            checkpoint_path=Path(policy_config["checkpoint_path"]),
            policy_camera_keys=tuple(camera_key_map.values()),
            state_key=str(policy_config["state_key"]),
            action_horizon=int(rtc_config["action_horizon"]),
        )
    )
    robot_config = FlexivRTCConfig(
        env=flexiv_config,
        camera_key_map=camera_key_map,
        translation_unit=str(control_config["translation_unit"]),
        rotation_representation=str(control_config["rotation_representation"]),
        rotation_unit=str(control_config["rotation_unit"]),
        tcp_delta_frame=str(control_config["tcp_delta_frame"]),
        translation_delta_limit_m=float(control_config["translation_delta_limit_m"]),
        rotation_delta_limit_rad=float(control_config["rotation_delta_limit_rad"]),
        gripper_state_range=_pair(
            control_config["gripper_state_range"], "gripper_state_range"
        ),
        gripper_command_range=_pair(
            control_config["gripper_command_range"], "gripper_command_range"
        ),
        gripper_state_open_at_max=bool(control_config["gripper_state_open_at_max"]),
        gripper_command_open_at_max=bool(control_config["gripper_command_open_at_max"]),
    )
    robot_config.validate()
    scheduler = RTCChunkScheduler(
        policy.infer,
        action_horizon=int(rtc_config["action_horizon"]),
        execution_horizon=int(rtc_config["execution_horizon"]),
        initial_delay=int(rtc_config["initial_delay"]),
        delay_history_length=int(rtc_config["delay_history_length"]),
        max_guidance_weight=float(rtc_config["max_guidance_weight"]),
        num_steps=int(rtc_config["num_steps"]),
    )
    runtime = RTCRobotRuntime(
        FlexivRTCAdapter(robot_config),
        scheduler,
        RuntimeConfig(
            prompt=str(policy_config["prompt"]),
            control_hz=float(control_config["frequency_hz"]),
            max_ticks=control_config.get("max_ticks"),
            late_chunk_behavior=LateChunkBehavior(
                str(rtc_config["late_chunk_behavior"])
            ),
        ),
    )
    signal.signal(signal.SIGINT, lambda _signum, _frame: runtime.request_stop())
    signal.signal(signal.SIGTERM, lambda _signum, _frame: runtime.request_stop())
    try:
        return runtime.run()
    except Exception:
        logging.exception("Flexiv PI0.5 RTC inference failed.")
        runtime.close()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
