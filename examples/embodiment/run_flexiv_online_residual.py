#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Run TCP delta or absolute joint Stage1 policies and Stage2 handoffs."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import signal
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from omegaconf import OmegaConf

from rlinf.algorithms.online_residual_rl.config import OnlineResidualRLConfig
from rlinf.algorithms.online_residual_rl.deployment import (
    ResidualDeploymentConfig,
    ResidualHandoffRuntime,
)
from rlinf.envs.real.flexiv.rtc_adapter import (
    FlexivRTCAdapter,
    FlexivRTCConfig,
    project_flexiv_actions,
)


def parse_args() -> argparse.Namespace:
    """Read checkpoint and robot settings for one deployment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inference-config",
        type=Path,
        help="Load deployment defaults from YAML; command-line values take precedence",
    )
    parser.add_argument(
        "--action-representation", choices=("eef_delta", "joint_absolute")
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--norm-stats", type=Path)
    parser.add_argument(
        "--disable-debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Suppress detailed diagnostics; keep inference metrics, phase changes, startup messages and errors",
    )
    parser.add_argument(
        "--binary-gripper",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Transmit pre-unnormalization gripper actions for robot-side binary control",
    )
    parser.add_argument("--backend", choices=("torch", "apx"), default="torch")
    parser.add_argument("--gripper-threshold", type=float, default=0.5)
    parser.add_argument("--actor-checkpoint", type=Path)
    parser.add_argument(
        "--stage1-only",
        action=argparse.BooleanOptionalAction,
        help="Use only the trained Torch or APX PI0.5 base for every episode",
    )
    parser.add_argument(
        "--base-chunk-steps",
        type=int,
        default=22,
        help="Coarse-phase handoff index in each 32-step prediction (default: 22)",
    )
    parser.add_argument(
        "--base-request-steps",
        type=int,
        default=16,
        help="Coarse-phase request index in each prediction (default: 16)",
    )
    parser.add_argument(
        "--disable-rtc",
        action="store_true",
        help="Disable cross-chunk RTC guidance for Torch Stage1-only inference",
    )
    parser.add_argument(
        "--rtc-max-guidance-weight",
        "--rtc_max_guidance_weight",
        type=float,
        default=10.0,
        help="Maximum Torch RTC guidance weight (default: 10)",
    )
    parser.add_argument(
        "--rtc-delay-buffer-size",
        "--rtc_delay_buffer_size",
        type=int,
        default=8,
        help="Number of accepted inference delays retained for RTC (default: 8)",
    )
    parser.add_argument(
        "--rtc-guidance-mode",
        choices=("approx", "jacobian"),
        default="jacobian",
        help="RTC correction: model-Jacobian VJP (default) or lightweight approximation",
    )
    parser.add_argument(
        "--disable-torch-cuda-graph",
        action="store_true",
        help="Use eager PyTorch action-expert inference instead of CUDA graph replay",
    )
    parser.add_argument(
        "--base-only-episodes",
        type=int,
        default=50,
        help="Stage1-only labelled episodes collected before Stage2 warmup",
    )
    parser.add_argument("--apx-checkpoint", type=Path)
    parser.add_argument(
        "--num-flow-steps",
        type=int,
        default=10,
        help="PI0.5 flow denoising steps for Torch or APX inference (default: 10)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-ticks", type=int)
    parser.add_argument("--task-timeout-seconds", type=float)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--dummy-robot", action="store_true")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run policy in this environment; Flexiv runs in its own conda environment",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument(
        "--leader-serial", help="Distinct Flexiv main arm for drag intervention"
    )
    parser.add_argument("--operator-console", action="store_true")
    parser.add_argument("--online-train", action=argparse.BooleanOptionalAction)
    parser.add_argument("--episode-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--resume-stage2", type=Path)
    parser.add_argument("--terminal-label", choices=("success", "failure"))
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("examples/sft/config/flexiv_rlt_stage1_sft_openpi_pi05.yaml"),
    )
    parser.add_argument(
        "--stage2-config",
        type=Path,
        default=Path("examples/embodiment/config/flexiv_online_residual_rl_pi05.yaml"),
    )
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=Path("examples/embodiment/config/flexiv_pi05_rtc.yaml"),
    )
    preliminary, _ = parser.parse_known_args()
    if preliminary.inference_config is not None:
        defaults = OmegaConf.to_container(
            OmegaConf.load(preliminary.inference_config), resolve=True
        )
        defaults.pop("joint_control", None)
        defaults.pop("home_dataset_path", None)
        defaults.pop("auto_fine", None)
        gripper_policy = defaults.pop("gripper_policy", {})
        defaults["binary_gripper"] = gripper_policy.get("binary", False)
        defaults["gripper_threshold"] = gripper_policy.get("threshold", 0.5)
        known = {action.dest for action in parser._actions}
        if set(defaults) - known:
            parser.error(
                f"Unknown inference config keys: {sorted(set(defaults) - known)}"
            )
        parser.set_defaults(**defaults)
    args = parser.parse_args()
    if args.checkpoint is None:
        parser.error("--checkpoint or checkpoint in --inference-config is required")
    return args


def _build_policy(args: argparse.Namespace, stage2, contract):
    if args.backend == "torch":
        from rlinf.models.embodiment.openpi_rlinf.inference import (
            OpenPIRLInfResidualAdapter,
            OpenPIRLInfStage1Adapter,
            load_stage1_model,
        )

        if args.stage1_only and (
            args.actor_checkpoint or args.resume_stage2 or args.online_train
        ):
            raise ValueError("--stage1-only cannot load or train a Stage2 Actor.")
        rtc_guidance_mode = "approx" if args.disable_rtc else args.rtc_guidance_mode
        model = load_stage1_model(
            args.training_config,
            args.checkpoint,
            args.device,
            norm_stats_path=args.norm_stats,
            num_flow_steps=args.num_flow_steps,
            use_cuda_graph=not getattr(args, "disable_torch_cuda_graph", False),
            rtc_max_guidance_weight=args.rtc_max_guidance_weight,
            rtc_guidance_mode=rtc_guidance_mode,
            action_representation=contract.action_representation.value,
        )
        # Online training starts from the zero-residual Actor, which reproduces
        # the Stage1 base until the labelled episode and Critic warmups finish.
        if (
            args.actor_checkpoint is None
            and args.resume_stage2 is None
            and not args.online_train
            and (args.stage1_only or args.episode_dir is None)
        ):
            return OpenPIRLInfStage1Adapter(
                model,
                args.base_chunk_steps,
                use_rtc=not args.disable_rtc,
                rtc_guidance_mode=rtc_guidance_mode,
            )
        return OpenPIRLInfResidualAdapter(
            model,
            contract,
            actor_checkpoint=args.actor_checkpoint,
            base_horizon=args.base_chunk_steps,
            rtc_guidance_mode=args.rtc_guidance_mode,
        )

    if args.apx_checkpoint is None:
        raise ValueError("--apx-checkpoint is required for --backend apx.")
    from rlinf.models.embodiment.openpi.apxinf_adapter import (
        OnlineResidualApxInfAdapter,
        OpenPIApxInfAdapter,
    )
    from rlinf.models.embodiment.openpi_rlinf.inference import (
        build_apx_stage1_model_config,
    )

    model_cfg = build_apx_stage1_model_config(
        stage2.rollout.model,
        training_config=args.training_config,
        checkpoint=args.checkpoint,
        norm_stats_path=args.norm_stats,
        apx_checkpoint=args.apx_checkpoint,
        action_representation=contract.action_representation.value,
        num_flow_steps=args.num_flow_steps,
        stage1_only=args.stage1_only,
        actor_checkpoint=args.actor_checkpoint,
    )
    base = OpenPIApxInfAdapter(model_cfg, args.device)
    if (
        base.metadata["action_horizon"] != 32
        or base.metadata["model_action_dim"] != 32
        or base.metadata["num_views"] != 2
        or base.action_dim != contract.action_dim
    ):
        base.close()
        raise ValueError(
            "APX Stage1 export must be 32 steps, 32 model dims, 2 views, "
            "and match the configured physical action width."
        )
    if args.stage1_only:
        if args.actor_checkpoint or args.resume_stage2 or args.online_train:
            base.close()
            raise ValueError("--stage1-only cannot load or train a Stage2 Actor.")
        return _Stage1ApxOnlyPolicy(base, args.base_chunk_steps)
    try:
        adapter = OnlineResidualApxInfAdapter(model_cfg, args.device, base=base)
    except BaseException:
        base.close()
        raise
    return adapter


class _Stage1ApxOnlyPolicy:
    """Use only the trained APX base when no Stage2 Actor exists yet."""

    actor_version = 0
    supports_fine_phase = False

    def __init__(self, base, inference_horizon: int = 6) -> None:
        if not 1 <= inference_horizon <= base.metadata["action_horizon"]:
            raise ValueError("Base execution steps must fit in the model horizon.")
        self.base = base
        self.inference_horizon = inference_horizon

    def predict_action_batch(self, env_obs, *, mode="eval", committed=None):
        if committed is not None:
            raise ValueError("APX Stage1-only policy does not use committed actions.")
        physical, metadata = self.base.predict_action_batch(env_obs, mode=mode)
        if physical.shape[1] < self.inference_horizon:
            raise ValueError("APX prediction is shorter than the base chunk.")
        return physical, metadata

    def close(self) -> None:
        self.base.close()


def _build_robot(config, *, dummy: bool, leader_serial: str | None) -> FlexivRTCAdapter:
    control = config["control"]
    camera_keys = dict(config["policy"]["camera_keys"])
    if camera_keys != {
        "main": "observation/image",
        "wrist": "observation/wrist_image",
    }:
        raise ValueError("Stage1 Flexiv requires main and wrist OpenPI camera keys.")
    env = dict(config["flexiv"])
    arm = env.pop("arm", None)
    if not dummy and arm not in {"left", "right"}:
        raise ValueError("flexiv.arm must be left or right before hardware activation.")
    if dummy:
        env["is_dummy"] = True
    if float(env["control_hz"]) != float(control["frequency_hz"]):
        raise ValueError("Flexiv and deployment control frequencies must match.")
    if (
        control["translation_unit"] != "m"
        or control["rotation_representation"] != "rotvec"
        or control["rotation_unit"] != "rad"
        or control["tcp_delta_frame"] != "world"
    ):
        raise ValueError(
            "Stage1 Flexiv actions require metre, radian rotvec, world-frame deltas."
        )
    return FlexivRTCAdapter(
        FlexivRTCConfig(
            env=env,
            camera_key_map=camera_keys,
            translation_unit=str(control["translation_unit"]),
            rotation_representation=str(control["rotation_representation"]),
            rotation_unit=str(control["rotation_unit"]),
            tcp_delta_frame=str(control["tcp_delta_frame"]),
            translation_delta_limit_m=float(control["translation_delta_limit_m"]),
            rotation_delta_limit_rad=float(control["rotation_delta_limit_rad"]),
            enable_tcp_delta_limits=bool(control.get("enable_tcp_delta_limits", False)),
            gripper_state_range=tuple(control["gripper_state_range"]),
            gripper_command_range=tuple(control["gripper_command_range"]),
            gripper_state_open_at_max=bool(control["gripper_state_open_at_max"]),
            gripper_command_open_at_max=bool(control["gripper_command_open_at_max"]),
            leader_serial=leader_serial,
        )
    )


def _build_online_training(args, stage2, contract, policy, robot_config, mailbox):
    """Create a learner with the same action normalization and limits as rollout."""
    import torch

    from rlinf.algorithms.online_residual_rl.action_mapping import (
        FlexivDeltaActionMapping,
        FlexivJointActionMapping,
    )
    from rlinf.algorithms.online_residual_rl.learner import OnlineResidualLearner
    from rlinf.algorithms.online_residual_rl.learner_process import OnlineLearnerProcess
    from rlinf.algorithms.online_residual_rl.replay import EpisodeReplay
    from rlinf.algorithms.online_residual_rl.training import OnlineTraining

    if args.episode_dir is None or args.checkpoint_dir is None:
        raise ValueError("Online training requires --episode-dir and --checkpoint-dir.")
    from rlinf.models.embodiment.openpi_rlinf.inference import (
        OpenPIRLInfStage1Adapter,
    )

    if isinstance(policy, OpenPIRLInfStage1Adapter):
        raise ValueError(
            "Online residual training requires a residual adapter; "
            "--stage1-only cannot collect training features."
        )
    zeros = torch.zeros(1, 1, contract.action_dim)
    encoded_zero = policy.encode_receipts(zeros).detach().cpu().reshape(-1)
    encoded_one = (
        policy.encode_receipts(torch.ones_like(zeros)).detach().cpu().reshape(-1)
    )
    scale = 1.0 / (encoded_one - encoded_zero)
    offset = -encoded_zero * scale
    control = _build_robot_config(robot_config)
    if contract.action_representation.value == "joint_absolute":
        mapping = FlexivJointActionMapping(
            offset,
            scale,
            gripper_range=control.gripper_command_range,
            binary_gripper=args.binary_gripper,
            threshold=args.gripper_threshold,
        )
    else:
        mapping = FlexivDeltaActionMapping(
            offset,
            scale,
            translation_limit=control.translation_delta_limit_m
            * (1000.0 if control.translation_unit == "mm" else 1.0),
            rotation_limit=control.rotation_delta_limit_rad
            * (180.0 / 3.141592653589793 if control.rotation_unit == "deg" else 1.0),
            gripper_range=control.gripper_command_range,
            limit_tcp_delta=control.enable_tcp_delta_limits,
            binary_gripper=args.binary_gripper,
            threshold=args.gripper_threshold,
        )
    # Bootstrap on CPU; the child owns optimization and its configured device.
    learner = OnlineResidualLearner(
        contract,
        action_projector=mapping,
        actor_lr=float(stage2.online_residual.actor_lr),
        critic_lr=float(stage2.online_residual.critic_lr),
        max_grad_norm=float(stage2.online_residual.max_grad_norm),
    ).cpu()
    if args.actor_checkpoint is not None and args.resume_stage2 is None:
        from rlinf.algorithms.online_residual_rl.checkpoint import load_actor_state

        learner.actor.load_state_dict(
            load_actor_state(args.actor_checkpoint, contract), strict=True
        )
        learner.target_actor.load_state_dict(learner.actor.state_dict(), strict=True)
    replay = EpisodeReplay(
        contract,
        capacity=int(stage2.online_residual.replay_capacity),
        recent_window=int(stage2.online_residual.recent_sample_window),
        storage_dir=args.episode_dir,
    )
    training = OnlineTraining(
        learner,
        replay,
        initial_episodes=args.base_only_episodes,
        warmup_updates=int(stage2.online_residual.critic_warmup_updates),
        publish_interval=int(stage2.online_residual.actor_publish_interval),
        publish=mailbox.publish,
    )
    if args.resume_stage2 is not None:
        training.load(args.resume_stage2)
        mailbox.version = training.actor_updates
        policy.actor_version = training.actor_updates
    return OnlineLearnerProcess(
        training,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=int(stage2.online_residual.checkpoint_interval),
        device=str(getattr(stage2.online_residual, "learner_device", "cpu")),
        cpu_threads=int(getattr(stage2.online_residual, "learner_cpu_threads", 1)),
    )


def _build_robot_config(config):
    """Resolve physical action semantics without activating any SDK resource."""
    control = config["control"]
    result = FlexivRTCConfig(
        env=dict(config["flexiv"]),
        camera_key_map=dict(config["policy"]["camera_keys"]),
        translation_unit=str(control["translation_unit"]),
        rotation_representation=str(control["rotation_representation"]),
        rotation_unit=str(control["rotation_unit"]),
        tcp_delta_frame=str(control["tcp_delta_frame"]),
        translation_delta_limit_m=float(control["translation_delta_limit_m"]),
        rotation_delta_limit_rad=float(control["rotation_delta_limit_rad"]),
        enable_tcp_delta_limits=bool(control.get("enable_tcp_delta_limits", False)),
        gripper_state_range=tuple(control["gripper_state_range"]),
        gripper_command_range=tuple(control["gripper_command_range"]),
        gripper_state_open_at_max=bool(control["gripper_state_open_at_max"]),
        gripper_command_open_at_max=bool(control["gripper_command_open_at_max"]),
    )
    result.validate()
    return result


def _policy_input_shapes(config) -> dict[str, tuple[int, ...]]:
    """Map configured Flexiv sensor sizes to the PI0.5 input names."""
    camera_shapes = {
        name: (
            int(
                camera.get("resize_height", camera.get("crop_height", camera["height"]))
            ),
            int(camera.get("resize_width", camera.get("crop_width", camera["width"]))),
            3,
        )
        for name, camera in config["flexiv"]["cameras"].items()
    }
    return {
        "states": (8,),
        "main_images": camera_shapes["main"],
        "wrist_images": camera_shapes["wrist"],
    }


def main() -> int:
    """Load the policy before activating Flexiv, then own its control loop."""
    args = parse_args()
    if args.episode_dir is not None and args.stage1_only and not args.online_train:
        raise ValueError(
            "Transition collection requires a Stage2 adapter; disable --stage1-only."
        )
    if args.binary_gripper and not args.serve:
        raise ValueError("--binary-gripper requires split inference with --serve.")
    if args.disable_rtc and (args.backend != "torch" or not args.stage1_only):
        raise ValueError("--disable-rtc requires --backend torch --stage1-only.")
    if args.rtc_max_guidance_weight <= 0:
        raise ValueError("--rtc-max-guidance-weight must be positive.")
    if args.rtc_delay_buffer_size <= 0:
        raise ValueError("--rtc-delay-buffer-size must be positive.")
    if args.rtc_guidance_mode == "jacobian" and args.backend != "torch":
        raise ValueError(
            "Jacobian RTC guidance is available only with --backend torch."
        )
    logging.basicConfig(level=logging.WARNING if args.disable_debug else logging.INFO)
    if args.disable_debug:
        logging.getLogger().setLevel(logging.WARNING)
        logging.getLogger("RLinf").setLevel(logging.WARNING)
    logging.getLogger("flexiv.startup").setLevel(logging.INFO)
    logging.getLogger("flexiv.status").setLevel(logging.INFO)
    stage2 = OmegaConf.load(args.stage2_config)
    robot_config = OmegaConf.to_container(
        OmegaConf.load(args.robot_config), resolve=True
    )
    contract = OnlineResidualRLConfig(
        **{
            field.name: stage2.online_residual[field.name]
            for field in dataclasses.fields(OnlineResidualRLConfig)
            if field.name in stage2.online_residual
        }
    )
    if args.action_representation is not None:
        contract = dataclasses.replace(
            contract, action_representation=args.action_representation
        )
    from hashlib import sha256

    from rlinf.models.embodiment.openpi_rlinf.inference import (
        resolve_stage1_norm_stats,
        validate_stage1_norm_stats,
    )

    stats_path = resolve_stage1_norm_stats(
        args.training_config, args.checkpoint, args.norm_stats
    )
    validate_stage1_norm_stats(stats_path, contract.action_dim)
    args.norm_stats = stats_path
    if contract.proprio_normalization != "stage1":
        raise ValueError(
            "Flexiv Stage2 requires online_residual.proprio_normalization: stage1."
        )
    normalization_id = sha256(Path(stats_path).read_bytes()).hexdigest()
    if contract.normalization_id not in {None, normalization_id}:
        raise ValueError(
            "Configured normalization_id differs from the Stage1 statistics."
        )
    contract = dataclasses.replace(contract, normalization_id=normalization_id)
    if contract.inference_horizon != 6 or contract.actor_horizon != 12:
        raise ValueError("This Flexiv executor requires n=6 and Actor horizon=12.")
    if args.base_only_episodes < 0:
        raise ValueError("--base-only-episodes must be nonnegative.")
    if (
        args.base_chunk_steps is not None or args.base_request_steps is not None
    ) and not args.serve:
        raise ValueError(
            "Base request and chunk steps are supported only by the split server."
        )
    start_base = (
        args.stage1_only
        or args.base_chunk_steps is not None
        or args.base_request_steps is not None
    )
    args.base_chunk_steps = (
        6 if args.base_chunk_steps is None else args.base_chunk_steps
    )
    if not 1 <= args.base_chunk_steps <= 32:
        raise ValueError("--base-chunk-steps must be between 1 and 32.")
    if start_base and not args.stage1_only and args.base_chunk_steps < 6:
        raise ValueError("Switching to the Actor requires at least 6 base actions.")
    if args.base_request_steps is None:
        args.base_request_steps = 0
    if not 0 <= args.base_request_steps < args.base_chunk_steps:
        raise ValueError("--base-request-steps must be in [0, --base-chunk-steps).")
    if contract.action_representation.value == "joint_absolute":
        if not args.serve:
            raise ValueError("joint_absolute requires split inference with --serve.")
    if args.stage1_only and not args.serve:
        raise ValueError("--stage1-only is supported only by the split server.")
    if float(robot_config["control"]["frequency_hz"]) != 30.0:
        raise ValueError("Stage2 Flexiv handoff is configured for 30 Hz.")
    _build_robot_config(robot_config)
    if args.serve:
        from rlinf.algorithms.online_residual_rl.online import ActorMailbox
        from rlinf.algorithms.online_residual_rl.remote_server import (
            RemoteInferenceServer,
        )
        from rlinf.algorithms.online_residual_rl.storage import PendingEpisodeStore

        if args.episodes != 1 or args.dummy_robot or args.leader_serial:
            raise ValueError(
                "--serve does not take robot-local episode or hardware flags."
            )
        if (
            not args.stage1_only
            and args.actor_checkpoint is None
            and args.resume_stage2 is None
            and not args.online_train
        ):
            if args.backend == "apx":
                raise ValueError(
                    "Real-robot APX serving requires a trained Stage2 Actor checkpoint."
                )
        if args.resume_stage2 and not args.actor_checkpoint:
            args.actor_checkpoint = args.resume_stage2
        policy = _build_policy(args, stage2, contract)
        mailbox = ActorMailbox() if args.online_train else None
        trainer = None
        try:
            if (
                args.backend == "torch"
                and not args.disable_rtc
                and args.rtc_guidance_mode == "jacobian"
            ):
                from rlinf.models.embodiment.openpi_rlinf.inference import (
                    warmup_jacobian_inference,
                )

                warmup_jacobian_inference(
                    policy,
                    _policy_input_shapes(robot_config),
                    str(robot_config["flexiv"]["task_description"]),
                    base_horizon=args.base_chunk_steps,
                    request_steps=args.base_request_steps,
                )
            if args.online_train:
                trainer = _build_online_training(
                    args, stage2, contract, policy, robot_config, mailbox
                )
            commit_episode = None
            if args.episode_dir is not None and not args.online_train:
                from rlinf.algorithms.online_residual_rl.replay import EpisodeReplay

                replay = EpisodeReplay(contract, storage_dir=args.episode_dir)

                def commit_episode(rows, success, terminal_frame):
                    replay.commit_episode(
                        rows, success=success, terminal_frame=terminal_frame
                    )

            server = RemoteInferenceServer(
                policy,
                contract,
                trainer,
                mailbox,
                expected_inputs=_policy_input_shapes(robot_config),
                backend=args.backend,
                base_horizon=args.base_chunk_steps,
                base_request_steps=args.base_request_steps,
                start_base=start_base,
                rtc_delay_buffer_size=args.rtc_delay_buffer_size,
                binary_gripper=args.binary_gripper,
                gripper_threshold=args.gripper_threshold,
                commit_episode=commit_episode,
                pending_store=(
                    PendingEpisodeStore(args.episode_dir, contract)
                    if args.episode_dir is not None
                    else None
                ),
            )
            try:
                logging.getLogger("flexiv.startup").info(
                    "Policy ready checkpoint=%s backend=%s stage2_enabled=%s online_train=%s actor_checkpoint=%s phase=%s fine_available=%s",
                    args.checkpoint,
                    args.backend,
                    not args.stage1_only,
                    bool(args.online_train),
                    args.actor_checkpoint,
                    server.initial_phase,
                    server.fine_available,
                )
                server.serve(args.host, args.port)
            finally:
                server.close()
        except BaseException:
            if trainer is None:
                policy.close()
            raise
        return 0
    if args.leader_serial and not args.operator_console:
        raise ValueError(
            "Leader drag requires --operator-console for explicit switch edges."
        )
    if args.episodes <= 0 or (
        args.episodes > 1 and not args.dummy_robot and not args.operator_console
    ):
        raise ValueError("Multiple real episodes require --operator-console re-arming.")
    if args.online_train and not (args.operator_console or args.terminal_label):
        raise ValueError("Online collection requires a human terminal-label source.")
    if args.resume_stage2 and not args.actor_checkpoint:
        args.actor_checkpoint = args.resume_stage2
    inference_executor = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="residual-apx")
        if args.backend == "apx"
        else None
    )
    try:
        policy = (
            inference_executor.submit(_build_policy, args, stage2, contract).result()
            if inference_executor is not None
            else _build_policy(args, stage2, contract)
        )
    except BaseException:
        if inference_executor is not None:
            inference_executor.shutdown(wait=True, cancel_futures=True)
        raise
    trainer = None
    active_runtime = None
    stop_requested = threading.Event()
    try:
        robot = _build_robot(
            robot_config, dummy=args.dummy_robot, leader_serial=args.leader_serial
        )
        from rlinf.algorithms.online_residual_rl.online import (
            ActorMailbox,
            EpisodeCollector,
            OperatorConsole,
        )

        mailbox = ActorMailbox() if args.online_train else None
        if args.online_train:
            trainer = _build_online_training(
                args, stage2, contract, policy, robot_config, mailbox
            )
        operator = OperatorConsole() if args.operator_console else None
    except BaseException:
        if inference_executor is not None:
            try:
                inference_executor.submit(policy.close).result()
            finally:
                inference_executor.shutdown(wait=True, cancel_futures=True)
        else:
            policy.close()
        if trainer is not None:
            trainer.close()
        raise

    def request_stop(_signum, _frame):
        stop_requested.set()
        if active_runtime is not None:
            active_runtime.abort()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        for episode in range(args.episodes):
            if stop_requested.is_set():
                break
            if episode:
                if operator is not None and not operator.wait_next(stop_requested):
                    break
                robot = _build_robot(
                    robot_config,
                    dummy=args.dummy_robot,
                    leader_serial=args.leader_serial,
                )
            episode_id = str(uuid.uuid4())
            collector = (
                EpisodeCollector(
                    contract,
                    policy.encode_receipts,
                    trainer.submit,
                    episode_id=episode_id,
                    pending_store=trainer.pending_store,
                    record_metrics=partial(
                        trainer.record_metrics, episode_id=episode_id
                    ),
                )
                if trainer is not None
                else None
            )
            active_runtime = ResidualHandoffRuntime(
                robot,
                policy,
                contract,
                ResidualDeploymentConfig(
                    prompt=str(robot_config["policy"]["prompt"]),
                    control_hz=float(robot_config["control"]["frequency_hz"]),
                    max_ticks=args.max_ticks,
                    task_timeout_seconds=args.task_timeout_seconds,
                ),
                project_actions=partial(project_flexiv_actions, config=robot.config),
                collector=collector,
                actor_mailbox=mailbox,
                operator=operator,
                terminal_label=(args.terminal_label == "success")
                if args.terminal_label is not None
                else None,
                own_policy=False,
                inference_executor=inference_executor,
            )
            result = active_runtime.run()
            aborted = active_runtime.abort_requested
            active_runtime = None
            if result != 0:
                return result
            if aborted:
                break
        return 0
    except Exception:
        logging.exception("Flexiv online residual inference failed.")
        if active_runtime is not None:
            active_runtime.close()
        return 1
    finally:
        try:
            if trainer is not None:
                trainer.close()
        finally:
            if inference_executor is not None:
                try:
                    inference_executor.submit(policy.close).result()
                finally:
                    inference_executor.shutdown(wait=True, cancel_futures=True)
            else:
                policy.close()


if __name__ == "__main__":
    raise SystemExit(main())
