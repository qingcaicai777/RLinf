# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Stage1 PyTorch PI0.5 and Stage2 residual inference for Flexiv rollout."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation

from rlinf.algorithms.online_residual_rl import OnlineResidualRLConfig, ResidualActor
from rlinf.algorithms.online_residual_rl.checkpoint import load_actor_state
from rlinf.models.embodiment.openpi_rlinf import get_model

logger = logging.getLogger(__name__)


def resolve_stage1_norm_stats(
    training_config: str | Path,
    checkpoint: str | Path,
    override: str | Path | None = None,
) -> Path:
    """Resolve the frozen Stage1 action and state statistics for deployment."""
    checkpoint_path = Path(checkpoint).expanduser()
    root = (
        checkpoint_path.parent.parent if checkpoint_path.is_file() else checkpoint_path
    )
    if override is not None:
        candidates = [Path(override).expanduser()]
    else:
        model_cfg = OmegaConf.load(training_config).actor.model
        asset_dir = (
            "flexiv_joint_assets"
            if "joint_absolute" in str(model_cfg.openpi.config_name)
            else "flexiv_delta_assets"
        )
        candidates = [
            root / asset_dir / "norm_stats.json",
            Path(str(model_cfg.openpi_data.norm_stats_path)).expanduser(),
        ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Stage1 norm_stats.json was not found. Supply --norm-stats or place it "
        f"beside the checkpoint at {candidates[0]}."
    )


def validate_stage1_norm_stats(path: str | Path, action_dim: int) -> None:
    """Check state statistics and physical action channels before loading weights."""
    try:
        stats = json.loads(Path(path).read_text())["norm_stats"]
        for name in ("state", "actions"):
            for key in ("mean", "std", "q01", "q99"):
                stats[name][key]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"{path} requires state and actions normalization statistics."
        ) from exc
    for name, width in (("state", 8), ("actions", action_dim)):
        arrays = {}
        for key in ("mean", "std", "q01", "q99"):
            value = np.asarray(stats[name][key], dtype=np.float64)
            if (
                value.ndim != 1
                or len(value) not in ({8} if name == "state" else {action_dim, 32})
                or not np.isfinite(value).all()
                or (name == "actions" and np.any(value[width:] != 0))
            ):
                raise ValueError(
                    f"{name}.{key} must contain finite statistics for {width} channels."
                )
            arrays[key] = value[:width]
        if np.any(arrays["std"] <= 0) or np.any(arrays["q99"] <= arrays["q01"]):
            raise ValueError(
                f"{name} physical channels have degenerate normalization statistics."
            )


def build_apx_stage1_model_config(
    deployment_model_cfg: DictConfig,
    *,
    training_config: str | Path,
    checkpoint: str | Path,
    norm_stats_path: str | Path | None,
    apx_checkpoint: str | Path,
    action_representation: str,
    num_flow_steps: int,
    stage1_only: bool,
    actor_checkpoint: str | Path | None = None,
) -> DictConfig:
    """Select APX transforms and physical decoding for a Flexiv Stage1 recipe."""
    if action_representation not in {"eef_delta", "joint_absolute"}:
        raise ValueError("Unsupported Flexiv action representation.")
    expected_width = 8 if action_representation == "joint_absolute" else 7
    stage1 = OmegaConf.load(training_config).actor.model
    config_name = str(stage1.openpi.config_name)
    if (
        int(stage1.action_dim) != expected_width
        or int(stage1.openpi.action_env_dim) != expected_width
        or int(stage1.openpi.action_horizon) != 32
        or int(stage1.openpi.model_action_dim) != 32
        or ("joint_absolute" in config_name)
        != (action_representation == "joint_absolute")
    ):
        raise ValueError(
            "Flexiv APX Stage1 recipe shape or transforms differ from "
            f"action_representation={action_representation}."
        )
    stats_path = resolve_stage1_norm_stats(training_config, checkpoint, norm_stats_path)
    validate_stage1_norm_stats(stats_path, expected_width)

    model_cfg = OmegaConf.create(
        OmegaConf.to_container(deployment_model_cfg, resolve=True)
    )
    model_cfg.model_path = str(checkpoint)
    model_cfg.action_dim = expected_width
    model_cfg.openpi.config_name = config_name
    model_cfg.openpi.num_steps = num_flow_steps
    model_cfg.apxinf.num_flow_steps = num_flow_steps
    if stage1_only:
        model_cfg.apxinf.return_prefix_hidden_states = False
    model_cfg.openpi_data = OmegaConf.create(
        OmegaConf.to_container(stage1.openpi_data, resolve=True)
    )
    model_cfg.openpi_data.norm_stats_path = str(stats_path)
    model_cfg.apxinf.checkpoint = str(apx_checkpoint)
    model_cfg.rlt.checkpoint = str(checkpoint)
    model_cfg.actor_checkpoint = str(actor_checkpoint) if actor_checkpoint else None
    return model_cfg


def load_stage1_model(
    training_config: str | Path,
    checkpoint: str | Path,
    device: str,
    *,
    norm_stats_path: str | Path | None = None,
    num_flow_steps: int = 10,
    use_cuda_graph: bool = True,
    rtc_max_guidance_weight: float = 10.0,
    rtc_guidance_mode: str = "approx",
    action_representation: str = "eef_delta",
) -> Any:
    """Load the exact Stage1 model shape and transforms from its SFT recipe."""
    if num_flow_steps <= 0:
        raise ValueError("Inference flow steps must be positive.")
    if rtc_max_guidance_weight <= 0:
        raise ValueError("RTC maximum guidance weight must be positive.")
    if rtc_guidance_mode not in {"approx", "jacobian"}:
        raise ValueError("RTC guidance mode must be 'approx' or 'jacobian'.")
    recipe_path = Path(training_config)
    recipe = OmegaConf.load(recipe_path)
    template = OmegaConf.load(recipe_path.parent / "model" / "pi0_5_rlinf.yaml")
    model_cfg = OmegaConf.merge(template, recipe.actor.model)
    OmegaConf.resolve(model_cfg)
    model_cfg.model_path = str(checkpoint)
    model_cfg.openpi_data.norm_stats_path = str(
        resolve_stage1_norm_stats(training_config, checkpoint, norm_stats_path)
    )
    model_cfg.num_steps = num_flow_steps
    model_cfg.openpi.num_steps = num_flow_steps
    model_cfg.openpi.task = "eval"
    if not model_cfg.openpi.use_rlt or not model_cfg.openpi.rlt_normalized_actions:
        raise ValueError("Stage1 inference needs use_rlt and rlt_normalized_actions.")
    expected_width = 8 if action_representation == "joint_absolute" else 7
    if action_representation not in {"eef_delta", "joint_absolute"}:
        raise ValueError("Unsupported Flexiv action representation.")
    if (
        int(model_cfg.openpi.action_horizon) != 32
        or int(model_cfg.action_dim) != expected_width
        or int(model_cfg.openpi.action_env_dim) != expected_width
        or int(model_cfg.openpi.model_action_dim) != 32
        or ("joint_absolute" in str(model_cfg.openpi.config_name))
        != (action_representation == "joint_absolute")
    ):
        raise ValueError(
            "Flexiv Stage1 model shape or transforms differ from action representation."
        )
    validate_stage1_norm_stats(model_cfg.openpi_data.norm_stats_path, expected_width)
    model = get_model(model_cfg)
    model = model.to(device).eval()
    model.rtc_guidance_clip = float(rtc_max_guidance_weight)
    model.rtc_guidance_mode = rtc_guidance_mode
    graph_enabled = use_cuda_graph
    # Guided steps use compiled autograd; bootstrap and requests without an
    # overlap use the independent no-gradient CUDA graph path.
    if rtc_guidance_mode == "jacobian":
        model.requires_grad_(False)
        compile_jacobian = getattr(model, "enable_inference_jacobian_compile", None)
        if callable(compile_jacobian):
            compile_jacobian()
    if graph_enabled and torch.device(device).type == "cuda":
        model.enable_inference_cuda_graph()
    return model


def extract_stage1_features(
    model: Any,
    env_obs: dict[str, Any],
    *,
    committed: torch.Tensor | None,
    inference_horizon: int,
    rtc_delay_steps: int | None = None,
) -> dict[str, torch.Tensor]:
    """Sample Stage1 with guidance from remaining raw model-space actions."""
    rtc_context = _rtc_context(
        model, env_obs, committed, inference_horizon, delay_steps=rtc_delay_steps
    )
    return model.extract_rlt_obs(env_obs, rtc_context=rtc_context)


def _rtc_context(
    model: Any,
    env_obs: dict[str, Any],
    committed: torch.Tensor | None,
    inference_horizon: int,
    delay_steps: int | None = None,
) -> Any:
    """Validate the accepted raw chunk before RTC-guided sampling."""
    rtc_context = None
    if committed is not None:
        from rlinf.models.embodiment.openpi.rtc_guidance import RTCGuidanceContext

        committed = torch.as_tensor(committed, device=model.device, dtype=torch.float32)
        if (
            committed.ndim != 3
            or committed.shape[0] != len(env_obs["states"])
            or not 1 <= committed.shape[1] <= model.action_horizon
            or committed.shape[2] != model.action_dim
            or not torch.isfinite(committed).all()
        ):
            raise ValueError(
                "RTC raw remaining chunk must be finite [B, 1..32, model_action_dim]."
            )
        rtc_context = RTCGuidanceContext(
            prev_model_actions=committed,
            executed_horizon=0,
            delay_steps=inference_horizon if delay_steps is None else delay_steps,
        )
    return rtc_context


def warmup_jacobian_inference(
    policy: Any,
    input_shapes: dict[str, tuple[int, ...]],
    prompt: str,
    *,
    base_horizon: int,
    request_steps: int,
) -> None:
    """Compile and warm the guided Stage1 path before opening the robot socket."""
    base_policy = getattr(policy, "base_policy", policy)
    if getattr(base_policy, "rtc_guidance_mode", None) != "jacobian":
        return
    base_predict = getattr(policy, "predict_base_action_batch", None)
    if not callable(base_predict):
        raise TypeError("Jacobian warmup requires a Stage1 base prediction API.")
    if not 1 <= base_horizon <= 32 or not 0 <= request_steps < base_horizon:
        raise ValueError("Jacobian warmup received an invalid base schedule.")
    if not prompt:
        raise ValueError("Jacobian warmup requires the robot task prompt.")
    model = getattr(policy, "model", None)
    if model is None:
        model = base_policy.model
    device = torch.device(model.device)

    observation = {
        name: np.zeros(
            (1, *shape), dtype=np.uint8 if name.endswith("images") else np.float32
        )
        for name, shape in input_shapes.items()
    }
    observation["task_descriptions"] = [prompt]
    logger.info("Warming the Jacobian RTC path before accepting a robot client.")
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    try:
        _actions, metadata = base_predict(observation, horizon=base_horizon)
        raw_actions = metadata.get("raw_model_actions")
        if (
            not torch.is_tensor(raw_actions)
            or raw_actions.ndim != 3
            or raw_actions.shape[1] <= request_steps
        ):
            raise ValueError("Stage1 warmup did not return a usable raw action chunk.")

        # The robot's bootstrap is followed by a same-frame guided warmup. This
        # second call triggers lazy Torch compilation before socket deadlines.
        base_predict(
            observation,
            horizon=base_horizon,
            committed=raw_actions[:, request_steps:].detach(),
            rtc_delay_steps=0,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    finally:
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, device)
        torch.random.set_rng_state(cpu_rng_state)
    logger.info("Jacobian RTC warmup complete; the inference server is ready.")


def rebase_rtc_world_deltas(
    model: Any,
    actions: torch.Tensor,
    previous_pose: np.ndarray,
    current_pose: np.ndarray,
) -> torch.Tensor:
    """Express old model-space TCP deltas relative to the current observation."""
    previous = np.asarray(previous_pose, dtype=np.float64)
    current = np.asarray(current_pose, dtype=np.float64)
    if (
        actions.ndim != 3
        or actions.shape[0] != 1
        or actions.shape[-1] < 7
        or previous.shape != (7,)
        or current.shape != (7,)
        or not np.isfinite(previous).all()
        or not np.isfinite(current).all()
    ):
        raise ValueError("RTC rebasing requires [1,H,32] actions and two TCP poses.")
    physical = model.denormalize_rlt_actions(actions[..., :7]).cpu().numpy()
    rebased = physical.copy()
    rebased[..., :3] += previous[:3] - current[:3]
    previous_rotation = Rotation.from_quat(
        [previous[4], previous[5], previous[6], previous[3]]
    )
    current_rotation = Rotation.from_quat(
        [current[4], current[5], current[6], current[3]]
    )
    target_rotation = (
        Rotation.from_rotvec(physical[..., 3:6].reshape(-1, 3)) * previous_rotation
    )
    rebased[..., 3:6] = (
        (target_rotation * current_rotation.inv())
        .as_rotvec()
        .reshape(physical.shape[:-1] + (3,))
    )
    result = actions.clone()
    result[..., :7] = model.normalize_rlt_actions(
        torch.as_tensor(rebased, dtype=result.dtype, device=result.device)
    )
    return result


class OpenPIRLInfStage1Adapter:
    """Expose the full base prediction and raw PI0.5 RTC context."""

    actor_version = 0
    supports_fine_phase = False

    def __init__(
        self,
        model: Any,
        inference_horizon: int,
        *,
        use_rtc: bool = True,
        rtc_guidance_mode: str = "approx",
    ) -> None:
        if not 1 <= inference_horizon <= model.action_horizon:
            raise ValueError("Base execution steps must fit in the model horizon.")
        self.model = model.eval()
        self.inference_horizon = inference_horizon
        self.conditions_on_committed = use_rtc
        self.rtc_guidance_mode = rtc_guidance_mode

    @torch.inference_mode()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        *,
        mode: str = "eval",
        committed: torch.Tensor | None = None,
        rtc_delay_steps: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return the physical model horizon with raw model-space metadata."""
        if mode != "eval":
            raise ValueError("Stage1-only deployment supports eval only.")
        if not self.conditions_on_committed or committed is None:
            physical, metadata = self.model.predict_action_batch(env_obs, mode="eval")
            if physical.shape[1] < self.inference_horizon:
                raise ValueError("Stage1 prediction is shorter than the base chunk.")
            return physical, {"raw_model_actions": metadata["model_actions"]}
        from rlinf.models.embodiment.openpi_rlinf.sampling.rtc_guidance import (
            sample_actions_with_rtc_guidance,
        )

        rtc_context = _rtc_context(
            self.model,
            env_obs,
            committed,
            self.inference_horizon,
            delay_steps=rtc_delay_steps,
        )
        context = (
            torch.inference_mode(False)
            if self.rtc_guidance_mode == "jacobian"
            else torch.inference_mode()
        )
        with context:
            observation = self.model.env_obs_to_observation(env_obs)
            model_actions = sample_actions_with_rtc_guidance(
                self.model,
                observation,
                rtc_context,
                num_steps=self.model.num_steps,
                guidance_clip=self.model.rtc_guidance_clip,
                guidance_mode=self.rtc_guidance_mode,
            )
        reference = model_actions[:, :, : self.model.action_env_dim]
        physical = self.model.decode_rlt_actions(reference, observation.state)
        return physical, {"raw_model_actions": model_actions}

    def predict_base_action_batch(
        self,
        env_obs: dict[str, Any],
        *,
        horizon: int,
        committed: torch.Tensor | None = None,
        rtc_delay_steps: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Sample one base prediction for the configured execution horizon."""
        if horizon != self.inference_horizon:
            raise ValueError("Base horizon differs from the loaded policy.")
        return self.predict_action_batch(
            env_obs, committed=committed, rtc_delay_steps=rtc_delay_steps
        )

    def encode_receipts(self, physical_actions: torch.Tensor) -> torch.Tensor:
        """Normalize physical receipts with the frozen Stage1 statistics."""
        return self.model.normalize_rlt_actions(physical_actions)

    def rebase_committed_actions(
        self,
        actions: torch.Tensor,
        previous_pose: np.ndarray,
        current_pose: np.ndarray,
    ) -> torch.Tensor:
        """Align RTC overlap with the current TCP reference pose."""
        if int(self.model.action_env_dim) == 8:
            return actions
        return rebase_rtc_world_deltas(self.model, actions, previous_pose, current_pose)

    def close(self) -> None:
        """Release the loaded Stage1 model reference."""
        model = self.model
        disable_graph = getattr(model, "disable_inference_cuda_graph", None)
        if callable(disable_graph):
            disable_graph()
        disable_jacobian_compile = getattr(
            model, "disable_inference_jacobian_compile", None
        )
        if callable(disable_jacobian_compile):
            disable_jacobian_compile()
        self.model = None


class OpenPIRLInfResidualAdapter:
    """Use one Stage1 prefix pass for its token and 32-step base reference."""

    def __init__(
        self,
        model: Any,
        config: OnlineResidualRLConfig,
        *,
        actor_checkpoint: str | Path | None = None,
        actor: ResidualActor | None = None,
        base_horizon: int = 6,
        rtc_guidance_mode: str = "approx",
    ) -> None:
        if not model.rlt_cfg.use_rlt or not model.rlt_cfg.rlt_normalized_actions:
            raise ValueError("Stage1 model must expose normalized RLT features.")
        if model.action_horizon < config.actor_horizon:
            raise ValueError("Stage1 model horizon is shorter than the Actor horizon.")
        if int(model.action_env_dim) != config.action_dim:
            raise ValueError("Stage1 physical action width differs from Actor width.")
        self.model = model.eval()
        self.config = config
        self.base_policy = OpenPIRLInfStage1Adapter(
            model, base_horizon, rtc_guidance_mode=rtc_guidance_mode
        )
        self.actor = (actor or ResidualActor(config)).to(model.device).eval()
        if actor_checkpoint is not None:
            self.actor.load_state_dict(
                load_actor_state(actor_checkpoint, config), strict=True
            )
        self.actor.requires_grad_(False)
        self.conditions_on_committed = True
        self.actor_version = 0
        self.metadata = {
            "backend": "openpi_rlinf",
            "action_horizon": int(model.action_horizon),
            "action_dim": config.action_dim,
            "rlt": True,
        }

    def predict_base_action_batch(
        self,
        env_obs: dict[str, Any],
        *,
        horizon: int,
        committed: torch.Tensor | None = None,
        rtc_delay_steps: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Sample the full base horizon without running the residual Actor."""
        return self.base_policy.predict_base_action_batch(
            env_obs,
            horizon=horizon,
            committed=committed,
            rtc_delay_steps=rtc_delay_steps,
        )

    @torch.inference_mode()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        *,
        mode: str = "eval",
        committed: torch.Tensor | None = None,
        rtc_delay_steps: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return the 12-step proposal; only steps 6:12 may be committed."""
        if mode not in {"eval", "base", "rollout"}:
            raise ValueError(f"Unsupported inference mode: {mode!r}.")
        grad_context = (
            torch.inference_mode(False)
            if self.base_policy.rtc_guidance_mode == "jacobian"
            and committed is not None
            else torch.inference_mode()
        )
        with grad_context:
            features = extract_stage1_features(
                self.model,
                env_obs,
                committed=committed,
                inference_horizon=self.config.inference_horizon,
                rtc_delay_steps=rtc_delay_steps,
            )
        z = features["z_rl"]
        proprio = features["proprio"]
        reference = features["ref_chunk"][:, : self.config.actor_horizon]
        if reference.shape[1:] != (self.config.actor_horizon, self.config.action_dim):
            raise ValueError(f"Stage1 reference has invalid shape {reference.shape}.")
        if not torch.isfinite(reference).all() or not torch.isfinite(z).all():
            raise ValueError("Stage1 produced non-finite reference or RL token.")
        noise = None
        if mode == "rollout" and self.config.rollout_noise_std > 0.0:
            noise = torch.randn_like(reference) * self.config.rollout_noise_std
            noise[:, : self.config.inference_horizon] = 0
        actor_output = (
            reference
            if mode == "base"
            else self.actor(z, proprio, reference, pre_tanh_noise=noise)
        )
        if committed is not None and committed.shape[0] != z.shape[0]:
            raise ValueError("Committed prefix batch differs from Stage1 observation.")
        proposal = None
        physical = self.model.decode_rlt_actions(actor_output, features["model_state"])
        return physical, {
            "normalized_actions": actor_output,
            "raw_model_actions": features["raw_model_actions"],
            "forward_inputs": {
                "z_rl": z,
                "proprio": proprio,
                "base_ref": reference,
                "committed": committed,
                "actor_output": actor_output,
                "policy_proposal": proposal,
            },
        }

    def encode_receipts(self, physical_actions: torch.Tensor) -> torch.Tensor:
        """Normalize final accepted physical commands with Stage1 statistics."""
        return self.model.normalize_rlt_actions(physical_actions)

    def rebase_committed_actions(
        self,
        actions: torch.Tensor,
        previous_pose: np.ndarray,
        current_pose: np.ndarray,
    ) -> torch.Tensor:
        """Align RTC overlap with the current TCP reference pose."""
        return self.base_policy.rebase_committed_actions(
            actions, previous_pose, current_pose
        )

    @torch.inference_mode()
    def install_actor(self, state: dict[str, torch.Tensor], version: int) -> None:
        """Install a complete published Actor before the next inference request."""
        if version <= self.actor_version:
            raise ValueError("Published Actor version must increase.")
        self.actor.load_state_dict(state, strict=True)
        self.actor.eval()
        self.actor_version = version

    def close(self) -> None:
        """Release references; the caller owns the loaded model lifetime."""
        self.base_policy.close()
        self.model = None
        self.actor = None
