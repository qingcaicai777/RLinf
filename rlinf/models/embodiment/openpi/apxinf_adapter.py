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

"""OpenPI-specific adapter for the ApxInf bare-model rollout backend.

RLinf owns the complete OpenPI input/output transform chain. ApxInf receives
only resized uint8 RGB views, token ids and optional flow noise through its L1
``Model.infer_rgb`` API, and returns normalized model-space actions.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(value)


def _normalize_cuda_prefix(
    prefix_view: Any, scale_view: Any, eps: float
) -> torch.Tensor:
    """Normalize APX BF16 CUDA views without staging prefix values on the host."""
    prefix = torch.as_tensor(prefix_view).view(torch.bfloat16).float()
    scale = torch.as_tensor(scale_view).view(torch.bfloat16).float()
    if prefix.ndim != 2 or scale.shape != (prefix.shape[-1],):
        raise ValueError("ApxInf CUDA prefix and RMSNorm scale shapes disagree.")
    normalized = torch.nn.functional.rms_norm(
        prefix, (prefix.shape[-1],), weight=scale, eps=eps
    )
    # The finite check also synchronizes PyTorch's stream before its zero-copy
    # views can release the APX-owned CUDA allocations.
    if not bool(torch.isfinite(normalized).all()):
        raise FloatingPointError("ApxInf returned non-finite CUDA prefix features")
    return normalized


def _batch_size(env_obs: Mapping[str, Any]) -> int:
    for value in env_obs.values():
        if torch.is_tensor(value) or isinstance(value, np.ndarray):
            return int(value.shape[0])
        if isinstance(value, (list, tuple)):
            return len(value)
    raise ValueError("Cannot infer batch size from RLinf observation")


def _item(value: Any, index: int) -> Any:
    if torch.is_tensor(value):
        return _to_numpy(value[index])
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value[index])
    if isinstance(value, (list, tuple)):
        return value[index]
    raise TypeError(f"Unsupported batched observation value: {type(value)!r}")


def _active_token_ids(transformed: Mapping[str, Any]) -> np.ndarray:
    padded_token_ids = np.asarray(transformed["tokenized_prompt"])
    token_mask = np.asarray(transformed["tokenized_prompt_mask"], dtype=np.bool_)
    if padded_token_ids.shape != token_mask.shape:
        raise ValueError(
            "RLinf OpenPI token ids and mask have different shapes: "
            f"{padded_token_ids.shape} and {token_mask.shape}"
        )
    # OpenPI pads prompts to max_token_len and passes a separate mask to its
    # PyTorch model. ApxInf L1 has no mask argument, so give it only the active
    # prefix; otherwise padding would become real text tokens.
    token_ids = np.ascontiguousarray(padded_token_ids[token_mask], dtype=np.uint32)
    if token_ids.size == 0:
        raise ValueError("RLinf OpenPI produced an empty token sequence")
    return token_ids


class _RLInfOpenPITransforms:
    """Build and run RLinf's native OpenPI transforms without PyTorch weights."""

    def __init__(self, model_cfg: DictConfig | Mapping[str, Any]) -> None:
        import openpi.shared.download as download
        import openpi.transforms as transforms
        from openpi.models.model import IMAGE_KEYS
        from openpi.shared import normalize as normalize_lib
        from openpi.training import checkpoints

        from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

        self.model_cfg = model_cfg
        self.num_action_chunks = int(model_cfg.get("num_action_chunks"))
        self.action_dim = int(model_cfg.get("action_dim"))
        openpi_cfg = model_cfg.get("openpi", {})
        config_name = str(openpi_cfg.get("config_name", "pi05_libero"))
        data_kwargs = model_cfg.get("openpi_data", None)
        train_config = get_openpi_config(
            config_name,
            model_path=model_cfg.get("model_path"),
            data_kwargs=data_kwargs,
        )

        # Apply only genuine OpenPI model-config overrides. RLinf-only sampling
        # fields (for example noise_method) are intentionally ignored here.
        model_config = train_config.model
        model_fields = {field.name for field in dataclasses.fields(model_config)}
        overrides = {
            key: value for key, value in openpi_cfg.items() if key in model_fields
        }
        if overrides:
            model_config = dataclasses.replace(model_config, **overrides)

        checkpoint_dir = pathlib.Path(
            download.maybe_download(str(model_cfg.get("model_path")))
        )
        data_config = train_config.data.create(
            train_config.assets_dirs,
            model_config,
        )

        norm_stats_path = (
            data_kwargs.get("norm_stats_path", None)
            if data_kwargs is not None
            else None
        )
        if norm_stats_path:
            norm_dir = pathlib.Path(str(norm_stats_path)).expanduser()
            if norm_dir.is_file():
                norm_dir = norm_dir.parent
            norm_stats = normalize_lib.load(norm_dir)
        elif (checkpoint_dir / "norm_stats.json").is_file():
            # Converted LeRobot/ApxInf checkpoints keep stats at the root.
            norm_stats = normalize_lib.load(checkpoint_dir)
        elif data_config.norm_stats is not None:
            norm_stats = data_config.norm_stats
        else:
            if data_config.asset_id is None:
                raise ValueError("OpenPI asset_id is required to load norm stats")
            norm_stats = checkpoints.load_norm_stats(
                checkpoint_dir,
                data_config.asset_id,
            )

        self._input_transform = transforms.compose(
            [
                transforms.InjectDefaultPrompt(None),
                *data_config.data_transforms.inputs,
                transforms.Normalize(
                    norm_stats,
                    use_quantiles=data_config.use_quantile_norm,
                ),
                *data_config.model_transforms.inputs,
            ]
        )
        self._output_transform = transforms.compose(
            [
                *data_config.model_transforms.outputs,
                transforms.Unnormalize(
                    norm_stats,
                    use_quantiles=data_config.use_quantile_norm,
                ),
                *data_config.data_transforms.outputs,
            ]
        )
        self._action_norm_stats = norm_stats["actions"]
        self._use_quantile_norm = bool(data_config.use_quantile_norm)
        self._image_keys = tuple(IMAGE_KEYS)
        self._config_name = config_name
        state_indices = openpi_cfg.get("state_indices", None)
        self._state_indices = list(state_indices) if state_indices else None

    def normalize_action_batch(self, actions: Any) -> torch.Tensor:
        """Map final physical action receipts back to the model's action domain."""
        source = torch.as_tensor(actions, dtype=torch.float32)
        width = source.shape[-1]

        def value(name: str) -> torch.Tensor:
            array = getattr(self._action_norm_stats, name)
            return torch.as_tensor(
                array[:width], device=source.device, dtype=source.dtype
            )

        if self._use_quantile_norm:
            low, high = value("q01"), value("q99")
            return (source - low) / (high - low + 1.0e-6) * 2.0 - 1.0
        return (source - value("mean")) / (value("std") + 1.0e-6)

    def action_mapping(self, lower: Any, upper: Any):
        """Build the learner/rollout projection from the loaded fixed statistics."""
        from rlinf.algorithms.online_residual_rl.action_mapping import ActionMapping

        stats = self._action_norm_stats
        if self._use_quantile_norm:
            offset = np.asarray(stats.q01[: self.action_dim])
            scale = (np.asarray(stats.q99[: self.action_dim]) - offset + 1e-6) / 2
            offset = offset + scale
        else:
            offset = np.asarray(stats.mean[: self.action_dim])
            scale = np.asarray(stats.std[: self.action_dim]) + 1e-6
        return ActionMapping(offset, scale, lower, upper)

    def _select_state(self, states: Any) -> Any:
        if not self._state_indices:
            return states
        state_dim = int(states.shape[-1])
        if state_dim == len(self._state_indices):
            return states
        if state_dim <= max(self._state_indices):
            raise ValueError(
                f"Cannot select state_indices={self._state_indices} "
                f"from state dim {state_dim}"
            )
        if torch.is_tensor(states):
            indices = torch.as_tensor(self._state_indices, device=states.device)
            return states.index_select(-1, indices)
        return np.asarray(states)[..., self._state_indices]

    def _policy_observation(self, env_obs: Mapping[str, Any]) -> dict[str, Any]:
        required = ("main_images", "states", "task_descriptions")
        missing = [key for key in required if key not in env_obs]
        if missing:
            raise KeyError(f"RLinf observation is missing keys: {missing}")

        states = self._select_state(env_obs["states"])
        observation: dict[str, Any] = {
            "observation/image": env_obs["main_images"],
            "prompt": env_obs["task_descriptions"],
        }
        if "calvin" in self._config_name:
            observation["observation/state_ee_pos"] = states[:, :3]
            observation["observation/state_ee_rot"] = states[:, 3:6]
            observation["observation/state_gripper"] = states[:, 6:7]
        else:
            observation["observation/state"] = states
        if env_obs.get("wrist_images") is not None:
            observation["observation/wrist_image"] = env_obs["wrist_images"]
        if env_obs.get("extra_view_images") is not None:
            observation["observation/extra_view_image"] = env_obs["extra_view_images"]
        return observation

    def preprocess_batch(
        self,
        env_obs: Mapping[str, Any],
        *,
        num_views: int,
        image_size: int,
    ) -> list[dict[str, np.ndarray]]:
        """Return one L1 ApxInf request and postprocess context per env."""
        policy_obs = self._policy_observation(env_obs)
        batch_size = _batch_size(policy_obs)
        for key, value in policy_obs.items():
            if _batch_size({key: value}) != batch_size:
                raise ValueError(f"Observation field {key!r} has mismatched batch size")

        prepared = []
        for index in range(batch_size):
            sample = {key: _item(value, index) for key, value in policy_obs.items()}
            transformed = self._input_transform(sample)
            images = transformed["image"]
            masks = transformed["image_mask"]
            active_keys = [key for key in self._image_keys if bool(masks[key])]
            if len(active_keys) != num_views:
                raise ValueError(
                    "RLinf OpenPI transforms produced "
                    f"{len(active_keys)} active camera views {active_keys}, but "
                    f"the ApxInf checkpoint expects {num_views}"
                )
            rgb_u8 = np.ascontiguousarray(
                np.stack([images[key] for key in active_keys]),
                dtype=np.uint8,
            )
            expected_rgb_shape = (num_views, image_size, image_size, 3)
            if rgb_u8.shape != expected_rgb_shape:
                raise ValueError(
                    f"RLinf OpenPI RGB shape {rgb_u8.shape}, expected "
                    f"{expected_rgb_shape}"
                )
            token_ids = _active_token_ids(transformed)
            prepared.append(
                {
                    "rgb_u8": rgb_u8,
                    "token_ids": token_ids,
                    "state": np.ascontiguousarray(transformed["state"]),
                }
            )
        return prepared

    def postprocess_batch(
        self,
        normalized_actions: np.ndarray,
        prepared: list[dict[str, np.ndarray]],
        *,
        chunk_len: int | None = None,
    ) -> torch.Tensor:
        """Run RLinf's native OpenPI output transforms on normalized actions."""
        chunk_len = self.num_action_chunks if chunk_len is None else int(chunk_len)
        rows = []
        for actions, context in zip(normalized_actions, prepared, strict=True):
            output = self._output_transform(
                {
                    "actions": np.asarray(actions, dtype=np.float32),
                    "state": context["state"],
                }
            )
            row = np.asarray(output["actions"], dtype=np.float32)
            if row.ndim != 2:
                raise ValueError(
                    f"RLinf OpenPI actions must be rank 2, got {row.shape}"
                )
            if row.shape[0] < chunk_len:
                raise ValueError(
                    f"RLinf OpenPI postprocess returned {row.shape[0]} actions, "
                    f"but {chunk_len} must be executed"
                )
            if row.shape[1] != self.action_dim:
                raise ValueError(
                    f"RLinf OpenPI postprocess returned action_dim={row.shape[1]}, "
                    f"expected {self.action_dim}"
                )
            rows.append(row[:chunk_len])
        return torch.from_numpy(np.stack(rows))


class OpenPIApxInfAdapter:
    """Run ApxInf bare PI0.5 inference behind RLinf's OpenPI transforms."""

    def __init__(
        self,
        model_cfg: DictConfig | Mapping[str, Any],
        device: str,
        *,
        model: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        self.model_cfg = model_cfg
        self.apxinf_cfg = model_cfg.get("apxinf", {})
        self.action_dim = int(model_cfg.get("action_dim", 7))
        self.num_action_chunks = int(model_cfg.get("num_action_chunks"))
        self.action_horizon = int(self.apxinf_cfg.get("action_horizon", 10))
        self.num_flow_steps = int(
            self.apxinf_cfg.get(
                "num_flow_steps", model_cfg.get("openpi", {}).get("num_steps", 10)
            )
        )
        self.noise_source = str(self.apxinf_cfg.get("noise_source", "apxinf"))
        self.seed = int(self.apxinf_cfg.get("seed", 0))
        self.return_prefix_hidden_states = bool(
            self.apxinf_cfg.get("return_prefix_hidden_states", False)
        )
        self.device = str(device)

        self._validate_config()
        self.model = model if model is not None else self._load_model()
        self._closed = False
        self.processor = (
            processor if processor is not None else _RLInfOpenPITransforms(model_cfg)
        )
        self._validate_model_contract()

        self._noise_generator = None
        if self.noise_source == "torch":
            self._noise_generator = torch.Generator(device=self.device)
            self._noise_generator.manual_seed(self.seed)

        self.metadata = {
            "backend": "rlinf-apxinf-l1",
            "model_type": "pi05",
            "action_horizon": int(self.model.action_horizon),
            "model_action_dim": int(self.model.action_dim),
            "num_views": int(self.model.num_views),
            "image_size": int(self.model.image_size),
        }

    def _validate_config(self) -> None:
        if self.num_action_chunks <= 0:
            raise ValueError("rollout.model.num_action_chunks must be positive")
        if self.action_horizon < self.num_action_chunks:
            raise ValueError(
                "ApxInf action_horizon must be at least num_action_chunks; got "
                f"{self.action_horizon} < {self.num_action_chunks}"
            )
        if self.action_dim <= 0:
            raise ValueError("rollout.model.action_dim must be positive")
        if self.num_flow_steps <= 0:
            raise ValueError("ApxInf num_flow_steps must be positive")
        if self.noise_source not in {"apxinf", "torch", "observation"}:
            raise ValueError(
                "ApxInf noise_source must be one of: apxinf, torch, observation"
            )

        openpi_cfg = self.model_cfg.get("openpi", {})
        config_name = str(openpi_cfg.get("config_name", "pi05_libero"))
        if not config_name.startswith("pi05_"):
            raise ValueError(
                "The ApxInf bare-model adapter currently supports PI0.5 OpenPI "
                f"configs; got {config_name!r}"
            )
        noise_method = str(openpi_cfg.get("noise_method", "flow_ode"))
        if noise_method not in {"flow_ode", "flow_sde"}:
            raise ValueError(
                "ApxInf PI0.5 implements flow ODE inference; only flow_ode, or "
                "RLinf eval-mode flow_sde (which resolves to ODE), is supported; "
                f"got {noise_method!r}"
            )

    def _load_model(self):
        if bool(self.apxinf_cfg.get("disable_fa2_splitkv", False)):
            # The Flexiv SM89 deployment has produced non-finite actions with
            # split-KV enabled, so select regular FA2 before model loading.
            os.environ.setdefault("APXINF_DISABLE_FA2_SPLITKV", "1")
        try:
            import apxinf_py
        except ImportError as error:
            raise ImportError(
                "apxinf_py is not importable. Build the vendored "
                "APXinf-robo/apxinf/crates/apxinf-py binding with maturin "
                "--features cuda and install its wheel in this Python environment."
            ) from error

        model_path = self.model_cfg.get("model_path")
        if not model_path:
            raise ValueError("rollout.model.model_path is required for ApxInf")
        model_dir = pathlib.Path(str(model_path))
        checkpoint = self.apxinf_cfg.get("checkpoint", None)
        checkpoint_path = (
            pathlib.Path(str(checkpoint))
            if checkpoint
            else model_dir / "model.safetensors"
        )

        tactics = self.apxinf_cfg.get("tactics", None)
        if tactics is None:
            try:
                from apxinf._tactics import resolve_pi05_tactics

                tactics = resolve_pi05_tactics(
                    self.device,
                    str(self.apxinf_cfg.get("precision", "bf16")),
                    model_dir=model_dir,
                    override=None,
                    allow_missing=bool(self.apxinf_cfg.get("autotune", False)),
                )
            except ImportError:
                tactics = None

        kwargs: dict[str, Any] = {
            "device": self.device,
            "precision": str(self.apxinf_cfg.get("precision", "bf16")),
            "autotune": bool(self.apxinf_cfg.get("autotune", False)),
            "action_horizon": self.action_horizon,
            "num_flow_steps": self.num_flow_steps,
            "flow_start_time": float(self.apxinf_cfg.get("flow_start_time", 1.0)),
            "sampling_seed": self.seed,
        }
        calibration = self.apxinf_cfg.get("calibration", None)
        if calibration:
            kwargs["calibration"] = str(calibration)
        if tactics:
            kwargs["tactics"] = str(tactics)
        if self.apxinf_cfg.get("num_views", None) is not None:
            kwargs["num_views"] = int(self.apxinf_cfg.get("num_views"))
        return apxinf_py.Model.load("pi05", str(checkpoint_path), **kwargs)

    def _validate_model_contract(self) -> None:
        if int(self.model.action_horizon) != self.action_horizon:
            raise ValueError(
                f"ApxInf loaded action_horizon={self.model.action_horizon}, "
                f"expected {self.action_horizon}"
            )
        if int(self.model.action_dim) <= 0:
            raise ValueError("ApxInf loaded an invalid model action dimension")
        if int(self.model.action_dim) < self.action_dim:
            raise ValueError(
                "ApxInf model action dimension is smaller than the configured "
                "physical action width"
            )
        if int(self.model.num_views) <= 0:
            raise ValueError("ApxInf loaded an invalid camera-view count")

    def _sample_noise(self, batch_size: int) -> np.ndarray | None:
        if self.noise_source != "torch":
            return None
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=(batch_size, self.action_horizon, int(self.model.action_dim)),
            generator=self._noise_generator,
            device=self.device,
            dtype=torch.float32,
        )
        return noise.cpu().numpy()

    def predict_action_batch(
        self,
        env_obs: Mapping[str, Any],
        *,
        mode: str = "eval",
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run RLinf transforms -> ApxInf L1 -> RLinf output transforms."""
        if mode not in {"train", "eval"}:
            raise ValueError(f"Unsupported rollout mode {mode!r}")

        prepared = self.processor.preprocess_batch(
            env_obs,
            num_views=int(self.model.num_views),
            image_size=int(self.model.image_size),
        )
        batch_size = len(prepared)
        explicit_noise = None
        if self.noise_source == "observation":
            explicit_noise = env_obs.get("noise")
            if explicit_noise is None:
                raise ValueError("noise_source=observation requires env_obs['noise']")
        sampled_noise = self._sample_noise(batch_size)

        normalized_rows = []
        prefix_rows = []
        timings = []
        for index, model_input in enumerate(prepared):
            noise = None
            if explicit_noise is not None:
                noise = _item(explicit_noise, index)
            elif sampled_noise is not None:
                noise = sampled_noise[index]

            started = time.perf_counter()
            if self.return_prefix_hidden_states:
                inference = (
                    getattr(self.model, "infer_rgb_with_prefix_cuda", None)
                    if str(self.apxinf_cfg.get("precision", "bf16")) == "bf16"
                    else None
                )
                cuda_prefix = callable(inference)
                if not cuda_prefix:
                    inference = getattr(self.model, "infer_rgb_with_prefix", None)
                if not callable(inference):
                    raise RuntimeError(
                        "ApxInf prefix output was requested, but the loaded binding "
                        "does not expose a prefix inference API."
                    )
            else:
                inference = self.model.infer_rgb
            raw_output = inference(
                model_input["rgb_u8"],
                "nhwc",
                model_input["token_ids"],
                noise=noise,
            )
            if self.return_prefix_hidden_states:
                if cuda_prefix:
                    raw_output, prefix_view, scale_view, eps = raw_output
                    prefix = _normalize_cuda_prefix(prefix_view, scale_view, eps)
                else:
                    raw_output, prefix = raw_output
                    prefix = np.asarray(prefix, dtype=np.float32)
                    if prefix.ndim != 2 or not np.isfinite(prefix).all():
                        raise ValueError(
                            "ApxInf prefix hidden states must be finite rank-2 arrays; "
                            f"got {prefix.shape}."
                        )
                prefix_rows.append(prefix)
            normalized = np.asarray(raw_output, dtype=np.float32)
            model_ms = (time.perf_counter() - started) * 1000.0
            expected_shape = (self.action_horizon, int(self.model.action_dim))
            if normalized.shape != expected_shape:
                raise ValueError(
                    f"ApxInf normalized actions have shape {normalized.shape}, "
                    f"expected {expected_shape}"
                )
            if not np.isfinite(normalized).all():
                raise FloatingPointError("ApxInf returned non-finite actions")
            normalized_rows.append(normalized)
            timings.append({"model_ms": model_ms})

        actions = self.processor.postprocess_batch(
            np.stack(normalized_rows),
            prepared,
        )
        result: dict[str, Any] = {
            "apxinf_timing": timings,
            "normalized_actions": torch.from_numpy(np.stack(normalized_rows)),
            "prepared": prepared,
        }
        if prefix_rows:
            lengths = [row.shape[0] for row in prefix_rows]
            first = prefix_rows[0]
            hidden = torch.zeros(
                len(prefix_rows),
                max(lengths),
                first.shape[1],
                dtype=torch.float32,
                device=first.device if torch.is_tensor(first) else "cpu",
            )
            mask = torch.zeros(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
            for index, row in enumerate(prefix_rows):
                hidden[index, : len(row)] = torch.as_tensor(row, device=hidden.device)
                mask[index, : len(row)] = True
            result["prefix_hidden_states"] = hidden
            result["prefix_mask"] = mask
        return actions, result

    def close(self) -> None:
        """Release the native model on its inference-owner thread once."""
        if self._closed:
            return
        self._closed = True
        model, self.model = self.model, None
        close = getattr(model, "close", None)
        if callable(close):
            close()


class OnlineResidualApxInfAdapter:
    """Keep PI0.5 in ApxInf while running RLT and residual Actor in PyTorch."""

    def predict_base_action_batch(
        self,
        env_obs: Mapping[str, Any],
        *,
        horizon: int,
        committed: torch.Tensor | None = None,
        rtc_delay_steps: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Sample the full base horizon before the fine phase begins."""
        if committed is not None:
            raise ValueError("APX base inference does not use committed actions.")
        physical, metadata = self.base.predict_action_batch(env_obs, mode="eval")
        if physical.shape[1] < horizon:
            raise ValueError("APX prediction is shorter than the base chunk.")
        return physical, metadata

    def __init__(
        self,
        model_cfg: DictConfig | Mapping[str, Any],
        device: str,
        *,
        base: OpenPIApxInfAdapter | None = None,
    ) -> None:
        from rlinf.algorithms.online_residual_rl import (
            OnlineResidualController,
            OnlineResidualRLConfig,
            ResidualActor,
        )
        from rlinf.models.embodiment.modules.rlt_token_transformer import (
            RLTTokenEncoder,
        )

        self.model_cfg = model_cfg
        online_cfg = model_cfg.get("online_residual", {})
        contract = OnlineResidualRLConfig(
            **{
                field.name: online_cfg[field.name]
                for field in dataclasses.fields(OnlineResidualRLConfig)
                if field.name in online_cfg
            }
        )
        if int(model_cfg.get("action_dim", contract.action_dim)) != contract.action_dim:
            raise ValueError("ApxInf and online residual action dimensions disagree.")
        base_horizon = int(model_cfg.get("num_action_chunks", contract.actor_horizon))
        if base_horizon < contract.actor_horizon:
            raise ValueError(
                "ApxInf base must produce at least the residual reference horizon; "
                f"got {base_horizon} < {contract.actor_horizon}."
            )
        self.config = contract
        self.device = torch.device(device)
        self.base = base or OpenPIApxInfAdapter(model_cfg, device)
        if not self.base.return_prefix_hidden_states:
            raise ValueError(
                "online residual ApxInf requires apxinf.return_prefix_hidden_states=true"
            )
        rlt_cfg = model_cfg.get("rlt", {})
        if rlt_cfg.get("image_prefix_tokens") is not None:
            raise ValueError("Online residual RLT requires the full multimodal prefix.")
        self.rlt_module = RLTTokenEncoder(
            input_dim=int(rlt_cfg.get("input_dim", contract.z_dim)),
            embed_dim=int(rlt_cfg.get("embed_dim", contract.z_dim)),
            prefix_seq_len=int(rlt_cfg.get("prefix_seq_len", 1024)),
            num_layers=int(rlt_cfg.get("num_layers", 2)),
            num_heads=int(rlt_cfg.get("num_heads", 8)),
            mlp_ratio=float(rlt_cfg.get("mlp_ratio", 4.0)),
        ).to(self.device)
        self.actor = ResidualActor(contract).to(self.device)
        self.controller = OnlineResidualController(
            contract,
            rlt_encoder=self.rlt_module,
            actor=self.actor,
        )
        self._load_sidecar(
            rlt_cfg.get("checkpoint"),
            model_cfg.get("actor_checkpoint", None),
        )
        self.rlt_module.eval()
        self.rlt_module.requires_grad_(False)
        self.actor.eval()
        self.conditions_on_committed = False
        self.actor_version = 0
        self.metadata = {**self.base.metadata, "sidecar": "online_residual"}

    def _pad_rlt_prefix(
        self, prefix: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Restore Stage1's masked camera slots before RLT positional encoding."""
        from openpi.models.model import IMAGE_KEYS

        model = self.base.model
        image_slots = len(IMAGE_KEYS)
        views = int(model.num_views)
        patch_size = int(model.patch_size)
        image_size = int(model.image_size)
        if patch_size <= 0 or image_size % patch_size or views > image_slots:
            raise ValueError("APX image geometry disagrees with Stage1 image slots.")
        per_view = (image_size // patch_size) ** 2
        image_tokens = views * per_view
        if prefix.ndim != 3 or mask.shape != prefix.shape[:2]:
            raise ValueError("APX prefix and mask must have matching batch/length.")
        if prefix.shape[1] < image_tokens:
            raise ValueError("APX prefix is shorter than its image patch tokens.")
        missing = (image_slots - views) * per_view
        if not missing:
            return prefix, mask
        padding = prefix.new_zeros(prefix.shape[0], missing, prefix.shape[2])
        invalid = mask.new_zeros(mask.shape[0], missing)
        return (
            torch.cat(
                (prefix[:, :image_tokens], padding, prefix[:, image_tokens:]), dim=1
            ),
            torch.cat((mask[:, :image_tokens], invalid, mask[:, image_tokens:]), dim=1),
        )

    @staticmethod
    def _unwrap_state_dict(value: Any) -> Mapping[str, torch.Tensor]:
        if not isinstance(value, Mapping):
            raise ValueError("Sidecar checkpoint must contain a state-dict mapping.")
        for key in ("state_dict", "model", "module"):
            nested = value.get(key)
            if isinstance(nested, Mapping):
                return nested
        return value

    @staticmethod
    def _select_prefixed(
        state: Mapping[str, torch.Tensor], prefixes: tuple[str, ...]
    ) -> dict[str, torch.Tensor]:
        for prefix in prefixes:
            selected = {}
            for key, value in state.items():
                marker = f"{prefix}."
                position = key.find(marker)
                if position >= 0:
                    selected[key[position + len(marker) :]] = value
            if selected:
                return selected
        return {}

    def _load_sidecar(self, rlt_checkpoint: Any, actor_checkpoint: Any) -> None:
        if not rlt_checkpoint:
            raise ValueError(
                "rlt.checkpoint is required for APX online residual inference."
            )
        from rlinf.models.embodiment.openpi_rlinf.checkpoint import resolve_full_weights

        rlt_path = pathlib.Path(str(rlt_checkpoint)).expanduser()
        if rlt_path.is_dir():
            resolved = resolve_full_weights(rlt_path)
            if resolved is None:
                raise FileNotFoundError(
                    f"Stage1 full_weights.pt was not found under {rlt_path}."
                )
            rlt_path = resolved
        rlt_state = self._unwrap_state_dict(
            torch.load(str(rlt_path), map_location="cpu", weights_only=False)
        )
        selected_rlt = self._select_prefixed(
            rlt_state, ("rlt_module.encoder", "encoder")
        )
        self.rlt_module.load_state_dict(selected_rlt or rlt_state, strict=True)
        if actor_checkpoint:
            from rlinf.algorithms.online_residual_rl.checkpoint import load_actor_state

            self.actor.load_state_dict(load_actor_state(actor_checkpoint), strict=True)

    @torch.inference_mode()
    def predict_action_batch(
        self,
        env_obs: Mapping[str, Any],
        *,
        mode: str = "eval",
        committed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return the executable prefix of the full VLA action chunk.

        The APX VLA computes the configured full horizon (32 for Flexiv). The
        residual controller consumes the first 12 actions; the remaining 20
        are intentionally dropped at this boundary.

        Optional normalized committed actions construct Critic metadata only.
        They never condition the VLA or the Actor.
        """
        # Keep APX Stage1's established deterministic sampler. Exploration is
        # applied only by the PyTorch residual Actor below.
        _, base_result = self.base.predict_action_batch(env_obs, mode="eval")
        base_reference = base_result["normalized_actions"][
            ..., : self.config.actor_horizon, : self.config.action_dim
        ].to(self.device)
        prefix, prefix_mask = self._pad_rlt_prefix(
            base_result["prefix_hidden_states"], base_result["prefix_mask"]
        )
        prefix = prefix.to(self.device)
        state = env_obs["states"]
        if self.config.proprio_normalization == "stage1":
            state = np.stack([item["state"] for item in base_result["prepared"]])
            state = state[..., : self.config.proprio_dim]
        proprio = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        if (
            proprio.ndim != 2
            or proprio.shape
            != (
                base_reference.shape[0],
                self.config.proprio_dim,
            )
            or not torch.isfinite(proprio).all()
        ):
            raise ValueError(
                "APX residual proprio must be finite [batch, proprio_dim]."
            )
        decision = self.controller.decide(
            prefix_hidden_states=prefix,
            proprio=proprio,
            base_reference=base_reference,
            deterministic=(mode != "rollout"),
            committed=committed,
            prefix_mask=prefix_mask.to(self.device),
            use_actor=(mode != "base"),
        )
        prepared = base_result["prepared"]
        # OpenPI output transforms expect the full model action width, even
        # though the residual Actor changes only the executable action dims.
        model_actions = torch.zeros_like(
            base_result["normalized_actions"][:, : self.config.actor_horizon]
        )
        model_actions[..., : self.config.action_dim] = decision.actor_output
        actions = self.base.processor.postprocess_batch(
            model_actions.detach().cpu().numpy(),
            prepared,
            chunk_len=self.config.actor_horizon,
        )
        forward_inputs = {
            "z_rl": decision.z,
            "proprio": decision.proprio,
            "base_ref": decision.base_reference,
            "committed": decision.committed,
            "actor_output": decision.actor_output,
            "policy_proposal": decision.proposal,
        }
        return actions, {
            "normalized_actions": decision.actor_output,
            "prev_logprobs": None,
            "prev_values": None,
            "forward_inputs": forward_inputs,
            "apxinf_timing": base_result["apxinf_timing"],
        }

    def encode_receipts(self, physical_actions: torch.Tensor) -> torch.Tensor:
        """Normalize final robot/human receipts for replay, not the future queue."""
        stats = getattr(self.base.processor, "normalize_action_batch", None)
        if not callable(stats):
            raise RuntimeError(
                "RLinf OpenPI transforms must expose normalize_action_batch() "
                "to commit physical execution receipts."
            )
        normalized = stats(physical_actions)
        return normalized.detach().cpu()

    @torch.inference_mode()
    def install_actor(self, state: dict[str, torch.Tensor], version: int) -> None:
        """Install a complete Actor snapshot at an inference request boundary."""
        if version <= self.actor_version:
            raise ValueError("Published Actor version must increase.")
        self.actor.load_state_dict(state, strict=True)
        self.actor.eval()
        self.actor_version = version

    def reset(self) -> None:
        """Clear committed actions at an episode boundary."""
        self.controller.reset()

    def close(self) -> None:
        self.base.close()
