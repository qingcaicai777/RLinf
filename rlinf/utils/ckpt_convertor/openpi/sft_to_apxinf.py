# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Export the Stage1 PI0.5 base weights for APX's bare-model loader."""

from __future__ import annotations

import pathlib

import torch

from rlinf.utils.ckpt_convertor.openpi._core import (
    as_state_dict,
    copy_norm_stats,
    save_safetensors,
    strip_wrapper_prefix,
    write_config_json,
)
from rlinf.utils.ckpt_convertor.openpi.openpi_rlinf_to_openpi_pytorch import (
    new_to_old_state_dict,
)
from rlinf.utils.ckpt_convertor.openpi.pt_to_safetensors import (
    _resolve_full_weights,
    _validate_state_dict,
    resolve_model_spec,
)


def prepare_apx_weights(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map Stage1 base tensors to APX keys without its unused expert LM head."""
    bare = strip_wrapper_prefix(state_dict, cast_dtype=None)
    converted = new_to_old_state_dict(bare, action_expert_uses_adarms=True)
    required = {
        "paligemma_with_expert.paligemma.lm_head.weight",
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight",
        "action_in_proj.weight",
        "action_out_proj.weight",
        "time_mlp_in.weight",
        "time_mlp_out.weight",
    }
    missing = sorted(required - converted.keys())
    if missing:
        raise ValueError(f"Stage1 checkpoint is missing APX PI0.5 weights: {missing}")
    return {
        key: tensor.to(dtype=torch.bfloat16).contiguous()
        if tensor.is_floating_point()
        else tensor.contiguous()
        for key, tensor in converted.items()
    }


def convert(
    ckpt: str | pathlib.Path,
    input_norm_stats: str | pathlib.Path,
    output_model: str | pathlib.Path,
    *,
    config_name: str,
    num_views: int,
) -> pathlib.Path:
    """Write a native APX checkpoint while retaining Stage1's normalization."""
    if num_views <= 0:
        raise ValueError("num_views must be positive.")
    spec = resolve_model_spec(config_name)
    if not spec.pi05:
        raise ValueError("APX Stage1 export requires a PI0.5 config.")
    output_dir = pathlib.Path(output_model)
    weights_output = output_dir / "model.safetensors"
    if weights_output.exists():
        raise FileExistsError(f"APX export already exists: {weights_output}")
    weights_path = _resolve_full_weights(ckpt)
    loaded = torch.load(weights_path, map_location="cpu", weights_only=False, mmap=True)
    state_dict = as_state_dict(loaded)
    bare = strip_wrapper_prefix(state_dict, cast_dtype=None)
    _validate_state_dict(bare, spec)
    converted = prepare_apx_weights(state_dict)
    save_safetensors(converted, weights_output)
    config = dict(spec.config)
    config.update(num_views=num_views, dtype="bfloat16")
    write_config_json(config, output_dir)
    copy_norm_stats(input_norm_stats, output_dir / "norm_stats.json")
    return weights_output


def add_arguments(parser) -> None:
    """Register APX export arguments on the shared checkpoint CLI."""
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--input-norm-stats", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--num-views", type=int, required=True)


def run(args) -> None:
    """Export the requested checkpoint."""
    print(
        convert(
            args.ckpt,
            args.input_norm_stats,
            args.output_model,
            config_name=args.config_name,
            num_views=args.num_views,
        )
    )
