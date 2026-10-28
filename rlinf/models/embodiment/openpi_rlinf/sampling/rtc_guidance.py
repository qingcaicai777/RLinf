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

"""RTC overlap guidance for the vendored Pi0 eval sampler."""

from __future__ import annotations

import torch

from rlinf.models.embodiment.openpi.rtc_guidance import (
    RTCGuidanceContext,
    build_rtc_target_and_mask,
)
from rlinf.models.embodiment.openpi_rlinf.modules.model import preprocess_observation


@torch.no_grad()
def sample_actions_with_rtc_guidance(
    pi0_model,
    observation,
    rtc_context: RTCGuidanceContext | None,
    *,
    num_steps: int,
    noise: torch.Tensor | None = None,
    rng: torch.Generator | None = None,
    guidance_clip: float = 5.0,
    guidance_mode: str = "approx",
    prefix_cache: tuple | None = None,
) -> torch.Tensor:
    """Euler ODE sampling with overlap guidance against the previous chunk."""
    if guidance_mode not in {"approx", "jacobian"}:
        raise ValueError("guidance_mode must be 'approx' or 'jacobian'.")
    if prefix_cache is None:
        observation = preprocess_observation(observation, train=False)
    B = observation.state.shape[0]
    device = observation.state.device
    if noise is None:
        noise = torch.randn(
            B,
            pi0_model.action_horizon,
            pi0_model.action_dim,
            device=device,
            dtype=torch.float32,
            generator=rng,
        )
    else:
        noise = noise.to(device=device, dtype=torch.float32)

    if prefix_cache is None:
        _prefix_out, prefix_mask, kv_cache = pi0_model.build_prefix_cache(observation)
    else:
        prefix_mask, kv_cache = prefix_cache

    prev_remaining = (
        rtc_context.get_prev_remaining() if rtc_context is not None else None
    )
    target, mask = build_rtc_target_and_mask(
        prev_remaining=prev_remaining,
        horizon=pi0_model.action_horizon,
        action_dim=pi0_model.action_dim,
        delay_steps=0 if rtc_context is None else rtc_context.delay_steps,
        device=device,
        dtype=noise.dtype,
    )

    x_t = noise
    step_size = 1.0 / num_steps
    sample_velocity = getattr(pi0_model, "sample_velocity", None)

    for idx in range(num_steps):
        t_val = 1.0 - idx * step_size
        t_tensor = torch.full((B,), t_val, device=device, dtype=torch.float32)
        use_jacobian = (
            guidance_mode == "jacobian"
            and prev_remaining is not None
            and prev_remaining.numel() > 0
        )
        if use_jacobian:
            with torch.enable_grad():
                x_for_grad = x_t.detach().requires_grad_(True)
                # Match OpenPI's _apply_rtc_vjp_guidance: differentiate the
                # clean action estimate at every flow step.
                if hasattr(pi0_model, "run_suffix"):
                    run_suffix = (
                        getattr(pi0_model, "_jacobian_run_suffix", None)
                        or pi0_model.run_suffix
                    )
                    suffix_act = run_suffix(
                        observation, x_for_grad, t_tensor, kv_cache, prefix_mask
                    )
                    v_t = pi0_model.velocity_from_suffix(suffix_act)
                else:
                    v_t = sample_velocity(
                        observation, x_for_grad, t_tensor, kv_cache, prefix_mask
                    )
                clean_estimate = x_for_grad - t_tensor[:, None, None] * v_t
                jacobian_error = ((target - clean_estimate) * mask).detach()
                jacobian_correction = torch.autograd.grad(
                    clean_estimate,
                    x_for_grad,
                    grad_outputs=jacobian_error,
                    retain_graph=False,
                    create_graph=False,
                )[0]
            v_t = v_t.detach()
            jacobian_correction = jacobian_correction.detach()
        elif sample_velocity is None:
            suffix_act = pi0_model.run_suffix(
                observation, x_t, t_tensor, kv_cache, prefix_mask
            )
            v_t = pi0_model.velocity_from_suffix(suffix_act)
        else:
            v_t = sample_velocity(observation, x_t, t_tensor, kv_cache, prefix_mask)
        v_t = v_t.to(torch.float32)
        if prev_remaining is not None and prev_remaining.numel() > 0:
            if use_jacobian:
                # OpenPI uses t_clean = 1 - time and applies the VJP to the
                # velocity before the Euler update.
                t_clean = idx * step_size
                eps = 1e-6
                t_clamped = max(t_clean, eps)
                one_minus_t = max(1.0 - t_clean, eps)
                inv_r2 = (t_clamped**2 + one_minus_t**2) / (one_minus_t**2)
                c = one_minus_t / t_clamped
                guidance_weight = min(c * inv_r2, float(guidance_clip))
                guided_velocity = v_t - guidance_weight * jacobian_correction
                x_t_mean = x_t.to(torch.float32) - step_size * guided_velocity
            else:
                x_t_mean = x_t.to(torch.float32) - step_size * v_t
                guidance_scale = min(
                    step_size
                    * min(
                        (1.0 - idx * step_size) / (idx * step_size + 1e-4),
                        float(guidance_clip),
                    ),
                    1.0,
                )
                # Integrate the correction at the ODE step size and keep the
                # update between the unguided proposal and the overlap target.
                guidance_term = (target - x_t_mean) * mask
                x_t_mean = x_t_mean + guidance_scale * guidance_term
        else:
            x_t_mean = x_t.to(torch.float32) - step_size * v_t
        # The previous ODE path drew zero-weight noise at each step. Keep the
        # generator progression so seeded subsequent requests are unchanged.
        torch.randn(x_t.shape, device=device, dtype=torch.float32, generator=rng)
        x_t = x_t_mean.detach() if guidance_mode == "jacobian" else x_t_mean
    return x_t
