# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""CUDA graph for the fixed-shape PI0.5 action expert during inference."""

from __future__ import annotations

from typing import Any

import torch


class CudaGraphedSuffix:
    """Replay one action-expert step with current actions and prefix KV data."""

    def __init__(self, model: Any) -> None:
        if not model.pi05 or model.device.type != "cuda":
            raise ValueError("CUDA graph suffix requires a CUDA PI0.5 model.")
        self.model = model
        self.graph: torch.cuda.CUDAGraph | None = None
        self.signature: tuple[Any, ...] | None = None
        self.velocity: torch.Tensor | None = None

    def __call__(
        self,
        observation: Any,
        actions: torch.Tensor,
        timestep: torch.Tensor,
        kv_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        prefix_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the velocity for one flow step using current tensor values."""
        if torch.is_grad_enabled():
            raise RuntimeError("CUDA graph suffix is available only without gradients.")
        signature = (
            (actions.shape, actions.dtype, actions.device),
            (timestep.shape, timestep.dtype, timestep.device),
            (prefix_mask.shape, prefix_mask.dtype, prefix_mask.device),
            tuple(
                (k.shape, k.dtype, k.device, v.shape, v.dtype, v.device)
                for k, v in kv_cache
            ),
        )
        if signature != self.signature:
            self._capture(observation, actions, timestep, kv_cache, prefix_mask)
            self.signature = signature

        self.actions.copy_(actions)
        self.timestep.copy_(timestep)
        self.prefix_mask.copy_(prefix_mask)
        for (static_k, static_v), (key, value) in zip(
            self.kv_cache, kv_cache, strict=True
        ):
            static_k.copy_(key)
            static_v.copy_(value)
        assert self.graph is not None and self.velocity is not None
        self.graph.replay()
        return self.velocity

    def _capture(
        self,
        observation: Any,
        actions: torch.Tensor,
        timestep: torch.Tensor,
        kv_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        prefix_mask: torch.Tensor,
    ) -> None:
        """Capture a graph for the current prefix and action shapes."""
        self.graph = None
        self.velocity = None
        self.actions = actions.clone()
        self.timestep = timestep.clone()
        self.prefix_mask = prefix_mask.clone()
        self.kv_cache = tuple((k.clone(), v.clone()) for k, v in kv_cache)

        def velocity() -> torch.Tensor:
            # PI0.5 suffix embeddings depend on actions and timestep; the
            # observation is represented by the prefix KV cache copied below.
            return self.model.velocity_from_suffix(
                self.model.run_suffix(
                    observation,
                    self.actions,
                    self.timestep,
                    self.kv_cache,
                    self.prefix_mask,
                )
            )

        stream = torch.cuda.Stream(device=actions.device)
        stream.wait_stream(torch.cuda.current_stream(actions.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                velocity()
        torch.cuda.current_stream(actions.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_velocity = velocity()
        self.graph = graph
        self.velocity = static_velocity
