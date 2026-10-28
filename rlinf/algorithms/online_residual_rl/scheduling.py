# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Control-frame handoff state independent of inference completion time."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import OnlineResidualRLConfig


@dataclass(frozen=True)
class InferenceRequest:
    """A request owns its observation time, queue snapshot and Actor version."""

    sequence: int
    frame: int
    model_version: int
    committed: torch.Tensor


class HandoffSchedule:
    """Stage results early and switch only on the fixed n-frame boundary.

    The control loop owns this object. Inference runs externally and submits its
    result back to that loop. A missed deadline clears the pending request and
    requires recovery with a new accepted command queue; late results cannot
    modify that queue. No synthetic transition should be built during recovery.
    """

    def __init__(self, config: OnlineResidualRLConfig) -> None:
        self.config = config
        self._sequence = 0
        self._request: InferenceRequest | None = None
        self._ready: torch.Tensor | None = None
        self._queue: torch.Tensor | None = None
        self.recovering = True

    def recover(self, accepted_queue: torch.Tensor) -> None:
        """Install an explicitly accepted startup/recovery block."""
        self._validate(accepted_queue)
        self._queue = accepted_queue.detach().clone()
        self._request = None
        self._ready = None
        self.recovering = False

    def begin(self, frame: int, model_version: int) -> InferenceRequest:
        """Capture the n commands about to execute while inference runs."""
        if self.recovering or self._queue is None or self._request is not None:
            raise RuntimeError("Recover the queue and finish the prior handoff first.")
        request = InferenceRequest(
            self._sequence, frame, model_version, self._queue.clone()
        )
        self._sequence += 1
        self._request = request
        return request

    def submit(
        self, sequence: int, execution: torch.Tensor, *, completed_frame: float
    ) -> bool:
        """Accept only the active request completed by its original deadline."""
        request = self._request
        if request is None or sequence != request.sequence:
            return False
        if completed_frame > request.frame + self.config.inference_horizon:
            return False
        self._validate(execution)
        self._ready = execution.detach().clone()
        return True

    def handoff(self, frame: int) -> torch.Tensor | None:
        """Return the next accepted block, or signal that recovery is required."""
        if self._request is None:
            raise RuntimeError("No active inference request.")
        boundary = self._request.frame + self.config.inference_horizon
        if frame < boundary:
            raise ValueError("Early results must wait for the fixed handoff.")
        result = self._ready if frame == boundary else None
        self._request = None
        self._ready = None
        if result is None:
            self._queue = None
            self.recovering = True
            return None
        self._queue = result.clone()
        return result.clone()

    def record_execution(self, frame: int, action: torch.Tensor) -> None:
        """Store the final accepted normalized command for this queue slot."""
        if self._queue is None:
            raise RuntimeError("No active execution queue.")
        offset = frame % self.config.inference_horizon
        value = torch.as_tensor(action, dtype=self._queue.dtype)
        if value.shape != (self.config.action_dim,) or not torch.isfinite(value).all():
            raise ValueError("Executed action must be finite [action_dim].")
        self._queue[offset] = value.detach().clone()

    def _validate(self, actions: torch.Tensor) -> None:
        if actions.shape != (self.config.inference_horizon, self.config.action_dim):
            raise ValueError("Accepted queue must have shape [n, action_dim].")
        if not torch.isfinite(actions).all():
            raise ValueError("Accepted queue must be finite.")
