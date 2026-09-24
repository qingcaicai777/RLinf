# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Online residual policy backed by frozen RLT observation features."""

from omegaconf import DictConfig

from .policy import OnlineResidualPolicy


def get_model(cfg: DictConfig, torch_dtype=None) -> OnlineResidualPolicy:
    """Build an online residual policy from an RLinf model config."""
    del torch_dtype
    return OnlineResidualPolicy.from_config(cfg)


__all__ = ["OnlineResidualPolicy", "get_model"]
