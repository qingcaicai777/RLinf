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

"""Flexiv compact-v3 SFT dataset support."""

from .flexiv_sft_dataset import (
    FlexivSftDataset,
    absolute_tcp_chunk_to_control_delta,
    absolute_tcp_chunk_to_world_delta,
)


def build_flexiv_sft_dataloader(*args, **kwargs):
    """Lazily build the OpenPI-dependent Flexiv loader."""
    from .flexiv_sft_data_loader import build_flexiv_sft_dataloader as builder

    return builder(*args, **kwargs)

__all__ = [
    "FlexivSftDataset",
    "absolute_tcp_chunk_to_control_delta",
    "absolute_tcp_chunk_to_world_delta",
    "build_flexiv_sft_dataloader",
]
