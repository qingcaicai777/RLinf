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

"""Collation shared by transformed OpenPI RLinf SFT datasets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from rlinf.models.embodiment.openpi_rlinf.modules.model import Observation

_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def collate_openpi_sft_items(
    items: Sequence[Mapping[str, Any]],
) -> tuple[Observation, torch.Tensor]:
    """Collate transformed OpenPI examples into observation and action tensors."""
    if not items:
        raise ValueError("Cannot collate an empty OpenPI SFT batch.")
    images = {
        key: torch.from_numpy(
            np.stack([np.asarray(item["image"][key]) for item in items])
        )
        for key in _IMAGE_KEYS
    }
    image_masks = {
        key: torch.from_numpy(
            np.stack(
                [np.asarray(item["image_mask"][key], dtype=np.bool_) for item in items]
            )
        )
        for key in _IMAGE_KEYS
    }
    batch = {
        "image": images,
        "image_mask": image_masks,
        "state": torch.from_numpy(
            np.stack([np.asarray(item["state"], dtype=np.float32) for item in items])
        ),
        "tokenized_prompt": torch.from_numpy(
            np.stack(
                [np.asarray(item["tokenized_prompt"], dtype=np.int64) for item in items]
            )
        ).long(),
        "tokenized_prompt_mask": torch.from_numpy(
            np.stack(
                [
                    np.asarray(item["tokenized_prompt_mask"], dtype=np.bool_)
                    for item in items
                ]
            )
        ),
    }
    actions = torch.from_numpy(
        np.stack([np.asarray(item["actions"], dtype=np.float32) for item in items])
    )
    return Observation.from_dict(batch), actions
