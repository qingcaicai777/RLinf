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

"""OpenPI RLinf SFT loader for compact Flexiv demonstrations."""

from __future__ import annotations

import dataclasses
import multiprocessing
from collections.abc import Iterator
from typing import Any, cast

import torch
from openpi.transforms import compose
from torch.utils.data.distributed import DistributedSampler

from rlinf.data.datasets.openpi_rlinf.collate import collate_openpi_sft_items
from rlinf.data.datasets.openpi_rlinf.flexiv.flexiv_sft_dataset import (
    FlexivSftDataset,
)
from rlinf.data.storage.lerobot import resolve_lerobot_repo_id
from rlinf.models.embodiment.openpi_rlinf.transforms.pipeline import (
    build_openpi_transforms,
)
from rlinf.utils.logging import get_logger

logger = get_logger()


class _TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: FlexivSftDataset, transform: Any) -> None:
        self._dataset = dataset
        self._transform = transform

    def __getitem__(self, index: int) -> Any:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


@dataclasses.dataclass(frozen=True)
class FlexivSftDataConfig:
    """Resolved metadata exposed to the SFT worker."""

    repo_id: str
    action_dim: int
    action_horizon: int
    max_token_len: int


class FlexivSftDataLoader:
    """Infinite, distributed loop over transformed Flexiv SFT batches."""

    def __init__(
        self,
        torch_loader: torch.utils.data.DataLoader,
        data_config: FlexivSftDataConfig,
        sampler: DistributedSampler,
    ) -> None:
        self._torch_loader = torch_loader
        self._data_config = data_config
        self._sampler = sampler
        self._epoch = 0

    def data_config(self) -> FlexivSftDataConfig:
        """Return the resolved Flexiv data configuration."""
        return self._data_config

    def __iter__(self) -> Iterator[tuple[Any, torch.Tensor]]:
        while True:
            self._sampler.set_epoch(self._epoch)
            self._epoch += 1
            yield from self._torch_loader

    def __len__(self) -> int:
        return len(self._torch_loader)


def create_flexiv_sft_data_loader(
    *,
    data_path: str,
    model_path: str,
    config_name: str,
    action_dim: int,
    action_horizon: int,
    max_token_len: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    dist_rank: int,
    dist_world_size: int,
    data_kwargs: dict[str, Any] | None = None,
    cache_dir: str | None = None,
) -> FlexivSftDataLoader:
    """Create the Flexiv loader and apply OpenPI transforms after delta conversion."""
    dataset = FlexivSftDataset(
        data_path=data_path,
        action_horizon=action_horizon,
        cache_dir=cache_dir,
    )
    input_transforms, _ = build_openpi_transforms(
        model_path,
        config_name,
        data_kwargs=data_kwargs,
    )
    source = _TransformedDataset(dataset, compose(input_transforms))
    sampler = DistributedSampler(
        source,
        num_replicas=dist_world_size,
        rank=dist_rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=True,
    )
    multiprocessing_context = (
        multiprocessing.get_context("spawn") if num_workers > 0 else None
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    torch_loader = torch.utils.data.DataLoader(
        cast(torch.utils.data.Dataset, source),
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        multiprocessing_context=multiprocessing_context,
        persistent_workers=num_workers > 0,
        collate_fn=collate_openpi_sft_items,
        drop_last=True,
        generator=generator,
        pin_memory=True,
    )
    logger.info(
        "Flexiv SFT loader: root=%s, frames=%d, batch_size=%d, horizon=%d",
        dataset.dataset_root,
        len(dataset),
        batch_size,
        action_horizon,
    )
    data_config = FlexivSftDataConfig(
        repo_id=str(dataset.dataset_root),
        action_dim=action_dim,
        action_horizon=action_horizon,
        max_token_len=max_token_len,
    )
    return FlexivSftDataLoader(torch_loader, data_config, sampler)


def build_flexiv_sft_dataloader(
    cfg: Any,
    world_size: int,
    rank: int,
    data_paths: Any,
    eval_dataset: bool = False,
) -> tuple[FlexivSftDataLoader, FlexivSftDataConfig]:
    """Build the Flexiv loader selected by ``pi05_flexiv_*`` configs."""
    from omegaconf import OmegaConf

    data_path = resolve_lerobot_repo_id(data_paths)
    if data_path is None:
        raise ValueError("Flexiv SFT requires data.train_data_paths.")
    model_cfg = cfg.actor.model
    openpi_cfg = model_cfg.openpi
    data_kwargs = OmegaConf.select(model_cfg, "openpi_data", default=None)
    if data_kwargs is not None:
        data_kwargs = cast(
            dict[str, Any], OmegaConf.to_container(data_kwargs, resolve=True)
        )
    loader = create_flexiv_sft_data_loader(
        data_path=str(data_path),
        model_path=str(model_cfg.model_path),
        config_name=str(openpi_cfg.config_name),
        action_dim=int(openpi_cfg.model_action_dim),
        action_horizon=int(model_cfg.num_action_chunks),
        max_token_len=int(openpi_cfg.max_token_len),
        batch_size=(
            int(cfg.actor.eval_batch_size)
            if eval_dataset
            else int(cfg.actor.micro_batch_size)
        ),
        num_workers=int(OmegaConf.select(cfg, "data.num_workers", default=4)),
        shuffle=not eval_dataset,
        seed=int(cfg.actor.seed),
        dist_rank=rank,
        dist_world_size=world_size,
        data_kwargs=data_kwargs,
        cache_dir=OmegaConf.select(cfg, "data.cache_dir", default=None),
    )
    return loader, loader.data_config()
