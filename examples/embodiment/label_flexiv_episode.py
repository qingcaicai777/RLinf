#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""List pending Flexiv episodes and confirm or discard their terminal labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rlinf.algorithms.online_residual_rl.config import OnlineResidualRLConfig
from rlinf.algorithms.online_residual_rl.storage import PendingEpisodeStore


def main() -> int:
    """Publish labelled episodes for the running or next resumed learner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--episode-id")
    parser.add_argument("--label", choices=("success", "failure", "discard"))
    args = parser.parse_args()
    if bool(args.episode_id) != bool(args.label):
        parser.error("--episode-id and --label must be provided together")
    for path in sorted((args.episode_dir / "pending").glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        episode_id = payload["episode_id"]
        if args.episode_id is None:
            print(
                f"{episode_id}\tframes={payload['terminal_frame'] + 1}\twindows={len(payload['transitions'])}"
            )
        elif episode_id == args.episode_id:
            config = OnlineResidualRLConfig(**payload["config"])
            store = PendingEpisodeStore(args.episode_dir, config)
            if args.label == "discard":
                store.discard(episode_id)
                print(f"Discarded {episode_id}")
            else:
                print(store.label(episode_id, args.label == "success"))
            return 0
    if args.episode_id is not None:
        parser.error("Pending episode not found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
