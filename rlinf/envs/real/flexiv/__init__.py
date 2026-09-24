# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Single-arm Flexiv tasks and Gymnasium registration."""

from rlinf.envs.real.registry import register_tasks

from .base import FlexivEnv, FlexivEnvConfig
from .intervention import FlexivInterventionDevice
from .rtc_adapter import FlexivRTCAdapter, FlexivRTCConfig

TASKS = {"FlexivOnlineResidualEnv-v1": FlexivEnv}
_ENTRY_POINTS = register_tasks(__name__, globals(), TASKS)

__all__ = [
    "TASKS",
    "FlexivEnv",
    "FlexivEnvConfig",
    "FlexivInterventionDevice",
    "FlexivRTCAdapter",
    "FlexivRTCConfig",
    *_ENTRY_POINTS,
]
