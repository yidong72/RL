# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Logger configuration types.

This module contains TypedDict definitions for logger configuration.
It's intentionally kept separate from logger.py to avoid heavy imports
(mlflow, wandb, etc.) when only the type definitions are needed.
"""

from typing import NotRequired, TypedDict


class WandbConfig(TypedDict):
    project: NotRequired[str]
    name: NotRequired[str]


class SwanlabConfig(TypedDict):
    project: NotRequired[str]
    name: NotRequired[str]


class TensorboardConfig(TypedDict):
    log_dir: NotRequired[str]


class MLflowConfig(TypedDict):
    experiment_name: str
    run_name: str
    tracking_uri: NotRequired[str]
    artifact_location: NotRequired[str | None]


class GPUMonitoringConfig(TypedDict):
    collection_interval: int | float
    flush_interval: int | float


class LoggerConfig(TypedDict):
    log_dir: str
    wandb_enabled: bool
    swanlab_enabled: bool
    tensorboard_enabled: bool
    mlflow_enabled: bool
    wandb: WandbConfig
    tensorboard: NotRequired[TensorboardConfig]
    swanlab: NotRequired[SwanlabConfig]
    mlflow: NotRequired[MLflowConfig]
    monitor_gpus: bool
    gpu_monitoring: GPUMonitoringConfig
    num_val_samples_to_print: NotRequired[int]
