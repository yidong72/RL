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
"""GRPO Algorithm Package.

This package provides the Group Relative Policy Optimization (GRPO) algorithm
split into focused modules:

- config.py: Configuration classes (GRPOConfig, MasterConfig, etc.)
- loss.py: Loss functions and advantage computation
- data.py: Data transforms and batch processing
- trainer.py: GRPOTrainer extending BaseTrainer
- utils.py: Utility functions

For backward compatibility, all public symbols from the original grpo.py
are re-exported from this package.

Example (new API):
    >>> from nemo_rl.algorithms.grpo import GRPOTrainer, GRPOConfig
    >>> 
    >>> trainer = GRPOTrainer(config)
    >>> trainer.fit(dataset="nvidia/OpenMathInstruct-2")

Example (backward compatible):
    >>> from nemo_rl.algorithms.grpo import (
    ...     MasterConfig, grpo_train, setup
    ... )
    >>> 
    >>> # Legacy API still works
    >>> grpo_train(...)
"""

# Import only the trainer directly - it has minimal dependencies
from nemo_rl.algorithms.grpo.trainer import GRPOTrainer


def __getattr__(name: str):
    """Lazy import of GRPO components to avoid heavy import chain.
    
    This allows importing just GRPOTrainer without triggering the full
    import chain including mlflow, wandb, etc.
    """
    # Config classes
    if name in (
        "AsyncGRPOConfig", "GRPOConfig", "GRPOLoggerConfig", 
        "GRPOSaveState", "MasterConfig", "RewardScalingConfig",
        "default_grpo_save_state",
    ):
        from nemo_rl.algorithms.grpo.config import (
            AsyncGRPOConfig, GRPOConfig, GRPOLoggerConfig,
            GRPOSaveState, MasterConfig, RewardScalingConfig,
            default_grpo_save_state,
        )
        return {
            "AsyncGRPOConfig": AsyncGRPOConfig,
            "GRPOConfig": GRPOConfig,
            "GRPOLoggerConfig": GRPOLoggerConfig,
            "GRPOSaveState": GRPOSaveState,
            "MasterConfig": MasterConfig,
            "RewardScalingConfig": RewardScalingConfig,
            "default_grpo_save_state": default_grpo_save_state,
        }[name]
    
    # Data functions
    if name in (
        "check_batch_has_variance", "dynamic_sample_batch",
        "filter_overlong_sequences", "prepare_batch_for_training",
    ):
        from nemo_rl.algorithms.grpo.data import (
            check_batch_has_variance, dynamic_sample_batch,
            filter_overlong_sequences, prepare_batch_for_training,
        )
        return {
            "check_batch_has_variance": check_batch_has_variance,
            "dynamic_sample_batch": dynamic_sample_batch,
            "filter_overlong_sequences": filter_overlong_sequences,
            "prepare_batch_for_training": prepare_batch_for_training,
        }[name]
    
    # Loss functions
    if name in (
        "compute_grpo_loss", "create_loss_function",
        "normalize_advantages_in_batch", "scale_rewards",
    ):
        from nemo_rl.algorithms.grpo.loss import (
            compute_grpo_loss, create_loss_function,
            normalize_advantages_with_epsilon as normalize_advantages_in_batch,
            scale_rewards,
        )
        return {
            "compute_grpo_loss": compute_grpo_loss,
            "create_loss_function": create_loss_function,
            "normalize_advantages_in_batch": normalize_advantages_in_batch,
            "scale_rewards": scale_rewards,
        }[name]
    
    # Utility functions
    if name in (
        "compute_effective_batch_size", "get_generation_backend",
        "is_colocation_enabled", "log_training_metrics",
        "should_use_async_rollouts", "should_use_nemo_gym",
    ):
        from nemo_rl.algorithms.grpo.utils import (
            compute_effective_batch_size, get_generation_backend,
            is_colocation_enabled, log_training_metrics,
            should_use_async_rollouts, should_use_nemo_gym,
        )
        return {
            "compute_effective_batch_size": compute_effective_batch_size,
            "get_generation_backend": get_generation_backend,
            "is_colocation_enabled": is_colocation_enabled,
            "log_training_metrics": log_training_metrics,
            "should_use_async_rollouts": should_use_async_rollouts,
            "should_use_nemo_gym": should_use_nemo_gym,
        }[name]
    
    # Legacy API (backward compatibility)
    if name in (
        "_default_grpo_save_state", "_should_use_async_rollouts",
        "async_grpo_train", "dynamic_sampling", "grpo_train",
        "normalize_advantages_with_epsilon", "refit_policy_generation",
        "setup", "validate",
    ):
        from nemo_rl.algorithms.grpo_legacy import (
            _default_grpo_save_state, _should_use_async_rollouts,
            async_grpo_train, dynamic_sampling, grpo_train,
            normalize_advantages_with_epsilon, refit_policy_generation,
            setup, validate,
        )
        return {
            "_default_grpo_save_state": _default_grpo_save_state,
            "_should_use_async_rollouts": _should_use_async_rollouts,
            "async_grpo_train": async_grpo_train,
            "dynamic_sampling": dynamic_sampling,
            "grpo_train": grpo_train,
            "normalize_advantages_with_epsilon": normalize_advantages_with_epsilon,
            "refit_policy_generation": refit_policy_generation,
            "setup": setup,
            "validate": validate,
        }[name]
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # New modular API
    "GRPOTrainer",
    # Config classes
    "GRPOConfig",
    "MasterConfig",
    "AsyncGRPOConfig",
    "RewardScalingConfig",
    "GRPOSaveState",
    "GRPOLoggerConfig",
    "default_grpo_save_state",
    # Loss functions
    "normalize_advantages_with_epsilon",
    "normalize_advantages_in_batch",  # New batch-based API
    "scale_rewards",
    "compute_grpo_loss",
    "create_loss_function",
    # Data functions
    "filter_overlong_sequences",
    "check_batch_has_variance",
    "dynamic_sample_batch",
    "prepare_batch_for_training",
    # Utility functions
    "should_use_async_rollouts",
    "should_use_nemo_gym",
    "get_generation_backend",
    "is_colocation_enabled",
    "compute_effective_batch_size",
    "log_training_metrics",
    # Legacy API (backward compatibility)
    "setup",
    "grpo_train",
    "async_grpo_train",
    "validate",
    "_default_grpo_save_state",
    "_should_use_async_rollouts",
    "refit_policy_generation",
    "dynamic_sampling",
]
