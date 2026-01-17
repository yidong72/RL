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
"""GRPO Trainer implementation.

This module provides the GRPOTrainer class that wraps the legacy grpo_train
function with a modern, user-friendly API.

Example:
    >>> from nemo_rl.algorithms.grpo import GRPOTrainer
    >>> 
    >>> # Simple usage with from_pretrained
    >>> trainer = GRPOTrainer.from_pretrained(
    ...     "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    ...     num_prompts_per_step=32,
    ... )
    >>> trainer.fit(dataset="DeepScaler")
    >>> 
    >>> # Or with full config
    >>> trainer = GRPOTrainer(config)
    >>> trainer.fit(dataset="DeepScaler")
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence, Union

if TYPE_CHECKING:
    from nemo_rl.trainers.callbacks import Callback

logger = logging.getLogger(__name__)


class GRPOTrainer:
    """Trainer for Group Relative Policy Optimization (GRPO).

    GRPOTrainer provides a user-friendly interface for GRPO training that
    wraps the proven legacy implementation. It supports both simple
    `from_pretrained()` initialization and full configuration.

    Key features:
    - Multiple generations per prompt for relative comparison
    - Advantage normalization within prompt groups
    - Leave-one-out baseline computation
    - Support for built-in datasets (DeepScaler, OpenMathInstruct-2)
    - Integration with reward environments (math, custom)

    Attributes:
        config: Full training configuration (MasterConfig).
        num_prompts_per_step: Number of prompts per training step.
        num_generations_per_prompt: Number of responses per prompt.
        normalize_rewards: Whether to normalize advantages.
        use_leave_one_out_baseline: Use leave-one-out for baseline.

    Example:
        >>> # From pretrained model (simple API)
        >>> trainer = GRPOTrainer.from_pretrained(
        ...     "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        ...     num_prompts_per_step=32,
        ...     num_generations_per_prompt=16,
        ... )
        >>> trainer.fit(dataset="DeepScaler")
        >>> 
        >>> # Or from config directly
        >>> config = load_config("examples/configs/grpo_math_1B.yaml")
        >>> trainer = GRPOTrainer(config)
        >>> trainer.fit(dataset="DeepScaler")
    """

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        num_prompts_per_step: int = 32,
        num_generations_per_prompt: int = 16,
        learning_rate: float = 5e-6,
        max_steps: int = 1000,
        max_epochs: int = 1,
        max_sequence_length: int = 2048,
        max_new_tokens: int = 1024,
        tensor_parallel_size: int = 1,
        gpus_per_node: int = 8,
        num_nodes: int = 1,
        normalize_rewards: bool = True,
        use_leave_one_out_baseline: bool = True,
        checkpoint_dir: Optional[str] = None,
        log_dir: Optional[str] = None,
        wandb_enabled: bool = False,
        tensorboard_enabled: bool = True,
        seed: int = 42,
        **kwargs: Any,
    ) -> "GRPOTrainer":
        """Create a GRPOTrainer from a pretrained model.

        This provides a simplified interface for creating a trainer with
        sensible defaults. For full control, create a config dict and
        pass it directly to GRPOTrainer().

        Args:
            model_name_or_path: HuggingFace model name or local path.
            num_prompts_per_step: Prompts per training step (batch size).
            num_generations_per_prompt: Responses to generate per prompt.
            learning_rate: Optimizer learning rate.
            max_steps: Maximum training steps.
            max_epochs: Maximum training epochs.
            max_sequence_length: Maximum total sequence length.
            max_new_tokens: Maximum new tokens to generate.
            tensor_parallel_size: Tensor parallel degree.
            gpus_per_node: GPUs per node.
            num_nodes: Number of nodes.
            normalize_rewards: Whether to normalize rewards.
            use_leave_one_out_baseline: Use leave-one-out baseline.
            checkpoint_dir: Directory for checkpoints.
            log_dir: Directory for logs.
            wandb_enabled: Enable WandB logging.
            tensorboard_enabled: Enable TensorBoard logging.
            seed: Random seed.
            **kwargs: Additional config overrides.

        Returns:
            Configured GRPOTrainer instance.
        """
        # Build model short name for directories
        model_short = model_name_or_path.split("/")[-1] if "/" in model_name_or_path else model_name_or_path
        
        if checkpoint_dir is None:
            checkpoint_dir = f"results/{model_short}"
        if log_dir is None:
            log_dir = f"logs/{model_short}"

        # Build a minimal config that matches MasterConfig structure
        config = cls._build_config(
            model_name=model_name_or_path,
            num_prompts_per_step=num_prompts_per_step,
            num_generations_per_prompt=num_generations_per_prompt,
            learning_rate=learning_rate,
            max_steps=max_steps,
            max_epochs=max_epochs,
            max_sequence_length=max_sequence_length,
            max_new_tokens=max_new_tokens,
            tensor_parallel_size=tensor_parallel_size,
            gpus_per_node=gpus_per_node,
            num_nodes=num_nodes,
            normalize_rewards=normalize_rewards,
            use_leave_one_out_baseline=use_leave_one_out_baseline,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            wandb_enabled=wandb_enabled,
            tensorboard_enabled=tensorboard_enabled,
            seed=seed,
            **kwargs,
        )

        return cls(config)

    @staticmethod
    def _build_config(
        model_name: str,
        num_prompts_per_step: int,
        num_generations_per_prompt: int,
        learning_rate: float,
        max_steps: int,
        max_epochs: int,
        max_sequence_length: int,
        max_new_tokens: int,
        tensor_parallel_size: int,
        gpus_per_node: int,
        num_nodes: int,
        normalize_rewards: bool,
        use_leave_one_out_baseline: bool,
        checkpoint_dir: str,
        log_dir: str,
        wandb_enabled: bool,
        tensorboard_enabled: bool,
        seed: int,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build a MasterConfig-compatible dict from simple parameters."""
        train_global_batch_size = num_prompts_per_step * num_generations_per_prompt

        config: Dict[str, Any] = {
            "policy": {
                "model_name": model_name,
                "tokenizer": {
                    "name": model_name,
                    "chat_template_kwargs": None,
                },
                "precision": "bfloat16",
                "dtensor_cfg": {
                    "_v2": False,
                    "enabled": True,
                    "tensor_parallel_size": tensor_parallel_size,
                    "context_parallel_size": 1,
                    "sequence_parallel": False,
                    "compile": False,
                },
                "megatron_cfg": {
                    "enabled": False,
                },
                "train_global_batch_size": train_global_batch_size,
                "train_micro_batch_size": min(2, train_global_batch_size),
                "max_total_sequence_length": max_sequence_length,
                "optimizer": {
                    "name": "torch.optim.AdamW",
                    "kwargs": {
                        "lr": learning_rate,
                        "betas": [0.9, 0.999],
                        "eps": 1e-8,
                        "weight_decay": 0.01,
                        "foreach": False,
                        "fused": False,
                    },
                },
                "scheduler": [
                    {
                        "name": "torch.optim.lr_scheduler.LinearLR",
                        "kwargs": {
                            "start_factor": 0.1,
                            "end_factor": 1.0,
                            "total_iters": 50,
                        },
                    },
                    {
                        "name": "torch.optim.lr_scheduler.ConstantLR",
                        "kwargs": {
                            "factor": 1.0,
                            "total_iters": 10000000000,
                        },
                    },
                    {"milestones": [50]},
                ],
                "sequence_packing": {
                    "enabled": True,
                    "algorithm": "modified_first_fit_decreasing",
                    "sequence_length_round": 64,
                    "train_mb_tokens": 4096,
                    "logprob_mb_tokens": 8192,
                },
                "generation": {
                    "backend": "vllm",
                    "max_new_tokens": max_new_tokens,
                    "stop_token_ids": None,  # Will be set from tokenizer
                    "stop_strings": None,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": None,
                    "colocated": {
                        "enabled": True,
                        "resources": {
                            "gpus_per_node": None,
                            "num_nodes": None,
                        },
                    },
                    "vllm_cfg": {
                        "tensor_parallel_size": tensor_parallel_size,
                        "pipeline_parallel_size": 1,
                        "expert_parallel_size": 1,
                        "max_model_len": max_sequence_length,
                        "gpu_memory_utilization": 0.6,
                        "precision": "auto",
                        "kv_cache_dtype": "auto",
                        "skip_tokenizer_init": True,
                        "async_engine": False,
                        "enforce_eager": False,
                        "enable_vllm_metrics_logger": False,
                        "vllm_metrics_logger_interval": 0.5,
                    },
                    "vllm_kwargs": {},
                },
            },
            "loss_fn": {
                "use_importance_sampling_correction": True,
                "ratio_eps": 0.2,
                "entropy_coeff": 0.0,
                "kl_coeff": 0.001,
                "force_on_policy_ratio": False,
            },
            "env": {
                "math": {
                    "num_workers": 8,
                    "math_verify_impl": "hf_math_verify",
                    "answer_extraction_model": "gpt-4o",
                    "use_async_answer_extraction": False,
                    "use_majority_vote_baseline": False,
                    "num_attempts_for_baseline": 64,
                    "check_policy_answer_in_baseline": True,
                    "timeout": 5.0,
                },
            },
            "data": {
                "dataset_name": "DeepScaler",  # Default, can be overridden
                "processor": "math_hf_data_processor",
                "env_name": "math",
                "max_input_seq_length": max_sequence_length // 2,
                "prompt_file": None,
                "system_prompt_file": None,
                "shuffle": True,
                "num_workers": 4,
            },
            "grpo": {
                "num_prompts_per_step": num_prompts_per_step,
                "num_generations_per_prompt": num_generations_per_prompt,
                "max_num_epochs": max_epochs,
                "max_num_steps": max_steps,
                "max_rollout_turns": 1,
                "normalize_rewards": normalize_rewards,
                "use_leave_one_out_baseline": use_leave_one_out_baseline,
                "val_period": 100,
                "val_batch_size": 30,
                "val_at_start": False,
                "max_val_samples": 480,
                "seed": seed,
                "overlong_filtering": True,
                "use_dynamic_sampling": False,
                "dynamic_sampling_max_gen_batches": 1,
                "batch_multiplier": 1.0,
                "reward_shaping": {"enabled": False},
                "reward_scaling": {"enabled": False},
            },
            "logger": {
                "log_dir": log_dir,
                "wandb_enabled": wandb_enabled,
                "tensorboard_enabled": tensorboard_enabled,
                "swanlab_enabled": False,
                "mlflow_enabled": False,
                "monitor_gpus": True,
                "num_val_samples_to_print": 5,
                "wandb": {
                    "project": "",
                    "name": "",
                },
                "tensorboard": {},  # TensorBoard config (empty dict uses defaults)
                "swanlab": {},  # SwanLab config
                "mlflow": {},  # MLflow config
                "gpu_monitoring": {
                    "collection_interval": 10.0,
                    "flush_interval": 10.0,
                },
            },
            "cluster": {
                "gpus_per_node": gpus_per_node,
                "num_nodes": num_nodes,
            },
            "checkpointing": {
                "enabled": kwargs.get("checkpointing_enabled", True),
                "checkpoint_dir": checkpoint_dir,
                "save_period": kwargs.get("save_period", 100),
                "checkpoint_must_save_by": kwargs.get("checkpoint_must_save_by", None),
                "metric_name": None,  # Metric to use for best checkpoint selection
                "higher_is_better": True,  # Whether higher metric values are better
                "keep_top_k": kwargs.get("keep_top_k", 3),  # Keep top K checkpoints
            },
        }

        # Apply any additional overrides from kwargs
        for key, value in kwargs.items():
            if key.startswith("grpo_"):
                config_key = key[5:]  # Remove "grpo_" prefix
                if config_key in config["grpo"]:
                    config["grpo"][config_key] = value
            elif key.startswith("policy_"):
                config_key = key[7:]  # Remove "policy_" prefix
                if config_key in config["policy"]:
                    config["policy"][config_key] = value

        return config

    def __init__(self, config: Dict[str, Any]):
        """Initialize the GRPO trainer.

        Args:
            config: Full GRPO configuration (MasterConfig-compatible dict).
        """
        self.config = config
        
        # Extract commonly accessed values
        grpo_cfg = config.get("grpo", {})
        self.num_prompts_per_step = grpo_cfg.get("num_prompts_per_step", 32)
        self.num_generations_per_prompt = grpo_cfg.get("num_generations_per_prompt", 16)
        self.normalize_rewards = grpo_cfg.get("normalize_rewards", True)
        self.use_leave_one_out_baseline = grpo_cfg.get("use_leave_one_out_baseline", True)
        
        # State tracking
        self._is_setup = False
        self._policy = None
        self._policy_generation = None
        self._tokenizer = None
        self._logger = None

    def fit(
        self,
        dataset: Optional[str] = None,
        reward_fn: Optional[Callable[[str, str], float]] = None,
        max_steps: Optional[int] = None,
        max_epochs: Optional[int] = None,
        callbacks: Optional[Sequence["Callback"]] = None,
    ) -> Dict[str, Any]:
        """Train the model using GRPO.

        This method initializes all components and runs the GRPO training loop
        using the proven legacy implementation.

        Args:
            dataset: Dataset name. Supported built-in datasets:
                - "DeepScaler": DeepScaleR math dataset
                - "OpenMathInstruct-2": OpenMathInstruct-2 dataset
                If not provided, uses config["data"]["dataset_name"].
            reward_fn: Custom reward function (not yet supported - use
                built-in math environment for now).
            max_steps: Override max training steps.
            max_epochs: Override max epochs.
            callbacks: Training callbacks (not yet supported in legacy backend).

        Returns:
            Dictionary with training metrics.

        Raises:
            ValueError: If dataset is not supported.
        """
        import os
        import ray

        from nemo_rl.algorithms.grpo_legacy import grpo_train, setup
        from nemo_rl.algorithms.utils import get_tokenizer
        from nemo_rl.data.datasets import AllTaskProcessedDataset
        from nemo_rl.data.datasets.response_datasets import load_response_dataset
        from nemo_rl.data.interfaces import TaskDataSpec
        from nemo_rl.data.processors import math_hf_data_processor
        from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
        from nemo_rl.distributed.virtual_cluster import init_ray
        from nemo_rl.environments.math_environment import MathEnvironment
        from nemo_rl.models.generation import configure_generation_config
        from nemo_rl.utils.logger import get_next_experiment_dir

        # Update config with overrides
        if dataset is not None:
            self.config["data"]["dataset_name"] = dataset
        if max_steps is not None:
            self.config["grpo"]["max_num_steps"] = max_steps
        if max_epochs is not None:
            self.config["grpo"]["max_num_epochs"] = max_epochs

        # Warn about unsupported features
        if reward_fn is not None:
            logger.warning(
                "Custom reward_fn not yet supported in GRPOTrainer. "
                "Using built-in math environment. For custom rewards, use the legacy API."
            )
        if callbacks is not None:
            logger.warning(
                "Callbacks not yet supported in GRPOTrainer legacy backend. "
                "For callbacks, use the legacy API directly."
            )

        # Set up experiment directory
        log_dir = self.config["logger"]["log_dir"]
        self.config["logger"]["log_dir"] = get_next_experiment_dir(log_dir)
        logger.info(f"Using log directory: {self.config['logger']['log_dir']}")

        if self.config["checkpointing"]["enabled"]:
            logger.info(f"Using checkpoint directory: {self.config['checkpointing']['checkpoint_dir']}")

        # Initialize Ray
        init_ray()

        # Set up tokenizer
        tokenizer = get_tokenizer(self.config["policy"]["tokenizer"])
        self._tokenizer = tokenizer

        # Configure generation
        if self.config["policy"]["generation"] is not None:
            self.config["policy"]["generation"] = configure_generation_config(
                self.config["policy"]["generation"], tokenizer
            )

        # Set up data
        print("\n▶ Setting up data...")
        data_config = self.config["data"]
        env_configs = self.config["env"]
        seed = self.config["grpo"]["seed"]

        # Load dataset
        data = load_response_dataset(data_config, seed)
        task_name = data.task_name if hasattr(data, "task_name") else data.task_spec.task_name

        # Create task spec
        math_task_spec = TaskDataSpec(
            task_name="math",
            prompt_file=data_config.get("prompt_file"),
            system_prompt_file=data_config.get("system_prompt_file"),
        )

        # Set up data processor
        task_data_processors = defaultdict(lambda: (math_task_spec, math_hf_data_processor))
        task_data_processors[task_name] = (math_task_spec, math_hf_data_processor)

        # Set up math environment
        math_env = MathEnvironment.options(
            runtime_env={
                "py_executable": get_actor_python_env(
                    "nemo_rl.environments.math_environment.MathEnvironment"
                ),
                "env_vars": dict(os.environ),
            }
        ).remote(env_configs["math"])

        # Create dataset
        train_dataset = AllTaskProcessedDataset(
            data.formatted_ds["train"],
            tokenizer,
            math_task_spec,
            task_data_processors,
            max_seq_length=data_config["max_input_seq_length"],
        )

        val_dataset = None
        if data.formatted_ds.get("validation"):
            val_dataset = AllTaskProcessedDataset(
                data.formatted_ds["validation"],
                tokenizer,
                math_task_spec,
                task_data_processors,
                max_seq_length=data_config["max_input_seq_length"],
            )

        # Set up task-to-environment mapping
        task_to_env = defaultdict(lambda: math_env)
        task_to_env[task_name] = math_env

        # Call the legacy setup function
        (
            policy,
            policy_generation,
            cluster,
            dataloader,
            val_dataloader,
            loss_fn,
            grpo_logger,
            checkpointer,
            grpo_state,
            master_config,
        ) = setup(self.config, tokenizer, train_dataset, val_dataset)

        self._policy = policy
        self._policy_generation = policy_generation
        self._logger = grpo_logger
        self._is_setup = True

        # Run training
        print("\n🚀 Running synchronous GRPO training")
        grpo_train(
            policy,
            policy_generation,
            dataloader,
            val_dataloader,
            tokenizer,
            loss_fn,
            task_to_env,
            task_to_env,  # val_task_to_env
            grpo_logger,
            checkpointer,
            grpo_state,
            master_config,
        )

        # Return final metrics
        return {
            "total_steps": grpo_state.get("total_steps", 0),
            "current_epoch": grpo_state.get("current_epoch", 0),
            "val_reward": grpo_state.get("val_reward", None),
        }

    @property
    def effective_batch_size(self) -> int:
        """Get the effective batch size (prompts * generations)."""
        return self.num_prompts_per_step * self.num_generations_per_prompt

    def __repr__(self) -> str:
        return (
            f"GRPOTrainer(model={self.config['policy']['model_name']!r}, "
            f"batch={self.num_prompts_per_step}x{self.num_generations_per_prompt})"
        )
