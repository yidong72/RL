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
"""NeMo RL Algorithms Module.

This module provides RL training algorithms including:
- GRPO (Group Relative Policy Optimization)
- SFT (Supervised Fine-Tuning)
- DPO (Direct Preference Optimization)
- Rollout orchestration

Example:
    >>> from nemo_rl.algorithms import RolloutEngine, SamplingParams
    >>> 
    >>> engine = RolloutEngine(backend, environment)
    >>> result = engine.rollout(prompts, SamplingParams(temperature=0.7))
    
    >>> # Or use GRPO trainer
    >>> from nemo_rl.algorithms.grpo import GRPOTrainer
    >>> trainer = GRPOTrainer(config)
    >>> trainer.fit(dataset)
"""

# Import only trainers directly - they have minimal dependencies
from nemo_rl.algorithms.grpo import GRPOTrainer
from nemo_rl.algorithms.sft import SFTTrainer
from nemo_rl.algorithms.dpo import DPOTrainer


def __getattr__(name: str):
    """Lazy import of algorithm components to avoid heavy import chain."""
    # Rollout components
    if name in ("RolloutEngine", "RolloutResult", "SamplingParams", "create_rollout_engine"):
        from nemo_rl.algorithms.rollout import (
            RolloutEngine, RolloutResult, SamplingParams, create_rollout_engine,
        )
        return {
            "RolloutEngine": RolloutEngine,
            "RolloutResult": RolloutResult,
            "SamplingParams": SamplingParams,
            "create_rollout_engine": create_rollout_engine,
        }[name]
    
    # Config classes - lazy load to avoid heavy imports
    if name == "GRPOConfig":
        from nemo_rl.algorithms.grpo import GRPOConfig
        return GRPOConfig
    
    if name == "SFTConfig":
        from nemo_rl.algorithms.sft import SFTConfig
        return SFTConfig
    
    if name == "DPOConfig":
        from nemo_rl.algorithms.dpo import DPOConfig
        return DPOConfig
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Rollout
    "RolloutEngine",
    "RolloutResult",
    "SamplingParams",
    "create_rollout_engine",
    # GRPO
    "GRPOTrainer",
    "GRPOConfig",
    # SFT
    "SFTTrainer",
    "SFTConfig",
    # DPO
    "DPOTrainer",
    "DPOConfig",
]
