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
"""SFT Algorithm Package.

This package provides the Supervised Fine-Tuning (SFT) algorithm
split into focused modules:

- config.py: Configuration classes (SFTConfig, MasterConfig, etc.)
- loss.py: SFT loss function (NLLLoss wrapper)
- data.py: Data transforms and batch processing
- trainer.py: SFTTrainer extending BaseTrainer

For backward compatibility, all public symbols from the original sft.py
are re-exported from this package.

Example (new API):
    >>> from nemo_rl.algorithms.sft import SFTTrainer, SFTConfig
    >>> 
    >>> trainer = SFTTrainer(config)
    >>> trainer.fit(dataset="nvidia/OpenMathInstruct-2")

Example (backward compatible):
    >>> from nemo_rl.algorithms.sft import (
    ...     MasterConfig, sft_train, setup
    ... )
    >>> 
    >>> # Legacy API still works
    >>> sft_train(...)
"""

# Import only the trainer directly - it has minimal dependencies
from nemo_rl.algorithms.sft.trainer import SFTTrainer


def __getattr__(name: str):
    """Lazy import of SFT components to avoid heavy import chain."""
    # Config classes
    if name in ("MasterConfig", "SFTConfig", "SFTSaveState", "default_sft_save_state"):
        from nemo_rl.algorithms.sft.config import (
            MasterConfig, SFTConfig, SFTSaveState, default_sft_save_state,
        )
        return {
            "MasterConfig": MasterConfig,
            "SFTConfig": SFTConfig,
            "SFTSaveState": SFTSaveState,
            "default_sft_save_state": default_sft_save_state,
        }[name]
    
    # Data functions
    if name == "prepare_batch_for_sft":
        from nemo_rl.algorithms.sft.data import prepare_batch_for_sft
        return prepare_batch_for_sft
    
    # Loss functions
    if name in ("SFTLoss", "create_sft_loss_function"):
        from nemo_rl.algorithms.sft.loss import SFTLoss, create_sft_loss_function
        return {"SFTLoss": SFTLoss, "create_sft_loss_function": create_sft_loss_function}[name]
    
    # Legacy API (backward compatibility)
    if name in ("_default_sft_save_state", "setup", "sft_train", "validate"):
        from nemo_rl.algorithms.sft_legacy import (
            _default_sft_save_state, setup, sft_train, validate,
        )
        return {
            "_default_sft_save_state": _default_sft_save_state,
            "setup": setup,
            "sft_train": sft_train,
            "validate": validate,
        }[name]
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # New modular API
    "SFTTrainer",
    # Config classes
    "SFTConfig",
    "MasterConfig",
    "SFTSaveState",
    "default_sft_save_state",
    # Loss functions
    "SFTLoss",
    "create_sft_loss_function",
    # Data functions
    "prepare_batch_for_sft",
    # Legacy API (backward compatibility)
    "setup",
    "sft_train",
    "validate",
    "_default_sft_save_state",
]
