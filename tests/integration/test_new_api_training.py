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
"""Test the new API training functionality.

This script tests actual training execution with the new API.
"""

import argparse
import os
import sys
import traceback


def test_trainer_fit_simple():
    """Test the trainer.fit() method with simple data."""
    print("\n" + "=" * 60)
    print("Testing trainer.fit() with simple data...")
    print("=" * 60)
    
    from nemo_rl.algorithms.grpo import GRPOTrainer
    from nemo_rl.data.module import InMemoryDataModule
    
    # Create trainer
    print("  Creating GRPOTrainer.from_pretrained...")
    trainer = GRPOTrainer.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        num_prompts_per_step=2,
        num_generations_per_prompt=2,
        learning_rate=1e-6,
        max_steps=2,
    )
    print(f"  Trainer created: {trainer}")
    print(f"  Config: {trainer.config}")
    
    # Create simple data
    train_data = [
        {"prompt": "What is 2+2?"},
        {"prompt": "What is 3+3?"},
        {"prompt": "What is 4+4?"},
        {"prompt": "What is 5+5?"},
    ]
    
    datamodule = InMemoryDataModule(train_data=train_data, batch_size=2)
    
    # Try to fit
    print("  Calling trainer.fit()...")
    try:
        result = trainer.fit(
            datamodule=datamodule,
            max_steps=2,
        )
        print(f"  Result: {result}")
        print("  fit() completed!")
        return True
    except Exception as e:
        print(f"  ERROR in fit(): {e}")
        traceback.print_exc()
        return False


def test_trainer_setup():
    """Test the trainer.setup() method."""
    print("\n" + "=" * 60)
    print("Testing trainer.setup()...")
    print("=" * 60)
    
    from nemo_rl.algorithms.grpo import GRPOTrainer
    
    trainer = GRPOTrainer.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        num_prompts_per_step=2,
        num_generations_per_prompt=2,
        max_steps=2,
    )
    print(f"  Trainer created: {trainer}")
    
    print("  Calling trainer.setup()...")
    try:
        trainer.setup()
        print("  setup() completed!")
        print(f"  Logger: {trainer._logger}")
        print(f"  Checkpoint manager: {trainer._checkpoint_manager}")
        print(f"  Resource manager: {trainer._resource_manager}")
        return True
    except Exception as e:
        print(f"  ERROR in setup(): {e}")
        traceback.print_exc()
        return False


def test_config_accessors():
    """Test config accessor methods."""
    print("\n" + "=" * 60)
    print("Testing config accessor methods...")
    print("=" * 60)
    
    from nemo_rl.algorithms.grpo import GRPOTrainer
    
    trainer = GRPOTrainer.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        num_prompts_per_step=2,
        num_generations_per_prompt=4,
        max_steps=100,
    )
    
    print(f"  Config type: {type(trainer.config)}")
    print(f"  Config keys: {list(trainer.config.keys()) if isinstance(trainer.config, dict) else 'N/A'}")
    
    # Test accessor methods
    try:
        seed = trainer._get_seed()
        print(f"  _get_seed(): {seed}")
    except Exception as e:
        print(f"  _get_seed() ERROR: {e}")
    
    try:
        max_epochs = trainer._get_max_epochs()
        print(f"  _get_max_epochs(): {max_epochs}")
    except Exception as e:
        print(f"  _get_max_epochs() ERROR: {e}")
    
    try:
        max_steps = trainer._get_max_steps()
        print(f"  _get_max_steps(): {max_steps}")
    except Exception as e:
        print(f"  _get_max_steps() ERROR: {e}")
    
    try:
        log_interval = trainer._get_log_interval()
        print(f"  _get_log_interval(): {log_interval}")
    except Exception as e:
        print(f"  _get_log_interval() ERROR: {e}")
    
    try:
        logger_config = trainer._get_logger_config()
        print(f"  _get_logger_config(): {logger_config}")
    except Exception as e:
        print(f"  _get_logger_config() ERROR: {e}")
    
    try:
        cluster_config = trainer._get_cluster_config()
        print(f"  _get_cluster_config(): {cluster_config}")
    except Exception as e:
        print(f"  _get_cluster_config() ERROR: {e}")
    
    try:
        ckpt_config = trainer._get_checkpointing_config()
        print(f"  _get_checkpointing_config(): {ckpt_config}")
    except Exception as e:
        print(f"  _get_checkpointing_config() ERROR: {e}")
    
    return True


def run_all_tests():
    """Run all training tests."""
    print("\n" + "=" * 60)
    print("NeMo RL New API Training Tests")
    print("=" * 60)
    
    tests = [
        ("Config Accessors", test_config_accessors),
        ("Trainer Setup", test_trainer_setup),
        ("Trainer Fit", test_trainer_fit_simple),
    ]
    
    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"\nFailed: {name}")
            print(f"  Error: {e}")
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(results.values())
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed!"))
    
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test new NeMo RL API training")
    parser.add_argument("--test", type=str, default="all",
                       help="Which test to run: config, setup, fit, all")
    args = parser.parse_args()
    
    if args.test == "all":
        success = run_all_tests()
    elif args.test == "config":
        success = test_config_accessors()
    elif args.test == "setup":
        success = test_trainer_setup()
    elif args.test == "fit":
        success = test_trainer_fit_simple()
    else:
        print(f"Unknown test: {args.test}")
        success = False
    
    sys.exit(0 if success else 1)
