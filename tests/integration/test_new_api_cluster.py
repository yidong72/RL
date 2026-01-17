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
"""Integration test for new NeMo RL API on cluster.

This script tests the new API endpoints:
1. nemo_rl.train() - simplest API
2. GRPOTrainer.from_pretrained() - HuggingFace-style API
3. SFTTrainer and DPOTrainer

Run this inside the Slurm container with GPU access.
"""

import argparse
import os
import sys
from pathlib import Path


def test_imports():
    """Test that all new API imports work."""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)
    
    # Test main module imports
    import nemo_rl
    print(f"  nemo_rl version: {nemo_rl.__version__}")
    
    # Test lazy imports from nemo_rl
    assert hasattr(nemo_rl, 'train'), "nemo_rl.train should exist"
    assert hasattr(nemo_rl, 'TrainResult'), "nemo_rl.TrainResult should exist"
    print("  nemo_rl.train: OK")
    print("  nemo_rl.TrainResult: OK")
    
    # Test trainer imports
    from nemo_rl.algorithms.grpo import GRPOTrainer
    from nemo_rl.algorithms.sft import SFTTrainer
    from nemo_rl.algorithms.dpo import DPOTrainer
    print("  GRPOTrainer: OK")
    print("  SFTTrainer: OK")
    print("  DPOTrainer: OK")
    
    # Test API helpers
    from nemo_rl.api import create_trainer, list_algorithms, get_algorithm
    algorithms = list_algorithms()
    print(f"  Available algorithms: {algorithms}")
    assert 'grpo' in algorithms, "grpo should be in algorithms"
    assert 'sft' in algorithms, "sft should be in algorithms"
    
    # Test BaseTrainer
    from nemo_rl.trainers.base import BaseTrainer, TrainingResult
    print("  BaseTrainer: OK")
    print("  TrainingResult: OK")
    
    # Test DataModule
    from nemo_rl.data.module import DataModule, create_datamodule
    print("  DataModule: OK")
    
    # Test callbacks
    from nemo_rl.trainers.callbacks import (
        Callback, CallbackList, CheckpointCallback,
        LoggingCallback, EarlyStoppingCallback
    )
    print("  Callbacks: OK")
    
    print("\nAll imports successful!")
    return True


def test_from_pretrained():
    """Test the from_pretrained API."""
    print("\n" + "=" * 60)
    print("Testing from_pretrained API...")
    print("=" * 60)
    
    from nemo_rl.algorithms.grpo import GRPOTrainer
    
    # Test GRPOTrainer.from_pretrained
    print("  Creating GRPOTrainer.from_pretrained...")
    trainer = GRPOTrainer.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        num_prompts_per_step=2,
        num_generations_per_prompt=4,
        learning_rate=1e-6,
        max_steps=10,
    )
    print(f"  Trainer created: {trainer}")
    print(f"  Config type: {type(trainer.config)}")
    
    # Verify config values
    assert trainer.num_prompts_per_step == 2, "num_prompts_per_step should be 2"
    assert trainer.num_generations_per_prompt == 4, "num_generations_per_prompt should be 4"
    print("  Config values verified!")
    
    # Test SFTTrainer.from_pretrained
    from nemo_rl.algorithms.sft import SFTTrainer
    print("  Creating SFTTrainer.from_pretrained...")
    sft_trainer = SFTTrainer.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        batch_size=4,
        learning_rate=2e-5,
    )
    print(f"  SFT Trainer created: {sft_trainer}")
    
    # Test DPOTrainer.from_pretrained
    from nemo_rl.algorithms.dpo import DPOTrainer
    print("  Creating DPOTrainer.from_pretrained...")
    dpo_trainer = DPOTrainer.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        beta=0.1,
        batch_size=4,
    )
    print(f"  DPO Trainer created: {dpo_trainer}")
    
    print("\nfrom_pretrained API tests passed!")
    return True


def test_create_trainer():
    """Test the create_trainer functional API."""
    print("\n" + "=" * 60)
    print("Testing create_trainer API...")
    print("=" * 60)
    
    from nemo_rl.api import create_trainer
    
    # Create GRPO trainer
    print("  Creating GRPO trainer via create_trainer...")
    trainer = create_trainer(
        "grpo",
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        learning_rate=1e-6,
        batch_size=2,
    )
    print(f"  Created: {trainer}")
    
    # Create SFT trainer
    print("  Creating SFT trainer via create_trainer...")
    sft_trainer = create_trainer(
        "sft",
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        learning_rate=2e-5,
    )
    print(f"  Created: {sft_trainer}")
    
    print("\ncreate_trainer API tests passed!")
    return True


def test_callbacks():
    """Test the callback system."""
    print("\n" + "=" * 60)
    print("Testing callback system...")
    print("=" * 60)
    
    from nemo_rl.trainers.callbacks import (
        Callback, CallbackList, LoggingCallback, 
        EarlyStoppingCallback, LambdaCallback
    )
    
    # Create a simple mock trainer for callbacks that need it
    class MockState:
        total_steps = 100
        should_stop = False
    
    class MockTrainer:
        state = MockState()
        logger = None
        checkpoint_manager = None
        global_step = 10
    
    mock_trainer = MockTrainer()
    call_log = []
    
    class TestCallback(Callback):
        def on_train_begin(self, trainer):
            call_log.append("train_begin")
        
        def on_step_end(self, trainer, step, logs):
            call_log.append(f"step_{step}")
    
    # Test CallbackList with just the TestCallback (no LoggingCallback)
    # LoggingCallback logs to console anyway which clutters output
    cb_list = CallbackList([
        TestCallback(),
    ])
    
    # Simulate training lifecycle with mock trainer
    cb_list.on_train_begin(mock_trainer)
    cb_list.on_step_end(mock_trainer, 0, {"loss": 0.5})
    cb_list.on_step_end(mock_trainer, 1, {"loss": 0.4})
    cb_list.on_train_end(mock_trainer)
    
    assert "train_begin" in call_log, "on_train_begin should be called"
    assert "step_0" in call_log, "on_step_end(0) should be called"
    assert "step_1" in call_log, "on_step_end(1) should be called"
    
    print("  CallbackList working correctly!")
    
    # Test LoggingCallback separately with mock trainer
    logging_cb = LoggingCallback(log_every=1, log_to_console=False)
    logging_cb.on_step_end(mock_trainer, 0, {"loss": 0.5})  # Should not crash
    logging_cb.on_train_end(mock_trainer)
    print("  LoggingCallback working correctly!")
    
    # Test LambdaCallback
    lambda_calls = []
    lambda_cb = LambdaCallback(
        on_train_begin=lambda t: lambda_calls.append("begin"),
        on_train_end=lambda t: lambda_calls.append("end"),
    )
    lambda_cb.on_train_begin(mock_trainer)
    lambda_cb.on_train_end(mock_trainer)
    assert lambda_calls == ["begin", "end"], "LambdaCallback should work"
    print("  LambdaCallback working correctly!")
    
    print("\nCallback tests passed!")
    return True


def test_datamodule():
    """Test the DataModule system."""
    print("\n" + "=" * 60)
    print("Testing DataModule system...")
    print("=" * 60)
    
    from nemo_rl.data.module import (
        DataModule, InMemoryDataModule, create_datamodule
    )
    
    # Test InMemoryDataModule
    train_data = [
        {"prompt": "What is 2+2?", "response": "4"},
        {"prompt": "What is 3+3?", "response": "6"},
        {"prompt": "What is 4+4?", "response": "8"},
        {"prompt": "What is 5+5?", "response": "10"},
    ]
    
    dm = InMemoryDataModule(train_data=train_data, batch_size=2)
    dm.setup("fit")
    
    train_loader = dm.train_dataloader()
    batches = list(train_loader)
    print(f"  Created {len(batches)} batches from InMemoryDataModule")
    
    # Test create_datamodule factory
    dm2 = create_datamodule(train_data, batch_size=2)
    dm2.setup("fit")
    print("  create_datamodule factory works!")
    
    print("\nDataModule tests passed!")
    return True


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("NeMo RL New API Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("from_pretrained", test_from_pretrained),
        ("create_trainer", test_create_trainer),
        ("Callbacks", test_callbacks),
        ("DataModule", test_datamodule),
    ]
    
    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"\nFailed: {name}")
            print(f"  Error: {e}")
            import traceback
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
    parser = argparse.ArgumentParser(description="Test new NeMo RL API")
    parser.add_argument("--test", type=str, default="all",
                       help="Which test to run: imports, from_pretrained, create_trainer, callbacks, datamodule, all")
    args = parser.parse_args()
    
    if args.test == "all":
        success = run_all_tests()
    elif args.test == "imports":
        success = test_imports()
    elif args.test == "from_pretrained":
        success = test_from_pretrained()
    elif args.test == "create_trainer":
        success = test_create_trainer()
    elif args.test == "callbacks":
        success = test_callbacks()
    elif args.test == "datamodule":
        success = test_datamodule()
    else:
        print(f"Unknown test: {args.test}")
        success = False
    
    sys.exit(0 if success else 1)
