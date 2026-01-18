#!/usr/bin/env python3
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
"""Test script for the new GRPOTrainer API.

This script tests the simplified GRPOTrainer.from_pretrained() API
on the DeepScaler dataset with the DeepSeek-R1-Distill-Qwen-1.5B model.

Usage:
    python examples/test_new_grpo_api.py [--max-steps N] [--gpus-per-node N]
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Test new GRPO API")
    parser.add_argument("--max-steps", type=int, default=10, help="Max training steps")
    parser.add_argument("--gpus-per-node", type=int, default=8, help="GPUs per node")
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--num-prompts-per-step", type=int, default=8, help="Prompts per step")
    parser.add_argument("--num-generations-per-prompt", type=int, default=4, help="Generations per prompt")
    parser.add_argument("--log-dir", type=str, default="/lustre/fsw/portfolios/nvr/users/yidong/logs/grpo-new-api-test", help="Log directory")
    parser.add_argument("--checkpoint-dir", type=str, default="/lustre/fsw/portfolios/nvr/users/yidong/results/grpo-new-api-test", help="Checkpoint directory")
    args = parser.parse_args()

    print("=" * 60)
    print(" Testing New GRPOTrainer API ")
    print("=" * 60)
    print(f"Max steps: {args.max_steps}")
    print(f"GPUs per node: {args.gpus_per_node}")
    print(f"Num nodes: {args.num_nodes}")
    print(f"Prompts per step: {args.num_prompts_per_step}")
    print(f"Generations per prompt: {args.num_generations_per_prompt}")
    print("=" * 60)

    # Import the new API
    from nemo_rl.algorithms.grpo import GRPOTrainer

    # Model path - use HuggingFace name (let HF handle caching and downloads)
    # This is more reliable than checking for local paths
    import os
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    print(f"Using HuggingFace model: {model_name}")

    # Create trainer using the simple API
    print("\n▶ Creating GRPOTrainer from pretrained model...")
    trainer = GRPOTrainer.from_pretrained(
        model_name,
        num_prompts_per_step=args.num_prompts_per_step,
        num_generations_per_prompt=args.num_generations_per_prompt,
        max_steps=args.max_steps,
        max_epochs=1,
        gpus_per_node=args.gpus_per_node,
        num_nodes=args.num_nodes,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        checkpointing_enabled=False,  # Disable checkpointing for quick test
        tensorboard_enabled=True,
        wandb_enabled=False,
    )

    print(f"\n✓ Trainer created: {trainer}")
    print(f"  Effective batch size: {trainer.effective_batch_size}")

    # Run training on DeepScaler dataset
    print("\n▶ Starting training on DeepScaler dataset...")
    result = trainer.fit(dataset="DeepScaler")

    print("\n" + "=" * 60)
    print(" Training Complete! ")
    print("=" * 60)
    # TrainingResult is a dataclass with metrics dict, total_steps, etc.
    print(f"Total steps: {result.total_steps}")
    print(f"Metrics: {result.metrics}")
    if result.best_checkpoint_path:
        print(f"Checkpoint: {result.best_checkpoint_path}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
