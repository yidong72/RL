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
"""Startup time benchmarks for NeMo RL.

This module measures startup time for various components:
- Trainer initialization time
- Model loading time
- Backend initialization time

Acceptance criteria:
- AC-13.3: Startup time regression < 10% vs legacy
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tests.benchmarks.utils import BenchmarkConfig, BenchmarkResult, BenchmarkTimer


@dataclass
class StartupMetrics:
    """Metrics for startup time benchmark.
    
    Attributes:
        import_time_sec: Time to import modules.
        config_time_sec: Time to create configuration.
        trainer_init_time_sec: Time to initialize trainer.
        total_startup_time_sec: Total startup time.
    """
    import_time_sec: float = 0.0
    config_time_sec: float = 0.0
    trainer_init_time_sec: float = 0.0
    total_startup_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "import_time_sec": self.import_time_sec,
            "config_time_sec": self.config_time_sec,
            "trainer_init_time_sec": self.trainer_init_time_sec,
            "total_startup_time_sec": self.total_startup_time_sec,
        }


class StartupBenchmark:
    """Benchmark for measuring startup/initialization time.
    
    Example:
        >>> benchmark = StartupBenchmark(
        ...     name="grpo_startup",
        ...     trainer_class="GRPOTrainer",
        ... )
        >>> result = benchmark.run()
        >>> print(f"Startup time: {result.total_time_sec:.2f}s")
    """
    
    def __init__(
        self,
        name: str,
        trainer_class: str = "GRPOTrainer",
        model_name: str = "test-model",
        num_runs: int = 3,
    ):
        """Initialize startup benchmark.
        
        Args:
            name: Benchmark name.
            trainer_class: Name of trainer class to benchmark.
            model_name: Model name for initialization.
            num_runs: Number of runs to average.
        """
        self.name = name
        self.trainer_class = trainer_class
        self.model_name = model_name
        self.num_runs = num_runs
        self.config = BenchmarkConfig(
            name=name,
            model_name=model_name,
            num_steps=0,
            warmup_steps=0,
        )
        
    def _measure_import_time(self) -> float:
        """Measure time to import NeMo RL modules.
        
        Note: First import is slower due to module compilation.
        Subsequent imports use cached modules.
        """
        with BenchmarkTimer() as timer:
            # Import relevant modules for the trainer class
            if self.trainer_class == "GRPOTrainer":
                from nemo_rl.algorithms.grpo import GRPOTrainer
            elif self.trainer_class == "SFTTrainer":
                from nemo_rl.algorithms.sft import SFTTrainer
            elif self.trainer_class == "DPOTrainer":
                from nemo_rl.algorithms.dpo import DPOTrainer
        
        return timer.elapsed
    
    def _measure_config_time(self) -> float:
        """Measure time to create configuration."""
        with BenchmarkTimer() as timer:
            from nemo_rl.algorithms.grpo import GRPOTrainer
            config = GRPOTrainer._build_config_from_pretrained(
                self.model_name,
                max_steps=10,
            )
        
        return timer.elapsed
    
    def _measure_trainer_init_time(self) -> float:
        """Measure time to initialize trainer."""
        with BenchmarkTimer() as timer:
            if self.trainer_class == "GRPOTrainer":
                from nemo_rl.algorithms.grpo import GRPOTrainer
                trainer = GRPOTrainer.from_pretrained(
                    self.model_name,
                    max_steps=10,
                )
            elif self.trainer_class == "SFTTrainer":
                from nemo_rl.algorithms.sft import SFTTrainer
                trainer = SFTTrainer.from_pretrained(
                    self.model_name,
                    max_steps=10,
                )
            elif self.trainer_class == "DPOTrainer":
                from nemo_rl.algorithms.dpo import DPOTrainer
                trainer = DPOTrainer.from_pretrained(
                    self.model_name,
                    max_steps=10,
                )
        
        return timer.elapsed
    
    def run(self) -> BenchmarkResult:
        """Run the startup benchmark.
        
        Returns:
            BenchmarkResult with startup time metrics.
        """
        all_metrics: List[StartupMetrics] = []
        
        for _ in range(self.num_runs):
            metrics = StartupMetrics()
            
            # Measure each phase
            metrics.import_time_sec = self._measure_import_time()
            metrics.config_time_sec = self._measure_config_time()
            metrics.trainer_init_time_sec = self._measure_trainer_init_time()
            metrics.total_startup_time_sec = (
                metrics.import_time_sec + 
                metrics.config_time_sec + 
                metrics.trainer_init_time_sec
            )
            
            all_metrics.append(metrics)
        
        # Average across runs
        avg_metrics = StartupMetrics(
            import_time_sec=sum(m.import_time_sec for m in all_metrics) / self.num_runs,
            config_time_sec=sum(m.config_time_sec for m in all_metrics) / self.num_runs,
            trainer_init_time_sec=sum(m.trainer_init_time_sec for m in all_metrics) / self.num_runs,
            total_startup_time_sec=sum(m.total_startup_time_sec for m in all_metrics) / self.num_runs,
        )
        
        return BenchmarkResult(
            name=self.name,
            config=self.config,
            total_time_sec=avg_metrics.total_startup_time_sec,
            metadata=avg_metrics.to_dict(),
        )


class CheckpointBenchmark:
    """Benchmark for checkpoint save/load operations.
    
    Acceptance criteria:
    - AC-13.4: Checkpoint save/load time unchanged
    
    Example:
        >>> benchmark = CheckpointBenchmark(
        ...     name="checkpoint_save_load",
        ...     state_size_mb=100,
        ... )
        >>> result = benchmark.run()
        >>> print(f"Save time: {result.metadata['save_time_sec']:.2f}s")
    """
    
    def __init__(
        self,
        name: str,
        state_size_mb: int = 100,
        num_runs: int = 3,
    ):
        """Initialize checkpoint benchmark.
        
        Args:
            name: Benchmark name.
            state_size_mb: Size of state to checkpoint (in MB).
            num_runs: Number of runs to average.
        """
        self.name = name
        self.state_size_mb = state_size_mb
        self.num_runs = num_runs
        self.config = BenchmarkConfig(
            name=name,
            num_steps=0,
        )
    
    def _create_state_dict(self) -> Dict[str, Any]:
        """Create a state dict of the specified size."""
        import torch
        
        # Calculate number of parameters to reach target size
        # Each float32 = 4 bytes, so MB * 256k = num params
        num_params = self.state_size_mb * 256 * 1024
        
        # Create state dict with model-like structure
        state_dict = {
            "model": {
                "layer1": torch.randn(num_params // 4, 1),
                "layer2": torch.randn(num_params // 4, 1),
            },
            "optimizer": {
                "state": {
                    0: {
                        "exp_avg": torch.randn(num_params // 8, 1),
                        "exp_avg_sq": torch.randn(num_params // 8, 1),
                    }
                },
                "param_groups": [{"lr": 1e-6}],
            },
            "trainer_state": {
                "epoch": 5,
                "global_step": 1000,
            },
        }
        
        return state_dict
    
    def run(self) -> BenchmarkResult:
        """Run the checkpoint benchmark.
        
        Returns:
            BenchmarkResult with checkpoint timing metrics.
        """
        import tempfile
        from pathlib import Path
        
        from nemo_rl.infra.checkpointing import CheckpointManager, CheckpointFormat
        
        save_times: List[float] = []
        load_times: List[float] = []
        
        for _ in range(self.num_runs):
            with tempfile.TemporaryDirectory() as tmpdir:
                manager = CheckpointManager(
                    checkpoint_dir=tmpdir,
                    format=CheckpointFormat.PYTORCH,
                )
                
                # Create state dict
                state_dict = self._create_state_dict()
                
                # Measure save time
                with BenchmarkTimer() as save_timer:
                    path = manager.save(state_dict, step=100)
                
                save_times.append(save_timer.elapsed)
                
                # Measure load time
                with BenchmarkTimer() as load_timer:
                    loaded = manager.load(path)
                
                load_times.append(load_timer.elapsed)
        
        # Calculate averages
        avg_save_time = sum(save_times) / self.num_runs
        avg_load_time = sum(load_times) / self.num_runs
        
        return BenchmarkResult(
            name=self.name,
            config=self.config,
            total_time_sec=avg_save_time + avg_load_time,
            metadata={
                "save_time_sec": avg_save_time,
                "load_time_sec": avg_load_time,
                "state_size_mb": self.state_size_mb,
                "num_runs": self.num_runs,
            },
        )


def run_startup_benchmarks() -> List[BenchmarkResult]:
    """Run all startup benchmarks.
    
    Returns:
        List of benchmark results.
    """
    results = []
    
    trainers = [
        ("grpo_startup", "GRPOTrainer"),
        ("sft_startup", "SFTTrainer"),
        ("dpo_startup", "DPOTrainer"),
    ]
    
    for name, trainer_class in trainers:
        benchmark = StartupBenchmark(
            name=name,
            trainer_class=trainer_class,
            num_runs=3,
        )
        result = benchmark.run()
        results.append(result)
        print(f"  {name}: {result.total_time_sec:.3f}s")
    
    return results


def run_checkpoint_benchmarks() -> List[BenchmarkResult]:
    """Run checkpoint save/load benchmarks.
    
    Returns:
        List of benchmark results.
    """
    results = []
    
    # Test different state sizes
    sizes_mb = [10, 50, 100]
    
    for size_mb in sizes_mb:
        benchmark = CheckpointBenchmark(
            name=f"checkpoint_{size_mb}mb",
            state_size_mb=size_mb,
            num_runs=3,
        )
        result = benchmark.run()
        results.append(result)
        print(f"  {result.name}: save={result.metadata['save_time_sec']:.3f}s, "
              f"load={result.metadata['load_time_sec']:.3f}s")
    
    return results


__all__ = [
    "StartupBenchmark",
    "StartupMetrics",
    "CheckpointBenchmark",
    "run_startup_benchmarks",
    "run_checkpoint_benchmarks",
]
