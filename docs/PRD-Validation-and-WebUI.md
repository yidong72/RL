# Product Requirements Document: NeMo RL Refactoring Validation & Web Configuration Tool

**Document Version:** 1.0  
**Date:** January 15, 2026  
**Project Duration:** 6 months  
**Target Audience:** Engineering Team, Researchers, Product Management, QA Team

---

## Executive Summary

This PRD defines the requirements for two major initiatives following the NeMo RL architecture refactoring:

1. **Refactoring Validation Phase**: Comprehensive testing and validation of the newly refactored codebase to ensure all changes meet quality standards and pass acceptance criteria.

2. **NeMo RL Web Configuration Tool**: A web-based application that provides a user-friendly interface for configuring and deploying NeMo RL training jobs on SLURM clusters.

### Project Scope Summary

| Initiative | Duration | Team Size | Priority |
|-----------|----------|-----------|----------|
| Refactoring Validation | 2 months | 3-4 engineers | P0 |
| Web Configuration Tool | 4 months | 4-6 engineers | P1 |

---

## Table of Contents

1. [Part 1: Refactoring Validation](#part-1-refactoring-validation)
   - [1.1 Overview of Refactoring Changes](#11-overview-of-refactoring-changes)
   - [1.2 Test Strategy](#12-test-strategy)
   - [1.3 Acceptance Criteria](#13-acceptance-criteria)
   - [1.4 Test Plan](#14-test-plan)
   - [1.5 Risk Assessment](#15-risk-assessment)
2. [Part 2: Web Configuration Tool](#part-2-web-configuration-tool)
   - [2.1 Product Vision](#21-product-vision)
   - [2.2 User Personas](#22-user-personas)
   - [2.3 Feature Requirements](#23-feature-requirements)
   - [2.4 Technical Architecture](#24-technical-architecture)
   - [2.5 UI/UX Requirements](#25-uiux-requirements)
   - [2.6 Implementation Phases](#26-implementation-phases)
3. [Success Metrics](#success-metrics)
4. [Timeline and Milestones](#timeline-and-milestones)
5. [Appendix](#appendix)

---

# Part 1: Refactoring Validation

## 1.1 Overview of Refactoring Changes

The refactoring effort introduced ~37,000 lines of new/modified code across 130 files. The major changes include:

### 1.1.1 New Module Summary

| Module | Files Added | Lines | Description |
|--------|------------|-------|-------------|
| `nemo_rl/api/` | 3 | ~780 | High-level training API (`train()`, functional helpers) |
| `nemo_rl/trainers/` | 4 | ~2,000 | Base trainer, callbacks, validation runner |
| `nemo_rl/config/` | 8 | ~2,700 | Configuration system with Pydantic validation |
| `nemo_rl/backends/` | 8 | ~2,200 | Backend abstraction (training + generation) |
| `nemo_rl/compat/` | 4 | ~1,200 | Backward compatibility layer |
| `nemo_rl/infra/` | 4 | ~2,250 | Resource management, checkpointing, logging |
| `nemo_rl/environments/` | 3 | ~1,400 | Environment base class, functional rewards |
| `nemo_rl/algorithms/grpo/` | 6 | ~1,000 | Decomposed GRPO algorithm |
| `nemo_rl/algorithms/sft/` | 5 | ~470 | Decomposed SFT algorithm |
| `nemo_rl/algorithms/dpo/` | 5 | ~530 | Decomposed DPO algorithm |
| `tests/` | 45+ | ~15,000 | Unit and integration tests |

### 1.1.2 Key API Changes

| Old API | New API | Status |
|---------|---------|--------|
| `GRPO.from_config(cfg)` | `GRPOTrainer.from_pretrained()` | Implemented |
| `grpo_train()` function | `trainer.fit()` method | Implemented |
| Environment Ray actors | Simple callable `reward_fn` | Implemented |
| YAML-first configuration | Programmatic `nemo_rl.train()` | Implemented |
| TypedDict configs | Pydantic-validated configs | Implemented |

### 1.1.3 Files Modified from Original

| File | Change Type | Lines Changed |
|------|-------------|---------------|
| `nemo_rl/models/policy/lm_policy.py` | Refactored | -555 lines |
| `nemo_rl/algorithms/sft.py` | Renamed to `sft_legacy.py` | - |
| `nemo_rl/__init__.py` | New public API exports | +42 lines |
| `pyproject.toml` | Dependency update | +1 line |

---

## 1.2 Test Strategy

### 1.2.1 Test Pyramid

```
                    ┌───────────────┐
                    │  E2E Tests    │  5-10 tests
                    │  (Production) │  Real clusters
                    ├───────────────┤
                    │  Integration  │  20-30 tests
                    │    Tests      │  Multi-component
              ┌─────┴───────────────┴─────┐
              │       Unit Tests          │  200+ tests
              │   (New refactored code)   │  Isolated
              └───────────────────────────┘
```

### 1.2.2 Test Categories

| Category | Scope | Environment | Count |
|----------|-------|-------------|-------|
| **Unit Tests** | Single module/function | Local, mocked | 200+ |
| **Integration Tests** | Multi-module interaction | Local + Ray | 30+ |
| **E2E Tests** | Full training pipeline | SLURM cluster | 10+ |
| **Regression Tests** | Old API compatibility | Local + cluster | 20+ |
| **Performance Tests** | Throughput/memory | SLURM cluster | 10+ |

### 1.2.3 Test Infrastructure Requirements

| Resource | Specification | Purpose |
|----------|--------------|---------|
| CI/CD | GitHub Actions | Unit + integration tests |
| GPU Cluster | 4 nodes × 8 GPUs (H100) | E2E and performance tests |
| Storage | 10TB NFS | Checkpoints, datasets |
| Container | NGC PyTorch 24.12 | Consistent environment |

---

## 1.3 Acceptance Criteria

### 1.3.1 Core API Acceptance Criteria

#### AC-1: Simple Training API (`nemo_rl.train()`)

| ID | Criterion | Test Method |
|----|-----------|-------------|
| AC-1.1 | `nemo_rl.train()` accepts model string, dataset string, and reward function | Unit test |
| AC-1.2 | Training completes with valid configuration in < 5 minutes (small model) | Integration test |
| AC-1.3 | `TrainResult` contains trainer, metrics, checkpoint path | Unit test |
| AC-1.4 | Invalid inputs raise `ValueError` with helpful message | Unit test |
| AC-1.5 | Default parameters produce successful training | E2E test |

**Test Snippet:**
```python
# Test AC-1.1, AC-1.2, AC-1.3
def test_simple_train_api():
    result = nemo_rl.train(
        model="Qwen/Qwen2.5-1.5B",
        dataset="nvidia/OpenMathInstruct-2",
        reward_fn=lambda p, r: 1.0,
        max_steps=10,
    )
    assert isinstance(result.trainer, BaseTrainer)
    assert "loss" in result.metrics
    assert result.checkpoint_path is not None
```

#### AC-2: Trainer.from_pretrained()

| ID | Criterion | Test Method |
|----|-----------|-------------|
| AC-2.1 | `GRPOTrainer.from_pretrained("model_name")` creates valid trainer | Unit test |
| AC-2.2 | `SFTTrainer.from_pretrained("model_name")` creates valid trainer | Unit test |
| AC-2.3 | `DPOTrainer.from_pretrained("model_name")` creates valid trainer | Unit test |
| AC-2.4 | Local path models load correctly | Integration test |
| AC-2.5 | HuggingFace Hub models download and load | Integration test |
| AC-2.6 | Invalid model names raise descriptive errors | Unit test |

#### AC-3: Configuration System

| ID | Criterion | Test Method |
|----|-----------|-------------|
| AC-3.1 | All configs validate at construction time | Unit test |
| AC-3.2 | Invalid config values raise `ValidationError` | Unit test |
| AC-3.3 | Config defaults produce working configurations | Integration test |
| AC-3.4 | YAML configs load via `BaseConfig.from_yaml()` | Unit test |
| AC-3.5 | Config serializes to dict/JSON correctly | Unit test |
| AC-3.6 | Nested configs validate recursively | Unit test |

#### AC-4: Backend Abstraction

| ID | Criterion | Test Method |
|----|-----------|-------------|
| AC-4.1 | `get_training_backend("dtensor")` returns DTensorBackend | Unit test |
| AC-4.2 | `get_training_backend("megatron")` returns MegatronBackend | Unit test |
| AC-4.3 | `get_generation_backend("vllm")` returns VLLMBackend | Unit test |
| AC-4.4 | Unknown backend raises `BackendNotFoundError` | Unit test |
| AC-4.5 | Backend registration decorator works | Unit test |
| AC-4.6 | Backend switching works without code changes | Integration test |

#### AC-5: Functional Reward Functions

| ID | Criterion | Test Method |
|----|-----------|-------------|
| AC-5.1 | Lambda function `(p, r) -> float` works as reward | Unit test |
| AC-5.2 | Named function works as reward | Unit test |
| AC-5.3 | Class with `__call__` works as reward | Unit test |
| AC-5.4 | Batch reward function `([p], [r]) -> [float]` works | Unit test |
| AC-5.5 | `FunctionalRewardWrapper` creates valid environment | Unit test |
| AC-5.6 | Rewards integrate with GRPO training loop | Integration test |

#### AC-6: Backward Compatibility

| ID | Criterion | Test Method |
|----|-----------|-------------|
| AC-6.1 | Existing YAML configs work with deprecation warnings | Integration test |
| AC-6.2 | Old `GRPO.from_config()` API still functions | Unit test |
| AC-6.3 | Old Environment classes still work | Integration test |
| AC-6.4 | Deprecation warnings point to migration guide | Unit test |
| AC-6.5 | No breaking changes to checkpoint format | Integration test |

### 1.3.2 Algorithm-Specific Acceptance Criteria

#### AC-7: GRPO Algorithm

| ID | Criterion | Test Method |
|----|-----------|-------------|
| AC-7.1 | GRPO loss computation matches legacy implementation | Unit test |
| AC-7.2 | Rollout generation produces valid responses | Integration test |
| AC-7.3 | Leave-one-out baseline computes correctly | Unit test |
| AC-7.4 | Reward normalization works | Unit test |
| AC-7.5 | Multi-turn generation works | Integration test |
| AC-7.6 | Training produces improving rewards over steps | E2E test |
| AC-7.7 | KL divergence computation is correct | Unit test |
| AC-7.8 | Async GRPO mode works | E2E test |

#### AC-8: SFT Algorithm

| ID | Criterion | Test Method |
|----|-----------|-------------|
| AC-8.1 | SFT loss (cross-entropy) computes correctly | Unit test |
| AC-8.2 | Data loading handles various formats | Unit test |
| AC-8.3 | Training produces decreasing loss | E2E test |
| AC-8.4 | LoRA fine-tuning works | Integration test |

#### AC-9: DPO Algorithm

| ID | Criterion | Test Method |
|----|-----------|-------------|
| AC-9.1 | DPO loss computation is correct | Unit test |
| AC-9.2 | Preference data loading works | Unit test |
| AC-9.3 | Beta parameter affects training correctly | Unit test |
| AC-9.4 | Training produces expected behavior | E2E test |

### 1.3.3 Infrastructure Acceptance Criteria

#### AC-10: Checkpointing

| ID | Criterion | Test Method |
|----|-----------|-------------|
| AC-10.1 | Checkpoints save at configured intervals | Integration test |
| AC-10.2 | Training resumes from checkpoint | Integration test |
| AC-10.3 | Best checkpoint tracked correctly | Unit test |
| AC-10.4 | Checkpoint format is backward compatible | Integration test |
| AC-10.5 | Distributed checkpoint works across nodes | E2E test |

#### AC-11: Logging

| ID | Criterion | Test Method |
|----|-----------|-------------|
| AC-11.1 | TensorBoard logging works | Integration test |
| AC-11.2 | WandB logging works (optional) | Integration test |
| AC-11.3 | Console logging is configurable | Unit test |
| AC-11.4 | Metrics logged at correct intervals | Integration test |

#### AC-12: Resource Management

| ID | Criterion | Test Method |
|----|-----------|-------------|
| AC-12.1 | GPU allocation respects config | Integration test |
| AC-12.2 | Multi-node training works | E2E test |
| AC-12.3 | Resource cleanup on failure | Integration test |
| AC-12.4 | Memory tracking is accurate | Unit test |

### 1.3.4 Performance Acceptance Criteria

#### AC-13: Performance Regression

| ID | Criterion | Test Method |
|----|-----------|-------------|
| AC-13.1 | Throughput regression < 5% vs legacy | Performance test |
| AC-13.2 | Memory usage regression < 5% vs legacy | Performance test |
| AC-13.3 | Startup time regression < 10% vs legacy | Performance test |
| AC-13.4 | Checkpoint save/load time unchanged | Performance test |

---

## 1.4 Test Plan

### 1.4.1 Unit Test Execution

**New Unit Test Files to Execute:**

| Test File | Module Under Test | Est. Tests |
|-----------|------------------|------------|
| `tests/unit/api/test_train.py` | `nemo_rl.api.train` | 25 |
| `tests/unit/backends/test_backend_factory.py` | Backend factory | 20 |
| `tests/unit/backends/test_generation_backend.py` | Generation backends | 25 |
| `tests/unit/backends/test_training_backend.py` | Training backends | 25 |
| `tests/unit/config/test_base.py` | Base config | 15 |
| `tests/unit/config/test_cluster.py` | Cluster config | 10 |
| `tests/unit/config/test_defaults.py` | Default configs | 15 |
| `tests/unit/config/test_generation.py` | Generation config | 12 |
| `tests/unit/config/test_policy.py` | Policy config | 15 |
| `tests/unit/config/test_training.py` | Training config | 18 |
| `tests/unit/config/test_validation.py` | Config validation | 25 |
| `tests/unit/compat/test_algorithms.py` | Algorithm compat | 12 |
| `tests/unit/compat/test_config.py` | Config compat | 15 |
| `tests/unit/compat/test_deprecation.py` | Deprecation utils | 12 |
| `tests/unit/data/test_data_module.py` | Data module | 30 |
| `tests/unit/environments/test_callable_reward.py` | Callable rewards | 18 |
| `tests/unit/environments/test_environment_base.py` | Environment base | 25 |
| `tests/unit/environments/test_functional_reward.py` | Functional reward | 22 |
| `tests/unit/infra/test_checkpointing.py` | Checkpointing | 20 |
| `tests/unit/infra/test_logging.py` | Logging | 15 |
| `tests/unit/infra/test_resources.py` | Resource manager | 18 |
| `tests/unit/trainers/test_base_trainer.py` | Base trainer | 35 |
| `tests/unit/trainers/test_from_pretrained.py` | from_pretrained | 18 |
| `tests/unit/trainers/test_validation_runner.py` | Validation runner | 15 |
| `tests/unit/algorithms/test_dpo.py` | DPO algorithm | 10 |
| `tests/unit/algorithms/test_rollout.py` | Rollout engine | 25 |
| `tests/unit/utils/test_errors.py` | Error handling | 25 |
| `tests/unit/models/policy/test_policy_separation.py` | Policy separation | 18 |

**Execution Command:**
```bash
# Run all new unit tests
pytest tests/unit/api tests/unit/backends tests/unit/config \
       tests/unit/compat tests/unit/environments tests/unit/infra \
       tests/unit/trainers tests/unit/utils \
       -v --tb=short --cov=nemo_rl --cov-report=html

# Run specific test file
pytest tests/unit/api/test_train.py -v
```

### 1.4.2 Integration Test Execution

**Integration Test Files:**

| Test File | Scope | Resources |
|-----------|-------|-----------|
| `tests/integration/test_e2e_grpo.py` | Full GRPO pipeline | 1 node, 8 GPUs |
| `tests/integration/test_e2e_sft.py` | Full SFT pipeline | 1 node, 8 GPUs |
| `tests/integration/test_e2e_dpo.py` | Full DPO pipeline | 1 node, 8 GPUs |

**Execution Command:**
```bash
# Run integration tests (requires GPU)
pytest tests/integration/ -v --tb=long -x

# Run with specific GPU configuration
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 pytest tests/integration/test_e2e_grpo.py -v
```

### 1.4.3 E2E Test Scenarios

#### Scenario E2E-1: Basic GRPO Training (1 Node)

```yaml
test_name: basic_grpo_1n8g
description: Basic GRPO training on single node
resources:
  nodes: 1
  gpus_per_node: 8
  time_limit: 30m
config:
  model: Qwen/Qwen2.5-1.5B
  dataset: nvidia/OpenMathInstruct-2
  algorithm: grpo
  max_steps: 100
  batch_size: 32
  num_generations_per_prompt: 16
acceptance:
  - training_completes: true
  - final_loss < initial_loss
  - checkpoints_saved >= 1
  - no_oom_errors: true
```

#### Scenario E2E-2: Multi-Node GRPO Training

```yaml
test_name: multinode_grpo_2n8g
description: GRPO training on 2 nodes
resources:
  nodes: 2
  gpus_per_node: 8
  time_limit: 60m
config:
  model: Qwen/Qwen2.5-7B
  dataset: nvidia/OpenMathInstruct-2
  algorithm: grpo
  max_steps: 50
  backend: megatron
  tensor_parallel_size: 4
acceptance:
  - all_nodes_utilized: true
  - training_completes: true
  - throughput >= 80% of baseline
```

#### Scenario E2E-3: New API vs Legacy Equivalence

```yaml
test_name: api_equivalence
description: Verify new API produces same results as legacy
resources:
  nodes: 1
  gpus_per_node: 8
  time_limit: 45m
procedure:
  1. Run training with legacy API (100 steps)
  2. Run training with new API (100 steps, same seed)
  3. Compare metrics
acceptance:
  - loss_difference < 1%
  - reward_difference < 1%
  - gradient_norms_match: true
```

#### Scenario E2E-4: Checkpoint Resume

```yaml
test_name: checkpoint_resume
description: Verify training resumes correctly from checkpoint
resources:
  nodes: 1
  gpus_per_node: 8
  time_limit: 60m
procedure:
  1. Run training for 50 steps, save checkpoint
  2. Kill job
  3. Resume from checkpoint, run 50 more steps
  4. Compare with continuous 100-step run
acceptance:
  - training_resumes: true
  - final_metrics_match: within 2%
  - no_duplicate_data: true
```

#### Scenario E2E-5: Performance Benchmark

```yaml
test_name: performance_benchmark
description: Measure throughput and memory vs baseline
resources:
  nodes: 4
  gpus_per_node: 8
  time_limit: 120m
configs:
  - model: Qwen/Qwen2.5-1.5B, backend: dtensor
  - model: Qwen/Qwen2.5-7B, backend: megatron
  - model: Llama-3.1-8B, backend: megatron
acceptance:
  - throughput_regression < 5%
  - memory_regression < 5%
  - startup_time_regression < 10%
```

### 1.4.4 Regression Test Matrix

| Test Case | Old API | New API | Expected |
|-----------|---------|---------|----------|
| GRPO basic | `grpo_train()` | `nemo_rl.train(algorithm='grpo')` | Equivalent |
| SFT basic | `sft_train()` | `nemo_rl.train(algorithm='sft')` | Equivalent |
| DPO basic | `dpo_train()` | `nemo_rl.train(algorithm='dpo')` | Equivalent |
| YAML config | OmegaConf load | `Config.from_yaml()` | Equivalent |
| Environment | Ray actor class | Lambda function | Equivalent |
| Checkpoint | DCP format | DCP format | Compatible |

### 1.4.5 Test Execution Schedule

| Week | Focus | Tests | Environment |
|------|-------|-------|-------------|
| Week 1 | Unit tests (Part 1) | API, config, compat | Local CI |
| Week 2 | Unit tests (Part 2) | Backends, trainers, infra | Local CI |
| Week 3 | Integration tests | E2E pipelines | 1-node cluster |
| Week 4 | Multi-node E2E | Scale tests | 4-node cluster |
| Week 5 | Performance tests | Benchmarks | 4-node cluster |
| Week 6 | Regression tests | API equivalence | Mixed |
| Week 7 | Bug fixes | Failed tests | As needed |
| Week 8 | Final validation | All tests | Full suite |

---

## 1.5 Risk Assessment

### 1.5.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Backend abstraction breaks edge cases | Medium | High | Extensive integration testing |
| Config validation too strict | Medium | Medium | Allow escape hatches |
| Performance regression > 5% | Low | High | Continuous benchmarking |
| Checkpoint incompatibility | Low | Critical | Format versioning |
| Distributed training bugs | Medium | High | Multi-node stress tests |

### 1.5.2 Schedule Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| GPU cluster availability | Medium | Medium | Book resources early |
| Test failures delay timeline | High | Medium | 2-week buffer |
| Bug fixes take longer than expected | Medium | Medium | Parallel debugging |

---

# Part 2: Web Configuration Tool

## 2.1 Product Vision

### 2.1.1 Problem Statement

Currently, users must:
1. Understand complex YAML configuration files (300+ lines)
2. Manually write SLURM submission scripts
3. Know cluster-specific parameters (partitions, accounts, GRES)
4. Handle Ray cluster setup within SLURM
5. Debug configuration errors through trial and error

### 2.1.2 Solution Overview

The **NeMo RL Web Configurator** provides:
- Visual configuration builder with form-based inputs
- Real-time validation and error feedback
- SLURM script generation for various cluster types
- Configuration templates and presets
- Job submission and monitoring (optional)

### 2.1.3 Target Outcomes

| Metric | Current | Target |
|--------|---------|--------|
| Time to create valid config | 30-60 min | 5-10 min |
| Config errors at submission | ~40% | <5% |
| User training required | 4+ hours | 30 min |
| Documentation lookups needed | 10+ | 0-2 |

---

## 2.2 User Personas

### Persona 1: ML Researcher (Primary)

**Profile:**
- PhD student or research scientist
- Strong ML knowledge, moderate systems knowledge
- Uses NeMo RL for experiments
- Runs 10-50 jobs per week

**Goals:**
- Quickly iterate on training configurations
- Avoid infrastructure debugging
- Focus on algorithms and results

**Pain Points:**
- YAML syntax errors
- SLURM parameter confusion
- Ray cluster issues

### Persona 2: MLOps Engineer (Secondary)

**Profile:**
- Infrastructure engineer supporting research team
- Strong systems knowledge, moderate ML knowledge
- Manages cluster resources and job scheduling
- Supports 5-20 researchers

**Goals:**
- Standardize configurations across team
- Monitor resource utilization
- Debug job failures efficiently

**Pain Points:**
- Inconsistent configs across team
- Resource waste from misconfigurations
- Supporting diverse user needs

### Persona 3: New User (Tertiary)

**Profile:**
- Just started using NeMo RL
- Following tutorials or documentation
- Needs guided experience

**Goals:**
- Get first successful training run
- Understand available options
- Learn best practices

**Pain Points:**
- Overwhelming number of options
- Unclear defaults
- No feedback on choices

---

## 2.3 Feature Requirements

### 2.3.1 Core Features (P0)

#### F-1: Configuration Builder

| ID | Feature | Description | Priority |
|----|---------|-------------|----------|
| F-1.1 | Algorithm selection | Select GRPO/SFT/DPO with visual descriptions | P0 |
| F-1.2 | Model selector | Search/browse HuggingFace models or enter custom | P0 |
| F-1.3 | Dataset selector | Search HF datasets or specify local path | P0 |
| F-1.4 | Hyperparameter forms | Form inputs for learning rate, batch size, etc. | P0 |
| F-1.5 | Backend selector | Choose DTensor/Megatron with requirement hints | P0 |
| F-1.6 | Cluster config | Nodes, GPUs, memory, time limit | P0 |
| F-1.7 | Real-time validation | Validate config as user types | P0 |
| F-1.8 | Contextual help | Tooltips and explanations for each field | P0 |

**UI Mockup - Algorithm Selection:**
```
┌─────────────────────────────────────────────────────────────────┐
│  Select Training Algorithm                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐│
│  │      GRPO        │  │       SFT        │  │      DPO       ││
│  │  ─────────────   │  │  ─────────────   │  │  ───────────   ││
│  │  Reinforcement   │  │  Supervised      │  │  Preference    ││
│  │  learning from   │  │  fine-tuning     │  │  optimization  ││
│  │  rewards         │  │  on examples     │  │  from pairs    ││
│  │                  │  │                  │  │                ││
│  │  [Selected ✓]    │  │  [Select]        │  │  [Select]      ││
│  └──────────────────┘  └──────────────────┘  └────────────────┘│
│                                                                  │
│  GRPO is best for: Improving model behavior through reward       │
│  signals (e.g., math problem solving, code generation)           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### F-2: SLURM Script Generation

| ID | Feature | Description | Priority |
|----|---------|-------------|----------|
| F-2.1 | Script preview | Live preview of generated SLURM script | P0 |
| F-2.2 | Cluster presets | Pre-configured settings for known clusters | P0 |
| F-2.3 | Custom parameters | Add arbitrary SLURM directives | P0 |
| F-2.4 | Download script | Download ready-to-run .sh file | P0 |
| F-2.5 | Copy to clipboard | One-click copy of script | P0 |
| F-2.6 | Environment setup | Container image, mounts, env vars | P0 |

**Output Example:**
```bash
#!/bin/bash
#SBATCH --job-name=nemo-rl-grpo-qwen
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=4:00:00
#SBATCH --partition=batch
#SBATCH --account=myaccount
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Auto-generated by NeMo RL Web Configurator
# Generated: 2026-01-15 10:30:00 UTC
# Config ID: abc123

export CONTAINER="nvcr.io/nvidia/nemo-rl:24.12"
export MOUNTS="/data:/data,/home/$USER:/home/$USER"
export GPUS_PER_NODE=8

# Training command
export COMMAND="python -m nemo_rl.train \
    --model Qwen/Qwen2.5-1.5B \
    --dataset nvidia/OpenMathInstruct-2 \
    --algorithm grpo \
    --max_steps 1000 \
    --batch_size 32 \
    --learning_rate 1e-6"

# Launch Ray cluster and run training
bash ray.sub
```

#### F-3: Configuration Validation

| ID | Feature | Description | Priority |
|----|---------|-------------|----------|
| F-3.1 | Type validation | Check parameter types match schema | P0 |
| F-3.2 | Range validation | Check values within valid ranges | P0 |
| F-3.3 | Resource validation | Check GPU/memory requirements feasible | P0 |
| F-3.4 | Compatibility checks | Check backend + model + config compatibility | P0 |
| F-3.5 | Warning indicators | Non-blocking warnings for suboptimal configs | P0 |
| F-3.6 | Error messages | Clear, actionable error descriptions | P0 |

**Validation Examples:**
```
✓ Model: Qwen/Qwen2.5-1.5B (verified on HuggingFace)
✓ Batch size: 32 (valid for selected GPU memory)
⚠ Learning rate: 1e-4 (high for fine-tuning, typical: 1e-6 to 5e-6)
✗ Tensor parallel size: 3 (must divide evenly into GPUs per node: 8)
```

### 2.3.2 Enhanced Features (P1)

#### F-4: Templates and Presets

| ID | Feature | Description | Priority |
|----|---------|-------------|----------|
| F-4.1 | Built-in templates | Common configurations (math, code, chat) | P1 |
| F-4.2 | Recipe library | Curated configs from documentation | P1 |
| F-4.3 | Save custom templates | User can save their configs | P1 |
| F-4.4 | Share templates | Generate shareable links | P1 |
| F-4.5 | Import existing config | Load YAML/JSON configs | P1 |

#### F-5: Advanced Configuration

| ID | Feature | Description | Priority |
|----|---------|-------------|----------|
| F-5.1 | Raw YAML editor | Switch to YAML mode for power users | P1 |
| F-5.2 | Config diff | Compare two configurations | P1 |
| F-5.3 | Multi-experiment | Generate configs for sweeps | P1 |
| F-5.4 | Version history | Track config changes over time | P1 |

#### F-6: Reward Function Builder

| ID | Feature | Description | Priority |
|----|---------|-------------|----------|
| F-6.1 | Code editor | Write reward function in browser | P1 |
| F-6.2 | Test harness | Test reward function with samples | P1 |
| F-6.3 | Template rewards | Common reward patterns (exact match, regex) | P1 |
| F-6.4 | Import function | Upload Python file with reward | P1 |

### 2.3.3 Optional Features (P2)

#### F-7: Job Management (Optional)

| ID | Feature | Description | Priority |
|----|---------|-------------|----------|
| F-7.1 | Submit job | Direct SLURM submission (if cluster accessible) | P2 |
| F-7.2 | Job status | Monitor running jobs | P2 |
| F-7.3 | Log viewer | Stream job logs | P2 |
| F-7.4 | Cancel job | Cancel running jobs | P2 |
| F-7.5 | Resource dashboard | Cluster utilization view | P2 |

#### F-8: Collaboration

| ID | Feature | Description | Priority |
|----|---------|-------------|----------|
| F-8.1 | User accounts | Save configs to account | P2 |
| F-8.2 | Team workspaces | Share configs within team | P2 |
| F-8.3 | Comments | Annotate configs | P2 |
| F-8.4 | Export/Import | Bulk config operations | P2 |

---

## 2.4 Technical Architecture

### 2.4.1 System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Browser                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    React Frontend                          │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │  │
│  │  │ Config   │  │ Preview  │  │ Template │  │ Reward   │  │  │
│  │  │ Builder  │  │ Panel    │  │ Library  │  │ Editor   │  │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ REST API / WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Backend Services                           │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │   FastAPI        │  │  Config          │  │  HuggingFace  │ │
│  │   Application    │  │  Validator       │  │  API Client   │ │
│  └────────┬─────────┘  └────────┬─────────┘  └───────┬───────┘ │
│           │                     │                     │         │
│  ┌────────┴─────────────────────┴─────────────────────┴───────┐ │
│  │                    Script Generator                         │ │
│  │  - SLURM template engine                                   │ │
│  │  - Ray cluster configuration                               │ │
│  │  - Python training command builder                         │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ (Optional)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    External Services                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  HuggingFace │  │    SLURM     │  │   Database   │          │
│  │     Hub      │  │   Cluster    │  │  (Optional)  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### 2.4.2 Technology Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| Frontend | React 18 + TypeScript | Modern, type-safe, rich ecosystem |
| UI Framework | Tailwind CSS + shadcn/ui | Fast development, consistent design |
| State Management | Zustand | Lightweight, TypeScript-friendly |
| Code Editor | Monaco Editor | VSCode-quality editing |
| Backend | FastAPI (Python) | Fast, async, automatic OpenAPI |
| Validation | Pydantic | Same models as NeMo RL configs |
| Template Engine | Jinja2 | Flexible script generation |
| Database (optional) | SQLite / PostgreSQL | User configs persistence |

### 2.4.3 API Design

#### Configuration API

```
POST /api/v1/config/validate
Request:
{
  "algorithm": "grpo",
  "model": "Qwen/Qwen2.5-1.5B",
  "dataset": "nvidia/OpenMathInstruct-2",
  "hyperparameters": {
    "learning_rate": 1e-6,
    "batch_size": 32,
    "max_steps": 1000
  },
  "cluster": {
    "nodes": 2,
    "gpus_per_node": 8,
    "time_limit": "4:00:00"
  }
}

Response:
{
  "valid": true,
  "errors": [],
  "warnings": [
    {
      "field": "hyperparameters.learning_rate",
      "message": "Learning rate 1e-6 is on the lower end. Typical range: 1e-6 to 5e-6",
      "severity": "info"
    }
  ],
  "resource_estimate": {
    "memory_per_gpu": "24GB",
    "recommended_gpus": 8,
    "estimated_time": "2-3 hours"
  }
}
```

#### Script Generation API

```
POST /api/v1/script/generate
Request:
{
  "config": { ... },
  "cluster_preset": "dgx-cluster",
  "output_format": "slurm"
}

Response:
{
  "script": "#!/bin/bash\n#SBATCH ...",
  "files": {
    "ray.sub": "...",
    "config.yaml": "..."
  },
  "instructions": [
    "1. Save script as train.sh",
    "2. Run: sbatch train.sh"
  ]
}
```

#### Model Search API

```
GET /api/v1/models/search?q=qwen&size=small
Response:
{
  "results": [
    {
      "id": "Qwen/Qwen2.5-1.5B",
      "name": "Qwen2.5-1.5B",
      "size_gb": 3.0,
      "parameters": "1.5B",
      "recommended_config": {
        "min_gpus": 1,
        "tensor_parallel": false
      }
    }
  ]
}
```

### 2.4.4 Data Models

```python
# Backend Pydantic models
from pydantic import BaseModel, Field, validator
from typing import Optional, Literal
from enum import Enum

class Algorithm(str, Enum):
    GRPO = "grpo"
    SFT = "sft"
    DPO = "dpo"

class Backend(str, Enum):
    DTENSOR = "dtensor"
    MEGATRON = "megatron"

class ClusterConfig(BaseModel):
    nodes: int = Field(ge=1, le=256, default=1)
    gpus_per_node: int = Field(ge=1, le=8, default=8)
    time_limit: str = Field(regex=r"^\d+:\d{2}:\d{2}$", default="4:00:00")
    partition: Optional[str] = None
    account: Optional[str] = None
    
    @validator("gpus_per_node")
    def validate_gpu_count(cls, v):
        if v not in [1, 2, 4, 8]:
            raise ValueError("GPUs per node must be 1, 2, 4, or 8")
        return v

class HyperparametersConfig(BaseModel):
    learning_rate: float = Field(ge=1e-8, le=1e-2, default=1e-6)
    batch_size: int = Field(ge=1, le=1024, default=32)
    max_steps: int = Field(ge=1, le=1000000, default=1000)
    num_generations_per_prompt: int = Field(ge=1, le=64, default=16)
    
class TrainingConfig(BaseModel):
    algorithm: Algorithm
    model: str
    dataset: str
    backend: Backend = Backend.DTENSOR
    hyperparameters: HyperparametersConfig
    cluster: ClusterConfig
    
    class Config:
        use_enum_values = True
```

### 2.4.5 Frontend Component Architecture

```
src/
├── components/
│   ├── config-builder/
│   │   ├── AlgorithmSelector.tsx
│   │   ├── ModelSelector.tsx
│   │   ├── DatasetSelector.tsx
│   │   ├── HyperparameterForm.tsx
│   │   ├── BackendSelector.tsx
│   │   ├── ClusterConfig.tsx
│   │   └── index.tsx
│   ├── preview/
│   │   ├── ScriptPreview.tsx
│   │   ├── ConfigPreview.tsx
│   │   └── ValidationPanel.tsx
│   ├── templates/
│   │   ├── TemplateLibrary.tsx
│   │   ├── TemplateCard.tsx
│   │   └── ImportExport.tsx
│   ├── reward-editor/
│   │   ├── CodeEditor.tsx
│   │   ├── TestHarness.tsx
│   │   └── TemplateRewards.tsx
│   └── common/
│       ├── Tooltip.tsx
│       ├── ValidationBadge.tsx
│       └── CopyButton.tsx
├── hooks/
│   ├── useConfig.ts
│   ├── useValidation.ts
│   ├── useModelSearch.ts
│   └── useScriptGeneration.ts
├── store/
│   ├── configStore.ts
│   └── uiStore.ts
├── api/
│   └── client.ts
├── types/
│   └── config.ts
└── App.tsx
```

---

## 2.5 UI/UX Requirements

### 2.5.1 Design Principles

1. **Progressive Disclosure**: Show basic options first, advanced options behind expandable sections
2. **Instant Feedback**: Validate and update preview as user types
3. **Guided Experience**: Wizards for new users, quick forms for experts
4. **Error Prevention**: Disable invalid combinations, show warnings early
5. **Accessibility**: WCAG 2.1 AA compliance

### 2.5.2 Page Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Logo   [Templates]  [Docs]  [GitHub]                    [Dark/Light]   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────┐  ┌──────────────────────────────────┐ │
│  │     Configuration Panel      │  │        Preview Panel              │ │
│  │                              │  │                                   │ │
│  │  Algorithm: [GRPO ▼]         │  │  ┌─────────────────────────────┐ │ │
│  │                              │  │  │  # SLURM Script             │ │ │
│  │  Model: [Qwen/Qwen2.5-1.5B] │  │  │  #!/bin/bash                │ │ │
│  │         [Search models...]   │  │  │  #SBATCH --nodes=2          │ │ │
│  │                              │  │  │  #SBATCH --gpus-per-node=8  │ │ │
│  │  Dataset: [nvidia/OpenMath] │  │  │  ...                        │ │ │
│  │                              │  │  └─────────────────────────────┘ │ │
│  │  ▼ Hyperparameters           │  │                                   │ │
│  │    Learning Rate: [1e-6]     │  │  [Copy] [Download] [Run ▶]       │ │
│  │    Batch Size: [32]          │  │                                   │ │
│  │    Max Steps: [1000]         │  │  ──────────────────────────────── │ │
│  │                              │  │                                   │ │
│  │  ▼ Cluster Configuration     │  │  Validation                       │ │
│  │    Nodes: [2]                │  │  ✓ Model verified                 │ │
│  │    GPUs per Node: [8]        │  │  ✓ Config valid                   │ │
│  │    Time Limit: [4:00:00]     │  │  ⚠ High learning rate             │ │
│  │                              │  │                                   │ │
│  │  ► Advanced Options          │  │  Resource Estimate                │ │
│  │                              │  │  Memory: 24GB/GPU                 │ │
│  │                              │  │  Est. Time: 2-3 hours             │ │
│  └──────────────────────────────┘  └──────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.5.3 User Flows

#### Flow 1: New User - First Configuration

```
1. Land on homepage
   └─→ See "Get Started" button prominently
   
2. Click "Get Started"
   └─→ Wizard starts: "What do you want to train?"
   
3. Select use case (e.g., "Math Problem Solving")
   └─→ Algorithm pre-selected (GRPO)
   └─→ Recommended model shown
   
4. Choose model or accept default
   └─→ Model validated against HuggingFace
   
5. Select dataset or use default
   └─→ Dataset info displayed
   
6. Review cluster config (defaults for use case)
   └─→ Warnings if resources seem insufficient
   
7. Preview generated script
   └─→ Real-time validation results shown
   
8. Download script
   └─→ Instructions shown for submission
```

#### Flow 2: Expert User - Quick Configuration

```
1. Land on homepage (returning user)
   └─→ Last config shown or blank form
   
2. Fill form directly
   └─→ Real-time validation as typing
   
3. Expand "Advanced Options" if needed
   └─→ Additional parameters available
   
4. Copy script to clipboard
   └─→ Done in < 2 minutes
```

### 2.5.4 Responsive Design Requirements

| Breakpoint | Layout | Notes |
|------------|--------|-------|
| Desktop (>1200px) | Side-by-side panels | Full experience |
| Tablet (768-1200px) | Stacked panels | Config above preview |
| Mobile (<768px) | Single column with tabs | Limited but functional |

---

## 2.6 Implementation Phases

### Phase 1: Foundation (Month 3-4)

**Week 9-10: Project Setup**
- Initialize React + TypeScript project
- Set up FastAPI backend
- Configure CI/CD pipeline
- Design system setup (Tailwind + shadcn)

**Week 11-12: Core Configuration Form**
- Algorithm selector component
- Model search/selector component
- Dataset selector component
- Basic hyperparameter form

**Week 13-14: Validation System**
- Backend validation API
- Frontend validation hooks
- Error/warning display components
- Integration with NeMo RL config schemas

**Week 15-16: Script Generation**
- SLURM template engine
- Ray cluster config generation
- Preview panel component
- Download/copy functionality

**Deliverables:**
- MVP with basic config → script generation
- Validation for common errors
- 3 cluster presets

### Phase 2: Enhanced Features (Month 5)

**Week 17-18: Template System**
- Template data model and storage
- Template library UI
- Save/load custom templates
- Import existing YAML configs

**Week 19-20: Advanced Configuration**
- Backend selector with compatibility hints
- Advanced hyperparameters section
- Raw YAML editor mode
- Config diff viewer

**Deliverables:**
- 10+ built-in templates
- Full algorithm coverage
- YAML import/export

### Phase 3: Reward Builder & Polish (Month 6)

**Week 21-22: Reward Function Builder**
- Monaco code editor integration
- Reward function templates
- Test harness with sample data
- Syntax validation

**Week 23-24: Polish & Documentation**
- Comprehensive help system
- Tutorial walkthrough
- Performance optimization
- Accessibility audit
- User testing and feedback

**Deliverables:**
- Production-ready application
- User documentation
- API documentation

---

# Success Metrics

## Validation Phase Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Unit test pass rate | 100% | CI pipeline |
| Integration test pass rate | 100% | CI pipeline |
| E2E test pass rate | 95%+ | Manual runs |
| Performance regression | <5% | Benchmark suite |
| Code coverage | 75%+ | Coverage reports |
| Critical bugs | 0 | Bug tracking |

## Web Tool Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Time to first valid config | <10 min | User testing |
| Config validation accuracy | >98% | Error rate tracking |
| User satisfaction | 4.0/5.0 | Survey |
| Page load time | <3 seconds | Performance monitoring |
| Monthly active users (Month 6) | 100+ | Analytics |

---

# Timeline and Milestones

```
Month 1-2: Validation Phase
├── Week 1-2: Unit tests (API, config, compat)
├── Week 3-4: Unit tests (backends, trainers, infra)
├── Week 5-6: Integration tests
├── Week 7-8: E2E tests + performance tests

Month 3-4: Web Tool Foundation
├── Week 9-10: Project setup + design system
├── Week 11-12: Core config form
├── Week 13-14: Validation system
├── Week 15-16: Script generation

Month 5: Web Tool Enhancement
├── Week 17-18: Template system
├── Week 19-20: Advanced config + YAML editor

Month 6: Web Tool Polish
├── Week 21-22: Reward builder
├── Week 23-24: Polish, docs, launch
```

## Milestone Definitions

| Milestone | Date | Deliverables | Success Criteria |
|-----------|------|--------------|------------------|
| M1: Unit Tests Complete | Week 4 | All unit tests pass | 100% pass rate, 75% coverage |
| M2: Integration Tests Complete | Week 6 | Integration tests pass | 100% pass rate |
| M3: E2E Validation Complete | Week 8 | All E2E scenarios pass | 95%+ pass rate, <5% perf regression |
| M4: Web MVP | Week 16 | Basic config → script | User can generate valid SLURM script |
| M5: Web Feature Complete | Week 20 | Templates, YAML editor | Full feature set working |
| M6: Launch Ready | Week 24 | Documentation, polish | Production deployment ready |

---

# Appendix

## A.1 Test Command Reference

```bash
# Run all unit tests
pytest tests/unit/ -v --cov=nemo_rl

# Run specific test module
pytest tests/unit/api/test_train.py -v

# Run integration tests
pytest tests/integration/ -v -x --tb=long

# Run with GPU
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 pytest tests/integration/ -v

# Run performance benchmarks
python tests/benchmarks/run_benchmarks.py --output results.json

# Run E2E on SLURM
sbatch tests/test_suites/llm/grpo_e2e_1n8g.sh
```

## A.2 Cluster Presets

### NVIDIA DGX Cloud

```yaml
preset_name: dgx-cloud
partition: batch
gpus_per_node: 8
container: nvcr.io/nvidia/nemo-rl:24.12
gres_format: gpu:8
```

### NVIDIA BCM Cluster

```yaml
preset_name: nvidia-bcm
partition: luna
gpus_per_node: 8
container: nvcr.io/nvidia/nemo-rl:24.12
gres_format: gpu:h100:8
```

### Generic SLURM

```yaml
preset_name: generic
partition: null  # User must specify
gpus_per_node: 8
container: null  # User must specify
gres_format: gpu:8
```

## A.3 Configuration Schema (Full)

```yaml
# Complete configuration schema for reference
algorithm:
  type: enum
  values: [grpo, sft, dpo]
  required: true

model:
  type: string
  description: HuggingFace model ID or local path
  required: true
  validation:
    - pattern: "^[a-zA-Z0-9-_./]+$"
    - huggingface_exists: true (optional)

dataset:
  type: string
  description: HuggingFace dataset ID, local path, or list
  required: true

backend:
  type: enum
  values: [dtensor, megatron]
  default: dtensor
  constraints:
    - dtensor: requires FSDP2 support
    - megatron: requires tensor parallel > 1 for large models

hyperparameters:
  learning_rate:
    type: float
    range: [1e-8, 1e-2]
    default: 1e-6
  batch_size:
    type: int
    range: [1, 1024]
    default: 32
  max_steps:
    type: int
    range: [1, 1000000]
    default: 1000
  num_generations_per_prompt:
    type: int
    range: [1, 64]
    default: 16
    only_for: grpo

cluster:
  nodes:
    type: int
    range: [1, 256]
    default: 1
  gpus_per_node:
    type: int
    values: [1, 2, 4, 8]
    default: 8
  time_limit:
    type: string
    pattern: "^\\d+:\\d{2}:\\d{2}$"
    default: "4:00:00"
  partition:
    type: string
    required: false
  account:
    type: string
    required: false

checkpointing:
  enabled:
    type: bool
    default: true
  save_every:
    type: int
    default: 100
  output_dir:
    type: string
    default: "./outputs"
```

## A.4 Error Message Guidelines

Errors should follow this format:
```
[Error Code] Brief Description

What went wrong:
  Detailed explanation

How to fix:
  1. Step one
  2. Step two

Related documentation:
  https://docs.nemo-rl.nvidia.com/...
```

Example:
```
[CONFIG-001] Invalid tensor parallel size

What went wrong:
  tensor_parallel_size=3 does not evenly divide gpus_per_node=8

How to fix:
  1. Change tensor_parallel_size to 1, 2, 4, or 8
  2. Or change gpus_per_node to a multiple of 3

Related documentation:
  https://docs.nemo-rl.nvidia.com/parallelism
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-15 | Engineering | Initial draft |

---

## Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Tech Lead | | | |
| Product Manager | | | |
| QA Lead | | | |
| Engineering Manager | | | |
