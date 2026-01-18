"""
NeMo RL Web Configurator - FastAPI Backend

This backend provides APIs for:
- Configuration validation
- SLURM script generation
- Model search via HuggingFace Hub
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
from enum import Enum
from datetime import datetime, timezone
import re

# =============================================================================
# Enums and Models
# =============================================================================

class Algorithm(str, Enum):
    GRPO = "grpo"
    SFT = "sft"
    DPO = "dpo"


class Backend(str, Enum):
    DTENSOR = "dtensor"
    MEGATRON = "megatron"


class ClusterConfig(BaseModel):
    """Cluster configuration for training jobs."""
    nodes: int = Field(ge=1, le=256, default=1)
    gpus_per_node: int = Field(ge=1, le=8, default=8)
    time_limit: str = Field(default="4:00:00", pattern=r"^\d+:\d{2}:\d{2}$")
    partition: Optional[str] = None
    account: Optional[str] = None

    @field_validator("gpus_per_node")
    @classmethod
    def validate_gpu_count(cls, v: int) -> int:
        if v not in [1, 2, 4, 8]:
            raise ValueError("GPUs per node must be 1, 2, 4, or 8")
        return v


class HyperparametersConfig(BaseModel):
    """Hyperparameters for training."""
    learning_rate: float = Field(ge=1e-8, le=1e-2, default=1e-6)
    batch_size: int = Field(ge=1, le=1024, default=32)
    max_steps: int = Field(ge=1, le=1000000, default=1000)
    num_generations_per_prompt: int = Field(ge=1, le=64, default=16)
    tensor_parallel_size: int = Field(ge=1, le=8, default=1)


class TrainingConfig(BaseModel):
    """Complete training configuration."""
    algorithm: Algorithm
    model: str
    dataset: str
    backend: Backend = Backend.DTENSOR
    hyperparameters: HyperparametersConfig
    cluster: ClusterConfig
    
    model_config = {"use_enum_values": True}


class ValidationError(BaseModel):
    """Validation error or warning."""
    field: str
    message: str
    severity: Literal["error", "warning", "info"]


class ResourceEstimate(BaseModel):
    """Estimated resource requirements."""
    memory_per_gpu: str
    recommended_gpus: int
    estimated_time: str


class ValidationResult(BaseModel):
    """Result of configuration validation."""
    valid: bool
    errors: list[ValidationError] = []
    warnings: list[ValidationError] = []
    resource_estimate: Optional[ResourceEstimate] = None


class ScriptGenerationRequest(BaseModel):
    """Request for script generation."""
    config: TrainingConfig
    cluster_preset: Optional[str] = None
    output_format: Literal["slurm"] = "slurm"


class ScriptGenerationResult(BaseModel):
    """Generated SLURM script and related files."""
    script: str
    files: dict[str, str] = {}
    instructions: list[str] = []


class ModelInfo(BaseModel):
    """HuggingFace model information."""
    id: str
    name: str
    size_gb: float
    parameters: str
    recommended_config: Optional[dict] = None


class ModelSearchResult(BaseModel):
    """Model search results."""
    results: list[ModelInfo]


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="NeMo RL Web Configurator API",
    description="Backend API for the NeMo RL Web Configuration Tool",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API Routes
# =============================================================================

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


# =============================================================================
# Validation Helpers
# =============================================================================

# Model size estimates for memory calculation (in billions of parameters)
MODEL_SIZE_MAP = {
    "1.5B": 1.5, "1.5b": 1.5, "1B": 1.0, "1b": 1.0,
    "3B": 3.0, "3b": 3.0,
    "7B": 7.0, "7b": 7.0, "8B": 8.0, "8b": 8.0,
    "13B": 13.0, "13b": 13.0, "14B": 14.0, "14b": 14.0,
    "30B": 30.0, "30b": 30.0, "32B": 32.0, "32b": 32.0, "34B": 34.0,
    "70B": 70.0, "70b": 70.0, "72B": 72.0,
}

def estimate_model_size(model_name: str) -> float:
    """Estimate model size in billions of parameters from model name."""
    for size_str, size_val in MODEL_SIZE_MAP.items():
        if size_str in model_name:
            return size_val
    return 7.0  # Default assumption for unknown models

def estimate_memory_per_gpu(model_size_b: float, backend: Backend, total_gpus: int) -> str:
    """Estimate memory per GPU in GB based on model size and parallelism."""
    # Rough estimate: ~2 bytes per param for mixed precision + optimizer states
    # Memory per param in training = ~16-20 bytes (weights + gradients + optimizer)
    base_memory_gb = model_size_b * 18 / total_gpus  # Distribute across GPUs
    
    # Add overhead for activations, KV cache, etc.
    total_memory_gb = base_memory_gb * 1.5
    
    if total_memory_gb < 24:
        return "24GB"
    elif total_memory_gb < 40:
        return "40GB"
    elif total_memory_gb < 80:
        return "80GB"
    else:
        return f"{int(total_memory_gb)}GB"

def estimate_training_time(model_size_b: float, max_steps: int, batch_size: int, 
                          total_gpus: int, algorithm: Algorithm) -> str:
    """Estimate training time based on configuration."""
    # Rough estimates based on typical throughput
    # Base throughput: ~1000 tokens/sec/GPU for training
    steps_per_hour = 100 * total_gpus / model_size_b  # Scale by model size
    
    # GRPO is slower due to generation
    if algorithm == Algorithm.GRPO:
        steps_per_hour *= 0.5
    
    estimated_hours = max_steps / max(steps_per_hour, 1)
    
    if estimated_hours < 1:
        return f"{int(estimated_hours * 60)} minutes"
    elif estimated_hours < 4:
        return f"1-{int(estimated_hours) + 1} hours"
    elif estimated_hours < 12:
        return f"{int(estimated_hours)}-{int(estimated_hours) + 4} hours"
    else:
        return f"{int(estimated_hours / 24)}-{int(estimated_hours / 24) + 1} days"

def calculate_recommended_gpus(model_size_b: float, batch_size: int) -> int:
    """Calculate recommended number of GPUs based on model size."""
    if model_size_b <= 1.5:
        return 1
    elif model_size_b <= 7:
        return 2
    elif model_size_b <= 13:
        return 4
    elif model_size_b <= 34:
        return 8
    else:
        return 16  # Multi-node


@app.post("/api/v1/config/validate", response_model=ValidationResult)
async def validate_config(config: TrainingConfig):
    """
    Validate a training configuration.
    
    Performs comprehensive validation including:
    - Type validation (field types and formats)
    - Range validation (min/max values)
    - Compatibility checks (backend + model + cluster)
    - Resource estimation (memory, GPUs, time)
    
    Returns validation errors, warnings, and resource estimates.
    """
    errors: list[ValidationError] = []
    warnings: list[ValidationError] = []

    # ==========================================================================
    # Type Validation
    # ==========================================================================
    
    # Validate model name format
    if not config.model:
        errors.append(ValidationError(
            field="model",
            message="Model name is required",
            severity="error"
        ))
    elif not re.match(r"^[\w\-./]+$", config.model):
        errors.append(ValidationError(
            field="model",
            message="Model name contains invalid characters. Use alphanumeric, hyphens, dots, and slashes only.",
            severity="error"
        ))

    # Validate dataset format
    if not config.dataset:
        errors.append(ValidationError(
            field="dataset",
            message="Dataset is required",
            severity="error"
        ))
    elif not re.match(r"^[\w\-./]+$", config.dataset):
        errors.append(ValidationError(
            field="dataset",
            message="Dataset name contains invalid characters",
            severity="error"
        ))

    # Validate time limit format
    if not re.match(r"^\d+:\d{2}:\d{2}$", config.cluster.time_limit):
        errors.append(ValidationError(
            field="cluster.time_limit",
            message="Time limit must be in HH:MM:SS format (e.g., 4:00:00)",
            severity="error"
        ))

    # ==========================================================================
    # Range Validation
    # ==========================================================================
    
    # Learning rate validation
    if config.hyperparameters.learning_rate < 1e-8:
        errors.append(ValidationError(
            field="hyperparameters.learning_rate",
            message=f"Learning rate {config.hyperparameters.learning_rate} is below minimum (1e-8)",
            severity="error"
        ))
    elif config.hyperparameters.learning_rate > 1e-2:
        errors.append(ValidationError(
            field="hyperparameters.learning_rate",
            message=f"Learning rate {config.hyperparameters.learning_rate} exceeds maximum (1e-2)",
            severity="error"
        ))
    elif config.hyperparameters.learning_rate > 1e-4:
        warnings.append(ValidationError(
            field="hyperparameters.learning_rate",
            message=f"Learning rate {config.hyperparameters.learning_rate:.0e} is high for fine-tuning. Typical range: 1e-6 to 5e-6",
            severity="warning"
        ))

    # Batch size validation
    if config.hyperparameters.batch_size < 1:
        errors.append(ValidationError(
            field="hyperparameters.batch_size",
            message="Batch size must be at least 1",
            severity="error"
        ))
    elif config.hyperparameters.batch_size > 1024:
        errors.append(ValidationError(
            field="hyperparameters.batch_size",
            message="Batch size exceeds maximum (1024)",
            severity="error"
        ))

    # Max steps validation
    if config.hyperparameters.max_steps < 1:
        errors.append(ValidationError(
            field="hyperparameters.max_steps",
            message="Max steps must be at least 1",
            severity="error"
        ))
    elif config.hyperparameters.max_steps > 1000000:
        errors.append(ValidationError(
            field="hyperparameters.max_steps",
            message="Max steps exceeds maximum (1,000,000)",
            severity="error"
        ))

    # Nodes validation
    if config.cluster.nodes < 1:
        errors.append(ValidationError(
            field="cluster.nodes",
            message="Number of nodes must be at least 1",
            severity="error"
        ))
    elif config.cluster.nodes > 256:
        errors.append(ValidationError(
            field="cluster.nodes",
            message="Number of nodes exceeds maximum (256)",
            severity="error"
        ))

    # ==========================================================================
    # Compatibility Checks
    # ==========================================================================
    
    model_size = estimate_model_size(config.model)
    total_gpus = config.cluster.nodes * config.cluster.gpus_per_node

    # Backend compatibility
    if config.backend == Backend.MEGATRON:
        if config.cluster.gpus_per_node < 2:
            warnings.append(ValidationError(
                field="backend",
                message="Megatron backend is typically used with tensor parallelism (2+ GPUs per node)",
                severity="warning"
            ))
        if model_size <= 1.5:
            warnings.append(ValidationError(
                field="backend",
                message="Small models (<=1.5B) typically don't benefit from Megatron. Consider DTensor.",
                severity="info"
            ))
    
    if config.backend == Backend.DTENSOR:
        if model_size > 13:
            warnings.append(ValidationError(
                field="backend",
                message=f"Large models (>{model_size:.0f}B) may perform better with Megatron backend for tensor parallelism",
                severity="info"
            ))

    # Model size vs GPU configuration
    recommended_gpus = calculate_recommended_gpus(model_size, config.hyperparameters.batch_size)
    if total_gpus < recommended_gpus:
        warnings.append(ValidationError(
            field="cluster",
            message=f"Model size (~{model_size:.1f}B params) may require at least {recommended_gpus} GPUs. You have {total_gpus} configured.",
            severity="warning"
        ))

    # GRPO-specific validations
    if config.algorithm == Algorithm.GRPO:
        if config.hyperparameters.num_generations_per_prompt < 4:
            warnings.append(ValidationError(
                field="hyperparameters.num_generations_per_prompt",
                message="GRPO typically works better with at least 4 generations per prompt for stable reward estimation",
                severity="info"
            ))
        if config.hyperparameters.num_generations_per_prompt > 32:
            warnings.append(ValidationError(
                field="hyperparameters.num_generations_per_prompt",
                message="Very high generations per prompt (>32) may slow training significantly",
                severity="warning"
            ))

    # DPO-specific validations
    if config.algorithm == Algorithm.DPO:
        if config.hyperparameters.batch_size % 2 != 0:
            warnings.append(ValidationError(
                field="hyperparameters.batch_size",
                message="DPO works with preference pairs; batch size should ideally be even",
                severity="info"
            ))

    # Tensor parallelism validation - CRITICAL: must divide evenly into GPUs
    tensor_parallel_size = config.hyperparameters.tensor_parallel_size
    if tensor_parallel_size > 1:
        if config.cluster.gpus_per_node % tensor_parallel_size != 0:
            errors.append(ValidationError(
                field="hyperparameters.tensor_parallel_size",
                message=f"Tensor parallel size ({tensor_parallel_size}) must divide evenly into GPUs per node ({config.cluster.gpus_per_node}). Valid values: 1, 2, 4, or 8.",
                severity="error"
            ))
        if config.backend == Backend.DTENSOR and tensor_parallel_size > 1:
            warnings.append(ValidationError(
                field="hyperparameters.tensor_parallel_size",
                message="Tensor parallelism is primarily used with Megatron backend. DTensor uses FSDP for sharding instead.",
                severity="warning"
            ))
    
    # Check tensor parallel vs model size
    if model_size > 13 and tensor_parallel_size == 1 and config.backend == Backend.MEGATRON:
        warnings.append(ValidationError(
            field="hyperparameters.tensor_parallel_size",
            message=f"Large models (~{model_size:.0f}B) typically benefit from tensor parallelism (tensor_parallel_size > 1)",
            severity="info"
        ))

    # Multi-node validation
    if config.cluster.nodes > 1:
        if not config.cluster.partition:
            warnings.append(ValidationError(
                field="cluster.partition",
                message="Multi-node jobs typically require a partition specification",
                severity="info"
            ))

    # ==========================================================================
    # Resource Estimation
    # ==========================================================================
    
    memory_estimate = estimate_memory_per_gpu(model_size, config.backend, total_gpus)
    time_estimate = estimate_training_time(
        model_size, 
        config.hyperparameters.max_steps,
        config.hyperparameters.batch_size,
        total_gpus,
        config.algorithm
    )
    
    resource_estimate = ResourceEstimate(
        memory_per_gpu=memory_estimate,
        recommended_gpus=recommended_gpus,
        estimated_time=time_estimate
    )

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        resource_estimate=resource_estimate
    )


@app.post("/api/v1/script/generate", response_model=ScriptGenerationResult)
async def generate_script(request: ScriptGenerationRequest):
    """
    Generate a SLURM script and related files from the configuration.
    
    Returns:
    - SLURM submission script with all #SBATCH directives
    - ray.sub for multi-node Ray cluster management
    - config.yaml with NeMo RL training configuration
    - Instructions for submission
    
    Supports cluster presets: dgx-cloud, bcm, generic
    """
    config = request.config
    cluster_preset = request.cluster_preset
    
    # Apply cluster preset defaults if specified
    partition = config.cluster.partition
    account = config.cluster.account
    container = "nvcr.io/nvidia/nemo-rl:24.12"
    
    if cluster_preset == "dgx-cloud":
        partition = partition or "batch"
        container = "nvcr.io/nvidia/nemo-rl:24.12-dgx"
    elif cluster_preset == "bcm":
        partition = partition or "luna"
        container = "nvcr.io/nvidia/nemo-rl:24.12"
    # generic preset uses whatever is configured
    
    # Get tensor parallel size (default to 1)
    tensor_parallel_size = config.hyperparameters.tensor_parallel_size
    
    # Generate SLURM script
    script = f"""#!/bin/bash
#SBATCH --job-name=nemo-rl-{config.algorithm}-training
#SBATCH --nodes={config.cluster.nodes}
#SBATCH --ntasks-per-node={config.cluster.gpus_per_node}
#SBATCH --gpus-per-node={config.cluster.gpus_per_node}
#SBATCH --time={config.cluster.time_limit}
{"#SBATCH --partition=" + partition if partition else "# #SBATCH --partition=batch"}
{"#SBATCH --account=" + account if account else "# #SBATCH --account=your-account"}
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Auto-generated by NeMo RL Web Configurator
# Generated: {datetime.now(timezone.utc).isoformat()}
# Cluster Preset: {cluster_preset or "generic"}

# Environment setup
export CONTAINER="{container}"
export MOUNTS="/data:/data,/home/$USER:/home/$USER"
export GPUS_PER_NODE={config.cluster.gpus_per_node}

# Create directories
mkdir -p logs checkpoints

# For multi-node training, start Ray cluster first
{"bash ray.sub" if config.cluster.nodes > 1 else "# Single-node training - no Ray cluster needed"}

# Training command using config file
python -m nemo_rl.train \\
    --config config.yaml

# Alternative: Training command with inline arguments
# python -m nemo_rl.train \\
#     --model {config.model} \\
#     --dataset {config.dataset} \\
#     --algorithm {config.algorithm} \\
#     --backend {config.backend} \\
#     --max_steps {config.hyperparameters.max_steps} \\
#     --batch_size {config.hyperparameters.batch_size} \\
#     --learning_rate {config.hyperparameters.learning_rate}{"" if config.algorithm != Algorithm.GRPO else f" \\\n#     --num_generations_per_prompt {config.hyperparameters.num_generations_per_prompt}"}{f" \\\n#     --tensor_parallel_size {tensor_parallel_size}" if tensor_parallel_size > 1 else ""}
"""

    # Generate ray.sub file for multi-node
    ray_sub = """#!/bin/bash
# Ray cluster startup script for multi-node training
# Auto-generated by NeMo RL Web Configurator

set -e

# Get the head node address
HEAD_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
HEAD_NODE_IP=$(getent hosts $HEAD_NODE | awk '{ print $1 }')
RAY_PORT=6379
DASHBOARD_PORT=8265

echo "===================================="
echo "Ray Cluster Configuration"
echo "Head Node: $HEAD_NODE ($HEAD_NODE_IP)"
echo "Total Nodes: $SLURM_NNODES"
echo "===================================="

# Start Ray on head node
if [[ $SLURM_NODEID -eq 0 ]]; then
    echo "[$(hostname)] Starting Ray head on $HEAD_NODE_IP:$RAY_PORT"
    ray start --head \\
        --port=$RAY_PORT \\
        --dashboard-port=$DASHBOARD_PORT \\
        --num-cpus=$SLURM_CPUS_PER_TASK \\
        --block &
    
    # Wait for head to be ready
    sleep 10
    echo "[$(hostname)] Ray head started successfully"
fi

# Start Ray workers on other nodes
if [[ $SLURM_NODEID -ne 0 ]]; then
    sleep 15  # Wait for head to be fully ready
    echo "[$(hostname)] Starting Ray worker, connecting to $HEAD_NODE_IP:$RAY_PORT"
    ray start \\
        --address=$HEAD_NODE_IP:$RAY_PORT \\
        --num-cpus=$SLURM_CPUS_PER_TASK \\
        --block &
fi

# Wait for all nodes to be ready
sleep 10
echo "[$(hostname)] Ray cluster setup complete"
"""

    # Generate config.yaml
    config_yaml = f"""# NeMo RL Training Configuration
# Auto-generated by NeMo RL Web Configurator
# Generated: {datetime.now(timezone.utc).isoformat()}

# Algorithm Configuration
algorithm: {config.algorithm}

# Model Configuration  
model:
  name: {config.model}
  backend: {config.backend}
{f"  tensor_parallel_size: {tensor_parallel_size}" if tensor_parallel_size > 1 else "  # tensor_parallel_size: 1  # default"}

# Dataset Configuration
dataset:
  name: {config.dataset}

# Training Hyperparameters
training:
  learning_rate: {config.hyperparameters.learning_rate}
  batch_size: {config.hyperparameters.batch_size}
  max_steps: {config.hyperparameters.max_steps}
{f"  num_generations_per_prompt: {config.hyperparameters.num_generations_per_prompt}" if config.algorithm == Algorithm.GRPO else "  # num_generations_per_prompt: 16  # GRPO only"}

# Cluster Configuration
cluster:
  nodes: {config.cluster.nodes}
  gpus_per_node: {config.cluster.gpus_per_node}
  time_limit: "{config.cluster.time_limit}"
{f'  partition: "{partition}"' if partition else "  # partition: batch"}
{f'  account: "{account}"' if account else "  # account: your-account"}

# Checkpointing
checkpointing:
  save_dir: checkpoints/
  save_every_n_steps: 100
  keep_last_n_checkpoints: 3

# Logging
logging:
  log_dir: logs/
  log_every_n_steps: 10
  wandb:
    enabled: false
    # project: nemo-rl-training
    # entity: your-entity
"""

    # Build files dictionary
    files: dict[str, str] = {"config.yaml": config_yaml}
    if config.cluster.nodes > 1:
        files["ray.sub"] = ray_sub

    # Generate instructions based on cluster setup
    instructions = [
        "1. Save the files to your working directory:",
        "   - train.sh (main SLURM script)",
        "   - config.yaml (training configuration)",
    ]
    
    if config.cluster.nodes > 1:
        instructions.append("   - ray.sub (Ray cluster startup script)")
    
    instructions.extend([
        "2. Make scripts executable: `chmod +x train.sh" + (" ray.sub" if config.cluster.nodes > 1 else "") + "`",
        "3. Review and customize config.yaml as needed",
        "4. Submit job to SLURM: `sbatch train.sh`",
        "5. Monitor progress: `tail -f logs/<job_id>.out`",
        "6. Check checkpoints in: `ls checkpoints/`",
    ])

    return ScriptGenerationResult(
        script=script,
        files=files,
        instructions=instructions
    )


# =============================================================================
# Model Database for Search
# =============================================================================

# Comprehensive model database with metadata
MODEL_DATABASE = [
    # Qwen models
    ModelInfo(
        id="Qwen/Qwen2.5-0.5B",
        name="Qwen2.5-0.5B",
        size_gb=1.0,
        parameters="0.5B",
        recommended_config={"min_gpus": 1, "tensor_parallel": False, "backend": "dtensor"}
    ),
    ModelInfo(
        id="Qwen/Qwen2.5-1.5B",
        name="Qwen2.5-1.5B",
        size_gb=3.0,
        parameters="1.5B",
        recommended_config={"min_gpus": 1, "tensor_parallel": False, "backend": "dtensor"}
    ),
    ModelInfo(
        id="Qwen/Qwen2.5-3B",
        name="Qwen2.5-3B",
        size_gb=6.0,
        parameters="3B",
        recommended_config={"min_gpus": 1, "tensor_parallel": False, "backend": "dtensor"}
    ),
    ModelInfo(
        id="Qwen/Qwen2.5-7B",
        name="Qwen2.5-7B",
        size_gb=14.0,
        parameters="7B",
        recommended_config={"min_gpus": 2, "tensor_parallel": True, "backend": "dtensor"}
    ),
    ModelInfo(
        id="Qwen/Qwen2.5-14B",
        name="Qwen2.5-14B",
        size_gb=28.0,
        parameters="14B",
        recommended_config={"min_gpus": 4, "tensor_parallel": True, "backend": "megatron"}
    ),
    ModelInfo(
        id="Qwen/Qwen2.5-32B",
        name="Qwen2.5-32B",
        size_gb=64.0,
        parameters="32B",
        recommended_config={"min_gpus": 8, "tensor_parallel": True, "backend": "megatron"}
    ),
    ModelInfo(
        id="Qwen/Qwen2.5-72B",
        name="Qwen2.5-72B",
        size_gb=145.0,
        parameters="72B",
        recommended_config={"min_gpus": 16, "tensor_parallel": True, "backend": "megatron"}
    ),
    # Llama models
    ModelInfo(
        id="meta-llama/Llama-3.1-8B",
        name="Llama-3.1-8B",
        size_gb=16.0,
        parameters="8B",
        recommended_config={"min_gpus": 2, "tensor_parallel": True, "backend": "dtensor"}
    ),
    ModelInfo(
        id="meta-llama/Llama-3.1-70B",
        name="Llama-3.1-70B",
        size_gb=140.0,
        parameters="70B",
        recommended_config={"min_gpus": 16, "tensor_parallel": True, "backend": "megatron"}
    ),
    ModelInfo(
        id="meta-llama/Llama-3.2-1B",
        name="Llama-3.2-1B",
        size_gb=2.0,
        parameters="1B",
        recommended_config={"min_gpus": 1, "tensor_parallel": False, "backend": "dtensor"}
    ),
    ModelInfo(
        id="meta-llama/Llama-3.2-3B",
        name="Llama-3.2-3B",
        size_gb=6.0,
        parameters="3B",
        recommended_config={"min_gpus": 1, "tensor_parallel": False, "backend": "dtensor"}
    ),
    # Mistral models
    ModelInfo(
        id="mistralai/Mistral-7B-v0.1",
        name="Mistral-7B-v0.1",
        size_gb=14.0,
        parameters="7B",
        recommended_config={"min_gpus": 2, "tensor_parallel": True, "backend": "dtensor"}
    ),
    ModelInfo(
        id="mistralai/Mixtral-8x7B-v0.1",
        name="Mixtral-8x7B-v0.1",
        size_gb=93.0,
        parameters="46.7B (8x7B MoE)",
        recommended_config={"min_gpus": 8, "tensor_parallel": True, "backend": "megatron"}
    ),
    ModelInfo(
        id="mistralai/Mistral-Small-24B",
        name="Mistral-Small-24B",
        size_gb=48.0,
        parameters="24B",
        recommended_config={"min_gpus": 4, "tensor_parallel": True, "backend": "megatron"}
    ),
    # Gemma models
    ModelInfo(
        id="google/gemma-2b",
        name="Gemma-2B",
        size_gb=4.0,
        parameters="2B",
        recommended_config={"min_gpus": 1, "tensor_parallel": False, "backend": "dtensor"}
    ),
    ModelInfo(
        id="google/gemma-7b",
        name="Gemma-7B",
        size_gb=14.0,
        parameters="7B",
        recommended_config={"min_gpus": 2, "tensor_parallel": True, "backend": "dtensor"}
    ),
    # Phi models
    ModelInfo(
        id="microsoft/phi-3-mini-4k-instruct",
        name="Phi-3-Mini-4K",
        size_gb=7.6,
        parameters="3.8B",
        recommended_config={"min_gpus": 1, "tensor_parallel": False, "backend": "dtensor"}
    ),
    ModelInfo(
        id="microsoft/phi-3-medium-4k-instruct",
        name="Phi-3-Medium-4K",
        size_gb=28.0,
        parameters="14B",
        recommended_config={"min_gpus": 4, "tensor_parallel": True, "backend": "megatron"}
    ),
    # DeepSeek models
    ModelInfo(
        id="deepseek-ai/deepseek-coder-6.7b-base",
        name="DeepSeek-Coder-6.7B",
        size_gb=13.4,
        parameters="6.7B",
        recommended_config={"min_gpus": 2, "tensor_parallel": True, "backend": "dtensor"}
    ),
    ModelInfo(
        id="deepseek-ai/deepseek-math-7b-base",
        name="DeepSeek-Math-7B",
        size_gb=14.0,
        parameters="7B",
        recommended_config={"min_gpus": 2, "tensor_parallel": True, "backend": "dtensor"}
    ),
]


@app.get("/api/v1/models/search", response_model=ModelSearchResult)
async def search_models(
    q: str,
    size: Optional[str] = None,
    limit: Optional[int] = None
):
    """
    Search for models on HuggingFace Hub.
    
    Args:
        q: Search query string (matches against model name and ID)
        size: Filter by model size - "small" (<10GB), "medium" (10-50GB), "large" (>50GB)
        limit: Maximum number of results to return (default: all matching)
    
    Returns:
        ModelSearchResult with list of matching models and their metadata
    
    Note: This is a mock implementation using a curated model database.
    In production, this would query the HuggingFace Hub API.
    """
    # Filter by query
    query_lower = q.lower()
    filtered = [
        m for m in MODEL_DATABASE 
        if query_lower in m.name.lower() or query_lower in m.id.lower()
    ]

    # Filter by size if specified
    if size:
        size_ranges = {
            "small": (0, 10),
            "medium": (10, 50),
            "large": (50, float("inf")),
        }
        if size in size_ranges:
            min_size, max_size = size_ranges[size]
            filtered = [m for m in filtered if min_size <= m.size_gb < max_size]

    # Sort by size (smallest first) for better UX
    filtered.sort(key=lambda m: m.size_gb)
    
    # Apply limit if specified
    if limit and limit > 0:
        filtered = filtered[:limit]

    return ModelSearchResult(results=filtered)


@app.get("/api/v1/datasets/search")
async def search_datasets(
    q: str,
    limit: Optional[int] = 10
):
    """
    Search for datasets on HuggingFace Hub.
    
    Args:
        q: Search query string
        limit: Maximum number of results (default: 10)
    
    Note: This is a mock implementation. In production, this would query
    the HuggingFace Datasets API.
    """
    # Mock dataset database
    all_datasets = [
        {
            "id": "nvidia/OpenMathInstruct-2",
            "name": "OpenMathInstruct-2",
            "description": "Large-scale math instruction dataset",
            "size": "14.5M examples",
            "features": ["problem", "solution", "answer"],
            "downloads": 50000
        },
        {
            "id": "nvidia/HelpSteer2",
            "name": "HelpSteer2",
            "description": "Human preference dataset for alignment",
            "size": "21K examples",
            "features": ["prompt", "response", "helpfulness"],
            "downloads": 25000
        },
        {
            "id": "openai/gsm8k",
            "name": "GSM8K",
            "description": "Grade school math problems",
            "size": "8.5K train examples",
            "features": ["question", "answer"],
            "downloads": 75000
        },
        {
            "id": "hendrycks/competition_math",
            "name": "MATH",
            "description": "Competition mathematics problems",
            "size": "12.5K examples",
            "features": ["problem", "solution", "level"],
            "downloads": 40000
        },
        {
            "id": "tatsu-lab/alpaca",
            "name": "Alpaca",
            "description": "Instruction-following dataset",
            "size": "52K examples",
            "features": ["instruction", "input", "output"],
            "downloads": 100000
        },
    ]
    
    # Filter by query
    query_lower = q.lower()
    filtered = [
        d for d in all_datasets
        if query_lower in d["name"].lower() or 
           query_lower in d["id"].lower() or
           query_lower in d["description"].lower()
    ]
    
    # Apply limit
    if limit and limit > 0:
        filtered = filtered[:limit]
    
    return {"results": filtered}


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
