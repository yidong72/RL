#!/bin/bash
#SBATCH --job-name=grpo-api-test-v2
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=02:00:00
#SBATCH --account=nvr_lpr_agentic
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/yidong/logs/grpo-api-test/output_v2_%j.log
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/yidong/logs/grpo-api-test/error_v2_%j.log

set -eou pipefail

echo "Starting GRPO API test v2..."
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-not_set}"

# Container settings
CONTAINER_IMAGE="/lustre/fsw/portfolios/nvr/users/yidong/data/models/images/nvidian+nemo+verl_v2_enroot_dev0.8.5.sqsh"
MOUNTS="/lustre/fsw/portfolios/nvr/users/yidong/Projects:/projects,/lustre/fsw/portfolios/nvr/users/yidong/data:/datasets,/lustre/fsw/portfolios/nvr/users/yidong/logs:/logs,/lustre/fsw/portfolios/nvr/users/yidong/results:/results"

# Run with enroot
srun --container-image=${CONTAINER_IMAGE} \
     --container-mounts=${MOUNTS} \
     --container-workdir=/projects/RL \
     bash -c '
set -eou pipefail
echo "Inside container at $(date)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

export PYTHONPATH=/projects/RL:${PYTHONPATH:-}
export NRL_IGNORE_VERSION_MISMATCH=1
export NRL_FORCE_REBUILD_VENVS=true
export HF_HOME=/datasets/models/huggingface_hub
export HF_HUB_CACHE=/datasets/models/huggingface_hub

mkdir -p /logs/grpo-api-test

echo ""
echo "========================================"
echo "Running GRPO training with DeepSeek-R1-Distill-Qwen-1.5B"
echo "NRL_FORCE_REBUILD_VENVS=true"
echo "========================================"

cd /projects/RL
python examples/run_grpo_math.py \
    --config examples/configs/recipes/llm/grpo-deepscaler-1.5b-8K.yaml \
    grpo.max_num_steps=3 \
    grpo.num_prompts_per_step=4 \
    grpo.num_generations_per_prompt=2 \
    policy.train_global_batch_size=4 \
    policy.train_micro_batch_size=1 \
    cluster.gpus_per_node=8 \
    logger.log_dir=/logs/grpo-api-test \
    logger.wandb_enabled=false \
    logger.tensorboard_enabled=true \
    logger.monitor_gpus=false \
    checkpointing.enabled=false

echo ""
echo "GRPO training completed!"
echo "Date: $(date)"
'
