#!/bin/bash
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --account=nvr_lpr_agentic
#SBATCH --job-name=grpo-api-2node
#SBATCH --partition=interactive
#SBATCH --time=4:00:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/yidong/logs/grpo-api-test/grpo-api-2node-%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/yidong/logs/grpo-api-test/grpo-api-2node-%j.err

# Create log directory
mkdir -p /lustre/fsw/portfolios/nvr/users/yidong/logs/grpo-api-test

# Set up the environment variables that ray.sub expects
export CONTAINER=/lustre/fsw/portfolios/nvr/users/yidong/data/models/images/nvidian+nemo+prorl_agent_0.10.1_fix.sqsh
export MOUNTS="/lustre/fsw/portfolios/nvr/users/yidong:/lustre/fsw/portfolios/nvr/users/yidong,/lustre:/lustre"
export GPUS_PER_NODE=8
export COMMAND="cd /lustre/fsw/portfolios/nvr/users/yidong/Projects/RL && export NEMO_RL_PY_EXECUTABLES_SYSTEM=1 && export PYTHONPATH=/lustre/fsw/portfolios/nvr/users/yidong/Projects/RL:\$PYTHONPATH && export HF_HOME=/lustre/fsw/portfolios/nvr/users/yidong/data/models/huggingface_hub && python examples/test_new_grpo_api.py --max-steps 10 --gpus-per-node 8 --num-nodes 2 --num-prompts-per-step 16 --num-generations-per-prompt 4"

# Run the ray.sub script
cd /lustre/fsw/portfolios/nvr/users/yidong/Projects/RL
bash ray.sub
