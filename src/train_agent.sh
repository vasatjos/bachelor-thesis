#!/usr/bin/env bash

set -e

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Agent name: ${AGENT_NAME}"
echo "Job name: ${JOB_NAME}"
echo "Hyperparameters: ${HYPERPARAMS}"

SCRATCH_DIR="${SCRATCHDIR:-/tmp}"
REPO_URL="git@github.com:vasatjos/bachelor-thesis.git"
PROJECT_DIR="${SCRATCH_DIR}/bachelor-thesis"
STORAGE_DIR="/storage/praha1/home/vasatjos/thesis"
AGENT_STRATEGIES_DIR="${STORAGE_DIR}/agent_strategies"
LOGS_DIR="${STORAGE_DIR}/logs"

# Python will create the hyperparam directory directly inside the agent's folder.
MODEL_BASE_DIR="${AGENT_STRATEGIES_DIR}/${AGENT_NAME}"

mkdir -p "${MODEL_BASE_DIR}"
mkdir -p "${LOGS_DIR}"

cd "${SCRATCH_DIR}"

# Setup SSH to use the deploy key from your storage home dir
export GIT_SSH_COMMAND="ssh -i /storage/praha1/home/vasatjos/.ssh/deploy_key -o StrictHostKeyChecking=no"

echo "Cloning repository..."
rm -rf "${PROJECT_DIR}"
git clone "${REPO_URL}"
cd "${PROJECT_DIR}"

echo "Installing uv..."
module add python/3.11.11-gcc-10.2.1-555dlyc
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="${HOME}/.local/bin:${PATH}"

echo "Installing dependencies..."
uv sync --all-groups

cd src

echo "Starting training at: $(date)"
echo "Running: uv run -m agents.${AGENT_NAME} ${HYPERPARAMS} --model_path ${MODEL_BASE_DIR}"

uv run -m "agents.${AGENT_NAME}" ${HYPERPARAMS} --model_path "${MODEL_BASE_DIR}"

echo "Training completed at: $(date)"

if [ -d "outputs" ]; then
    echo "Copying additional outputs..."
    cp -r outputs "${STORAGE_DIR}/${JOB_NAME}_outputs"
fi

echo "Cleaning up..."
cd "${SCRATCH_DIR}"
rm -rf "${PROJECT_DIR}"

echo "Job completed successfully at: $(date)"
