#!/usr/bin/env bash

set -e

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Agent name: $AGENT_NAME"
echo "Job name: $JOB_NAME"
echo "Hyperparameters: $HYPERPARAMS"

SCRATCH_DIR="${SCRATCHDIR:-/tmp}"
REPO_URL="git@github.com:vasatjos/bachelor-thesis.git"
PROJECT_DIR="$SCRATCH_DIR/bachelor-thesis"
STORAGE_DIR="/storage/praha1/home/vasatjos/thesis"
AGENT_STRATEGIES_DIR="$STORAGE_DIR/agent-strategies"
LOGS_DIR="$STORAGE_DIR/logs"

mkdir -p "$AGENT_STRATEGIES_DIR/$AGENT_NAME"
mkdir -p "$LOGS_DIR"

cd "$SCRATCH_DIR"

# Setup SSH to use the deploy key from your storage home dir
export GIT_SSH_COMMAND="ssh -i /storage/praha1/home/vasatjos/.ssh/deploy_key -o StrictHostKeyChecking=no"

echo "Cloning repository..."
git clone "$REPO_URL"
cd "$PROJECT_DIR"

echo "Installing uv..."
module add python/3.11.11-gcc-10.2.1-555dlyc
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "Installing dependencies..."
uv sync --all-groups

echo "Starting training at: $(date)"
echo "Running: uv run -m agents.${AGENT_NAME} $HYPERPARAMS"

cd src
mkdir -p "agent-strategies/$AGENT_NAME"

PYTHON_FILE="agents/${AGENT_NAME}.py"

echo "Starting training at: $(date)"
echo "Running: python $PYTHON_FILE $HYPERPARAMS"

uv run -m agents.$AGENT_NAME $HYPERPARAMS

echo "Training completed at: $(date)"

echo "Copying trained agents to storage..."
cp -r agent-strategies/* "$AGENT_STRATEGIES_DIR/"

if [ -d "outputs" ]; then
    echo "Copying additional outputs..."
    cp -r outputs "$STORAGE_DIR/${JOB_NAME}_outputs"
fi

echo "Cleaning up..."
cd "$SCRATCH_DIR"
rm -rf "$PROJECT_DIR"

echo "Job completed successfully at: $(date)"
