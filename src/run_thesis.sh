#!/usr/bin/env bash

if [ $# -lt 2 ]; then
    echo "Usage:"
    echo "  run_thesis.sh <agent_name> <job_name> [additional args...]"
    echo ""
    echo "Examples:"
    echo "  run_thesis.sh q_learning my_job --episodes 1000000 --alpha 0.1"
    echo "  run_thesis.sh dqn experiment1 --epsilon 0.3 --gamma 0.99"
    exit 1
fi

NCPUS=1
NGPUS=1
MEMORY="32gb"
STORAGE="32gb"
CPU_SPEED="5.5"
WALLTIME="48:0:0"

AGENT_NAME=$1
JOB_NAME=$2
shift 2  # Remove first two arguments, leaving only hyperparameters
HYPERPARAMS="$@"

echo
echo "Starting a new job with name '${JOB_NAME}'. Job ID:"

# Submit job to the cluster
qsub -o "/storage/praha1/home/vasatjos/thesis/logs/${JOB_NAME}.out" \
     -e "/storage/praha1/home/vasatjos/thesis/logs/${JOB_NAME}.err" \
     -l walltime=${WALLTIME} \
     -l select=1:ncpus=${NCPUS}:ngpus=${NGPUS}:mem=${MEMORY}:scratch_local=${STORAGE}:spec=${CPU_SPEED} \
     -v AGENT_NAME=${AGENT_NAME},JOB_NAME=${JOB_NAME},HYPERPARAMS="${HYPERPARAMS}" \
     -N ${JOB_NAME} \
     train_agent.sh

echo
echo "Run 'qstat <job_id>' to see job progress."
echo "Run 'qdel <job_id> to stop the job."
