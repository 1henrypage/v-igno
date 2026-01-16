#!/bin/bash
#SBATCH --job-name=med_vigno     # Job name
#SBATCH --qos=medium                    # Request QoS (short=4h, medium=2d, long=7d)
#SBATCH --time=23:00:00                 # Request run time (wall-clock)
#SBATCH --partition=general,insy        # Request against appropriate partitions
#SBATCH --ntasks=1                      # Number of (gpu) tasks (keep at 1)
#SBATCH --cpus-per-task=2               # 8 CPU cores (good for data loading)
#SBATCH --mem=32G                       # 64GB RAM (adjust if needed)
#SBATCH --gres=gpu:a40:1                # Request 1 L40 GPU
#SBATCH --output=slurm_logs/job_%j.out  # Set name of output log (%j = jobID)
#SBATCH --error=slurm_logs/job_%j.err   # Set name of error log (%j = jobID)
#SBATCH --mail-type=END,FAIL            # Send email on job end/failure
#SBATCH --mail-user=h.page@student.tudelft.nl
# =============================================================================
# SLURM Job Script for DGNO Training/Evaluation on DAIC
# =============================================================================
# This script sets up the environment and runs ML training or evaluation using 
# the DGNO framework on TU Delft's AI Cluster (DAIC).
#
# Usage:
#   sbatch med.sh <MODE> <CONFIG_FILE> [ADDITIONAL_ARGS]
#
# Examples:
#   sbatch med.sh train configs/my_experiment.yaml
#   sbatch med.sh train configs/my_experiment.yaml --seed 42
#   sbatch  evaluate configs/eval_config.yaml
# =============================================================================

# Print job information
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $(hostname)"
echo "Starting time: $(date)"
echo "=============================================="

# Check GPU allocation
echo "GPU Information:"
nvidia-smi
echo "=============================================="

# =============================================================================
# Navigate to project directory
# =============================================================================
# Use SLURM_SUBMIT_DIR instead of BASH_SOURCE
cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Working directory: $(pwd)"
echo "=============================================="

# =============================================================================
# Parse mode and configuration file
# =============================================================================
# Get mode (train or evaluate)
MODE="${1:?mode required (train or evaluate)}"
shift

# Validate mode
if [[ "$MODE" != "train" && "$MODE" != "evaluate" ]]; then
    echo "ERROR: Invalid mode: $MODE"
    echo "Mode must be either 'train' or 'evaluate'"
    exit 1
fi

# Get config file
CONFIG_FILE="${1:?config file required}"
shift

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Mode: $MODE"
echo "Using configuration: $CONFIG_FILE"
echo "Additional arguments: $*"
echo "=============================================="

# =============================================================================
# Run Training or Evaluation
# =============================================================================
# Determine which script to run based on mode
if [ "$MODE" == "train" ]; then
    SCRIPT="training.py"
    ACTION="Training"
else
    SCRIPT="evaluate.py"
    ACTION="Evaluation"
fi

echo "Starting ${ACTION}..."
echo "Command: uv run python $SCRIPT --config $CONFIG_FILE $*"
echo "=============================================="

# Run the script with srun
# srun ensures proper resource allocation and process management
srun uv run python "$SCRIPT" --config "$CONFIG_FILE" "$@"

# Capture exit status
EXIT_STATUS=$?

# =============================================================================
# Post-Execution Diagnostics
# =============================================================================
echo "=============================================="
echo "${ACTION} completed with exit status: $EXIT_STATUS"
echo "End time: $(date)"
echo "=============================================="
echo "=============================================="
echo "Job $SLURM_JOB_ID finished"
echo "=============================================="

exit $EXIT_STATUS
