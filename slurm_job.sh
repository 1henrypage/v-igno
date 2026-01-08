#!/bin/bash
#SBATCH --job-name=igno_training # Job name
#SBATCH --qos=medium # Request QoS (short=4h, medium=2d, long=7d)
#SBATCH --time=14:00:00                 # Request run time (wall-clock)
#SBATCH --gres=gpu:l40:1                # Request 1 A40 GPU (options: a40, l40, v100, turing)
#SBATCH --output=slurm_logs/job_%j.out  # Set name of output log (%j = jobID)
#SBATCH --error=slurm_logs/job_%j.err   # Set name of error log (%j = jobID)
#SBATCH --mail-type=END,FAIL            # Send email on job end/failure
#SBATCH --mail-user=h.page@student.tudelft.nl

# =============================================================================
# SLURM Job Script for DGNO Training on DAIC
# =============================================================================
# This script sets up the environment and runs ML training using the DGNO
# framework on TU Delft's AI Cluster (DAIC).
#
# Usage:
#   sbatch slurm_job.sh [CONFIG_FILE] [ADDITIONAL_ARGS]
#
# Examples:
#   sbatch slurm_job.sh
#   sbatch slurm_job.sh configs/my_experiment.yaml
#   sbatch slurm_job.sh configs/my_experiment.yaml --seed 42
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

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "Working directory: $(pwd)"
echo "=============================================="

# =============================================================================
# Create necessary directories
# =============================================================================

mkdir -p slurm_logs
mkdir -p runs

# =============================================================================
# Determine configuration file
# =============================================================================

# Use provided config file or default
CONFIG_FILE="${1:-configs/example_config.yaml}"
shift || true  # Remove first argument if it exists

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Using configuration: $CONFIG_FILE"
echo "Additional arguments: $*"
echo "=============================================="

# =============================================================================
# Run Training
# =============================================================================

echo "Starting training..."
echo "Command: uv run python run_training.py --config $CONFIG_FILE $*"
echo "=============================================="


# Run the training script with srun
# srun ensures proper resource allocation and process management
srun uv run python run_training.py --config "$CONFIG_FILE" "$@"

# Capture exit status
EXIT_STATUS=$?

# =============================================================================
# Post-Training Diagnostics
# =============================================================================

echo "=============================================="
echo "Training completed with exit status: $EXIT_STATUS"
echo "End time: $(date)"
echo "=============================================="

echo "=============================================="
echo "Job $SLURM_JOB_ID finished"
echo "=============================================="

exit $EXIT_STATUS
