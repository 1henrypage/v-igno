#!/bin/bash
#SBATCH --job-name=dgno_training        # Job name
#SBATCH --partition=general             # Request partition
#SBATCH --qos=long                      # Request QoS (short=4h, medium=2d, long=7d)
#SBATCH --time=24:00:00                 # Request run time (wall-clock)
#SBATCH --ntasks=1                      # Number of parallel tasks per job
#SBATCH --cpus-per-task=8               # Number of CPUs (threads) per task
#SBATCH --mem=32G                       # Request memory per node
#SBATCH --gres=gpu:a40:1                # Request 1 A40 GPU (options: a40, l40, v100, turing)
#SBATCH --output=slurm_logs/job_%j.out  # Set name of output log (%j = jobID)
#SBATCH --error=slurm_logs/job_%j.err   # Set name of error log (%j = jobID)
#SBATCH --mail-type=END,FAIL            # Send email on job end/failure
# Uncomment and add your email if you want notifications
# #SBATCH --mail-user=your.email@tudelft.nl

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

# Print SLURM environment details
echo "SLURM Job Details:"
/usr/bin/scontrol show job -d "$SLURM_JOB_ID"
echo "=============================================="

# Check GPU allocation
echo "GPU Information:"
nvidia-smi
echo "=============================================="

# =============================================================================
# Environment Setup
# =============================================================================

# Load required modules (if needed)
# Uncomment the following lines if you need specific modules
# module use /opt/insy/modulefiles
# module load cuda/11.8 cudnn/11.8-8.6.0.163
# module list

# Set up environment variables
export PYTHONUNBUFFERED=1  # Ensure Python output is not buffered

# Set number of threads for CPU operations
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Ensure CUDA is visible
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

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
# Setup Python environment with uv
# =============================================================================

echo "Setting up Python environment..."

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "ERROR: 'uv' command not found."
    echo "Please install uv or load it via a module."
    echo "Installation: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "uv version: $(uv --version)"
echo "=============================================="

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

# Measure GPU usage before training
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' 2>/dev/null | /usr/bin/tail -n '+2')

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

# Measure GPU usage after training
echo "GPU Usage Summary:"
/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' 2>/dev/null

echo "=============================================="
echo "Job $SLURM_JOB_ID finished"
echo "=============================================="

exit $EXIT_STATUS
