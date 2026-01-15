#!/bin/bash
# =============================================================================
# SLURM Job Submission Helper
# =============================================================================
# Usage: ./slurm_job.sh <QOS> <MODE> <CONFIG_FILE> [ADDITIONAL_ARGS]
#
# Examples:
#   ./slurm_job.sh short train configs/my_experiment.yaml
#   ./slurm_job.sh medium evaluate configs/eval_config.yaml --seed 42
# =============================================================================

set -e

# Parse arguments
QOS_LEVEL="${1:?QoS level required (short or medium)}"
MODE="${2:?mode required (train or evaluate)}"
CONFIG_FILE="${3:?config file required}"
shift 3
ADDITIONAL_ARGS="$@"

# Validate QoS level
if [[ "$QOS_LEVEL" != "short" && "$QOS_LEVEL" != "medium" && "$QOS_LEVEL" != "long" ]]; then
    echo "ERROR: Invalid QoS level: $QOS_LEVEL"
    echo "Must be one of: short, medium, long"
    exit 1
fi

# Validate mode
if [[ "$MODE" != "train" && "$MODE" != "evaluate" ]]; then
    echo "ERROR: Invalid mode: $MODE"
    echo "Mode must be either 'train' or 'evaluate'"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Determine which SLURM script to use
case "$QOS_LEVEL" in
    short)
        SLURM_SCRIPT="short.sh"
        ;;
    medium)
        SLURM_SCRIPT="med.sh"
        ;;
    long)
        SLURM_SCRIPT="long.sh"  # You can create this later if needed
        ;;
esac

# Check if SLURM script exists
if [ ! -f "$SLURM_SCRIPT" ]; then
    echo "ERROR: SLURM script not found: $SLURM_SCRIPT"
    exit 1
fi

# Submit the job
echo "=============================================="
echo "Submitting job with:"
echo "  QoS: $QOS_LEVEL"
echo "  Script: $SLURM_SCRIPT"
echo "  Mode: $MODE"
echo "  Config: $CONFIG_FILE"
echo "  Additional args: $ADDITIONAL_ARGS"
echo "=============================================="

sbatch "$SLURM_SCRIPT" "$MODE" "$CONFIG_FILE" $ADDITIONAL_ARGS

echo "Job submitted successfully!"