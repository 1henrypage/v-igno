#!/bin/bash
# =============================================================================
# SLURM Job Management Helper Script
# =============================================================================
# This script provides convenient commands for managing SLURM jobs on DAIC
#
# Usage:
#   ./slurm_helper.sh submit <mode> <config_file> [args]  # Submit single job
#   ./slurm_helper.sh submit-array [config_list]          # Submit array job
#   ./slurm_helper.sh status                              # Check job status
#   ./slurm_helper.sh cancel <job_id>                     # Cancel a job
#   ./slurm_helper.sh cancel-all                          # Cancel all your jobs
#   ./slurm_helper.sh logs <job_id>                       # View job logs
#   ./slurm_helper.sh interactive [gpu_type]              # Start interactive session
#   ./slurm_helper.sh gpu-status                          # Check GPU availability
#   ./slurm_helper.sh help                                # Show this help
# =============================================================================

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "${BLUE}=============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=============================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Command functions
cmd_submit() {
    print_header "Submitting Single Job"
    
    MODE="${1:-train}"
    shift || true
    
    # Validate mode
    if [[ "$MODE" != "train" && "$MODE" != "evaluate" ]]; then
        print_error "Invalid mode: $MODE"
        print_info "Mode must be either 'train' or 'evaluate'"
        print_info "Usage: ./slurm_helper.sh submit <mode> <config_file> [args]"
        exit 1
    fi
    
    CONFIG_FILE="${1:-configs/example_config.yaml}"
    shift || true
    
    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "Config file not found: $CONFIG_FILE"
        exit 1
    fi
    
    print_info "Mode: $MODE"
    print_info "Config: $CONFIG_FILE"
    print_info "Additional args: $*"
    
    # Create logs directory
    mkdir -p slurm_logs
    
    # Submit job with mode as first argument
    JOB_ID=$(sbatch slurm_job.sh "$MODE" "$CONFIG_FILE" "$@" | awk '{print $4}')
    
    if [ -n "$JOB_ID" ]; then
        print_success "Job submitted with ID: $JOB_ID"
        print_info "Monitor with: squeue -j $JOB_ID"
        print_info "View logs with: ./slurm_helper.sh logs $JOB_ID"
    else
        print_error "Failed to submit job"
        exit 1
    fi
}

cmd_submit_array() {
    print_header "Submitting Array Job"
    
    CONFIG_LIST="${1:-configs/array_configs.txt}"
    
    if [ ! -f "$CONFIG_LIST" ]; then
        print_error "Config list not found: $CONFIG_LIST"
        print_info "Create a file with one config per line:"
        print_info "  train configs/exp1.yaml --seed 42"
        print_info "  evaluate configs/exp2.yaml --seed 123"
        exit 1
    fi
    
    # Count number of configurations
    NUM_CONFIGS=$(wc -l < "$CONFIG_LIST")
    
    print_info "Config list: $CONFIG_LIST"
    print_info "Number of tasks: $NUM_CONFIGS"
    
    # Create logs directory
    mkdir -p slurm_logs
    
    # Submit array job
    JOB_ID=$(sbatch --array=1-$NUM_CONFIGS slurm_array_job.sh "$CONFIG_LIST" | awk '{print $4}')
    
    if [ -n "$JOB_ID" ]; then
        print_success "Array job submitted with ID: $JOB_ID"
        print_info "Monitor with: squeue -j $JOB_ID"
        print_info "View specific task logs with: ./slurm_helper.sh logs ${JOB_ID}_<task_id>"
    else
        print_error "Failed to submit array job"
        exit 1
    fi
}

cmd_status() {
    print_header "Your Job Status"
    squeue -u $USER -o "%.18i %.12j %.8T %.10M %.10L %.6D %.20R %.10b"
    echo ""
    print_info "Job states: PD=Pending, R=Running, CG=Completing, CD=Completed"
}

cmd_cancel() {
    JOB_ID=$1
    if [ -z "$JOB_ID" ]; then
        print_error "Please provide a job ID"
        print_info "Usage: ./slurm_helper.sh cancel <job_id>"
        exit 1
    fi
    
    print_warning "Cancelling job $JOB_ID..."
    scancel $JOB_ID
    print_success "Job $JOB_ID cancelled"
}

cmd_cancel_all() {
    print_warning "Cancelling all your jobs..."
    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        scancel -u $USER
        print_success "All jobs cancelled"
    else
        print_info "Cancelled operation"
    fi
}

cmd_logs() {
    JOB_ID=$1
    if [ -z "$JOB_ID" ]; then
        print_error "Please provide a job ID"
        print_info "Usage: ./slurm_helper.sh logs <job_id>"
        exit 1
    fi
    
    print_header "Logs for Job $JOB_ID"
    
    # Find log files
    OUT_FILE=$(find slurm_logs -name "*${JOB_ID}.out" -o -name "*${JOB_ID}_*.out" 2>/dev/null | head -n 1)
    ERR_FILE=$(find slurm_logs -name "*${JOB_ID}.err" -o -name "*${JOB_ID}_*.err" 2>/dev/null | head -n 1)
    
    if [ -n "$OUT_FILE" ]; then
        print_info "Output log: $OUT_FILE"
        echo ""
        tail -n 50 "$OUT_FILE"
    else
        print_warning "No output log found for job $JOB_ID"
    fi
    
    echo ""
    if [ -n "$ERR_FILE" ] && [ -s "$ERR_FILE" ]; then
        print_info "Error log: $ERR_FILE"
        echo ""
        tail -n 50 "$ERR_FILE"
    fi
}

cmd_interactive() {
    GPU_TYPE="${1:-a40}"
    
    print_header "Starting Interactive Session"
    print_info "GPU Type: $GPU_TYPE"
    print_info "Duration: 4 hours (max for 'short' QoS)"
    print_warning "Remember to exit when done to free resources!"
    
    sinteractive \
        --partition=general \
        --qos=short \
        --time=04:00:00 \
        --cpus-per-task=8 \
        --mem=32G \
        --gres=gpu:${GPU_TYPE}:1
}

cmd_gpu_status() {
    print_header "GPU Availability on DAIC"
    
    echo ""
    print_info "Available GPU types and their specifications:"
    echo ""
    echo "GPU Type | Count | Model                        | Memory   "
    echo "---------|-------|------------------------------|----------"
    echo "l40      | 18    | NVIDIA L40 (Ada Lovelace)   | 49 GB    "
    echo "a40      | 84    | NVIDIA A40 (Ampere)         | 46 GB    "
    echo "turing   | 24    | GeForce RTX 2080 Ti         | 11 GB    "
    echo "v100     | 11    | Tesla V100-SXM2             | 32 GB    "
    
    echo ""
    print_info "Current GPU node availability:"
    sinfo -o '%N %G %C %m %e' | grep gpu || print_warning "No GPU info available"
    
    echo ""
    print_info "To request a specific GPU type, use:"
    echo "  #SBATCH --gres=gpu:<type>:1"
    echo "  Examples: gpu:a40:1, gpu:l40:1, gpu:v100:1"
}

cmd_help() {
    cat << EOF
SLURM Job Management Helper Script for DAIC

USAGE:
  ./slurm_helper.sh <command> [options]

COMMANDS:
  submit <mode> <config> [args]  Submit a single training or evaluation job
                                 Mode: train or evaluate
                                 Example: ./slurm_helper.sh submit train configs/exp.yaml --seed 42
                                 Example: ./slurm_helper.sh submit evaluate configs/eval.yaml

  submit-array [config_list]     Submit an array job for parallel experiments
                                 Example: ./slurm_helper.sh submit-array configs/array_configs.txt

  status                         Show status of all your jobs

  cancel <job_id>                Cancel a specific job
                                 Example: ./slurm_helper.sh cancel 12345

  cancel-all                     Cancel all your jobs (with confirmation)

  logs <job_id>                  View logs for a specific job
                                 Example: ./slurm_helper.sh logs 12345

  interactive [gpu_type]         Start an interactive session with GPU
                                 Example: ./slurm_helper.sh interactive a40

  gpu-status                     Show GPU availability and specifications

  help                           Show this help message

COMMON SLURM COMMANDS:
  squeue -u \$USER               List your jobs
  squeue -j <job_id>             Show specific job details
  scancel <job_id>               Cancel a job
  seff <job_id>                  Show resource usage (after job completes)
  scontrol show job <job_id>     Show detailed job information

GPU TYPES ON DAIC:
  a40    - NVIDIA A40 (46 GB, most common, 84 available)
  l40    - NVIDIA L40 (49 GB, newest, 18 available)
  v100   - Tesla V100 (32 GB, 11 available)
  turing - RTX 2080 Ti (11 GB, 24 available)

QUALITY OF SERVICE (QoS):
  short  - Max 4 hours, highest priority
  medium - Max 2 days
  long   - Max 7 days

For more information, visit: https://daic.tudelft.nl/docs/

EOF
}

# Main command dispatcher
main() {
    if [ $# -eq 0 ]; then
        cmd_help
        exit 0
    fi
    
    COMMAND=$1
    shift
    
    case $COMMAND in
        submit)
            cmd_submit "$@"
            ;;
        submit-array)
            cmd_submit_array "$@"
            ;;
        status)
            cmd_status
            ;;
        cancel)
            cmd_cancel "$@"
            ;;
        cancel-all)
            cmd_cancel_all
            ;;
        logs)
            cmd_logs "$@"
            ;;
        interactive)
            cmd_interactive "$@"
            ;;
        gpu-status)
            cmd_gpu_status
            ;;
        help|--help|-h)
            cmd_help
            ;;
        *)
            print_error "Unknown command: $COMMAND"
            echo ""
            cmd_help
            exit 1
            ;;
    esac
}

main "$@"
