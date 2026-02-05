#!/bin/bash
#SBATCH --job-name=benchmark_all
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err
#SBATCH --partition=edu-medium
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Create logs directory if it doesn't exist
mkdir -p logs

echo "=== Starting Full Benchmark (Shazam + GraFP) ==="
date
echo "----------------------------------------"

# Current timestamp for filename
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Set Shazam environment variables (defaults from config.py)
export SHAZAM_N_FFT=2048
export SHAZAM_TARGET_SR=11025
export SHAZAM_FAN_OUT=5

# Create results directory
mkdir -p results

# Run full benchmark (Shazam + GraFP)
conda run -n songfinder python scripts/benchmark.py \
    --db_dir ./fingerprints \
    --test_dir ./data/fma_small \
    --aug_dir ./aug \
    --n_test 10 \
    --checkpoint ./checkpoints/model_tc_29_best.pth \
    --device cuda \
    --output ./results/benchmark_full_${TIMESTAMP}.json

echo "----------------------------------------"
echo "Full Benchmark completed."
date
