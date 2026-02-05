#!/bin/bash
#SBATCH --job-name=resample_bench
#SBATCH --output=results/resample_benchmark_%j.log
#SBATCH --error=results/resample_benchmark_%j.err
#SBATCH --time=00:02:00
#SBATCH --partition=edu-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8G

# Load conda
source ~/.bashrc
conda activate songfinder

echo "======================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "======================================="
echo

# Run the benchmark
python test_resample_speed.py

echo
echo "Benchmark completed!"
