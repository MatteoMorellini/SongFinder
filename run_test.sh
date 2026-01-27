#!/bin/bash
#SBATCH --job-name=songfinder_test
#SBATCH --output=logs/songfinder_test_%j.out
#SBATCH --error=logs/songfinder_test_%j.err
#SBATCH --partition=edu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:05:00

# Create logs directory if it doesn't exist
mkdir -p logs

# Path to the single song
SONG_PATH="./data/Runaway.flac"
CHECKPOINT="./checkpoints/model_tc_29_best.pth"

echo "=== Testing Shazam Approach ==="
# 1. Index song
conda run -n songfinder_clean python scripts/index_songs.py --approach shazam --folder ./data/ --pattern "Runaway.flac"

# 2. Recognize song
conda run -n songfinder_clean python scripts/recognize.py --approach shazam --query "$SONG_PATH"

echo -e "\n=== Testing GraFP Approach ==="
# 1. Index song
conda run -n songfinder_clean python scripts/index_songs.py --approach grafp --folder ./data/ --pattern "Runaway.flac" --checkpoint "$CHECKPOINT"

# 2. Recognize song
conda run -n songfinder_clean python scripts/recognize.py --approach grafp --query "$SONG_PATH" --checkpoint "$CHECKPOINT"
