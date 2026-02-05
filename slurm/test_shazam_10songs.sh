#!/bin/bash
#SBATCH --job-name=test_shazam_10
#SBATCH --output=logs/test_shazam_%j.out
#SBATCH --error=logs/test_shazam_%j.err
#SBATCH --partition=edu-medium
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Strict error handling: exit on error, treat unset variables as errors
set -e
set -u

# Create logs and results directories
mkdir -p logs results

echo "=== Testing Shazam on 10 Songs ==="
date
echo "----------------------------------------"
echo "Environment:"
echo "  Sample Rate: 16000 Hz"
echo "  Python: $(which python)"
echo "  Conda Env: songfinder"
echo "----------------------------------------"

# Step 1: Clean and rebuild fingerprints
echo "[1/3] Cleaning old Shazam fingerprints..."
rm -rf fingerprints/shazam
echo "✓ Cleaned"

# Step 2: Index test_songs
echo ""
echo "[2/3] Indexing test_songs with Shazam (SR=16000)..."
conda run -n songfinder python scripts/index_songs.py \
    --approach shazam \
    --folder test_songs \
    --output fingerprints \
    --pattern "*.flac"

if [ $? -ne 0 ]; then
    echo "✗ Indexing failed!"
    exit 1
fi

echo "✓ Indexing completed"

# Step 3: Benchmark
echo ""
echo "[3/3] Running benchmark on test_songs..."
conda run -n songfinder python scripts/benchmark.py \
    --db_dir fingerprints \
    --test_dir test_songs \
    --aug_dir aug \
    --n_test 10 \
    --shazam_only \
    --output results/test_shazam_10songs.json

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Test completed successfully!"
    echo "Results saved to: results/test_shazam_10songs.json"

    # Print summary if jq is available
    if command -v jq &> /dev/null; then
        echo ""
        echo "Summary:"
        jq -r '.shazam.conditions | to_entries[] | "  \(.key): \(.value.accuracy)% (\(.value.correct)/\(.value.total))"' results/test_shazam_10songs.json 2>/dev/null || true
    fi
else
    echo "✗ Test failed with exit code: $EXIT_CODE"
fi

date
echo "========================================"

exit $EXIT_CODE
