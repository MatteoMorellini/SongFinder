#!/bin/bash
#SBATCH --job-name=test_grafp_10
#SBATCH --output=logs/test_grafp_%j.out
#SBATCH --error=logs/test_grafp_%j.err
#SBATCH --partition=edu-medium
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

# Strict error handling: exit on error, treat unset variables as errors
set -e
set -u

# Create logs and results directories
mkdir -p logs results

echo "=== Testing GraFP on 10 Songs ==="
date
echo "----------------------------------------"
echo "Environment:"
echo "  Sample Rate: 16000 Hz (from config)"
echo "  Python: $(which python)"
echo "  Conda Env: songfinder"
echo ""
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo "----------------------------------------"

# Step 1: Clean and rebuild fingerprints
echo "[1/3] Cleaning old GraFP fingerprints..."
rm -rf fingerprints/grafp
echo "✓ Cleaned"

# Step 2: Index test_songs
echo ""
echo "[2/3] Indexing test_songs with GraFP (SR=16000)..."
conda run -n songfinder python scripts/index_songs.py \
    --approach grafp \
    --folder test_songs \
    --output fingerprints \
    --pattern "*.flac" \
    --checkpoint checkpoints/model_tc_29_best.pth \
    --config approaches/grafp/config/grafp.yaml \
    --device cuda \
    --chunk-duration 60.0 \
    --chunk-overlap 15.0

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
    --grafp_only \
    --checkpoint checkpoints/model_tc_29_best.pth \
    --config approaches/grafp/config/grafp.yaml \
    --device cuda \
    --output results/test_grafp_10songs.json

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Test completed successfully!"
    echo "Results saved to: results/test_grafp_10songs.json"

    # Print summary if jq is available
    if command -v jq &> /dev/null; then
        echo ""
        echo "Summary:"
        jq -r '.grafp.conditions | to_entries[] | "  \(.key): \(.value.accuracy)% (\(.value.correct)/\(.value.total))"' results/test_grafp_10songs.json 2>/dev/null || true
    fi
else
    echo "✗ Test failed with exit code: $EXIT_CODE"
fi

date
echo "========================================"

exit $EXIT_CODE
