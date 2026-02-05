#!/bin/bash
#SBATCH --job-name=bench_grafp_500
#SBATCH --output=logs/benchmark_grafp_%j.out
#SBATCH --error=logs/benchmark_grafp_%j.err
#SBATCH --partition=edu-long
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# Strict error handling
set -e
set -u

# Unbuffered output for real-time monitoring
export PYTHONUNBUFFERED=1

# Configuration
DATASET_DIR="${DATASET_DIR:-$HOME/datasets/top100}"
N_TEST=500
PATTERN="*.flac"
CHECKPOINT="checkpoints/model_tc_29_best.pth"
CONFIG="approaches/grafp/config/grafp.yaml"

# Create directories
mkdir -p logs results fingerprints

echo "========================================"
echo "GRAFP BENCHMARK - 500 Songs"
echo "========================================"
date
echo ""
echo "Configuration:"
echo "  Dataset: $DATASET_DIR"
echo "  N_test: $N_TEST"
echo "  Sample Rate: 16000 Hz (from config)"
echo "  Pattern: $PATTERN"
echo "  Checkpoint: $CHECKPOINT"
echo "  Config: $CONFIG"
echo "  Python: $(which python)"
echo "  Conda Env: songfinder"
echo ""
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo "========================================"

# Step 1: Clean old fingerprints
echo ""
echo "[1/3] Cleaning old GraFP fingerprints..."
rm -rf fingerprints/grafp
echo "✓ Cleaned"

# Step 2: Index entire dataset
echo ""
echo "[2/3] Indexing dataset with GraFP (SR=16000)..."
START_INDEX=$(date +%s)

conda run -n songfinder python scripts/index_songs.py \
    --approach grafp \
    --folder "$DATASET_DIR" \
    --output fingerprints \
    --pattern "$PATTERN" \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --device cuda \
    --chunk-duration 60.0 \
    --chunk-overlap 15.0

INDEX_EXIT=$?
END_INDEX=$(date +%s)
INDEX_TIME=$((END_INDEX - START_INDEX))

if [ $INDEX_EXIT -ne 0 ]; then
    echo "✗ Indexing failed with exit code: $INDEX_EXIT"
    exit 1
fi

echo "✓ Indexing completed in ${INDEX_TIME}s"

# Check GPU memory after indexing
echo ""
echo "GPU status after indexing:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

# Step 3: Run comprehensive benchmark
echo ""
echo "[3/3] Running benchmark on $N_TEST songs..."
START_BENCH=$(date +%s)

# Use || true to prevent immediate exit on benchmark failure
conda run -n songfinder python scripts/benchmark.py \
    --db_dir fingerprints \
    --test_dir "$DATASET_DIR" \
    --aug_dir aug \
    --n_test $N_TEST \
    --grafp_only \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --device cuda \
    --output results/benchmark_grafp_500songs.json || BENCH_EXIT=$?

END_BENCH=$(date +%s)
BENCH_TIME=$((END_BENCH - START_BENCH))

echo ""
echo "========================================"
echo "BENCHMARK COMPLETE"
echo "========================================"

if [ ${BENCH_EXIT:-0} -eq 0 ]; then
    echo "✓ Benchmark completed successfully!"
    echo ""
    echo "Timing:"
    echo "  Indexing: ${INDEX_TIME}s"
    echo "  Benchmark: ${BENCH_TIME}s"
    echo "  Total: $((INDEX_TIME + BENCH_TIME))s"
    echo ""
    echo "Results saved to: results/benchmark_grafp_500songs.json"

    # Print detailed summary if jq is available
    if command -v jq &> /dev/null; then
        echo ""
        echo "Accuracy Summary:"
        jq -r '.grafp | "DB Songs: \(.n_db_songs)\nQueries: \(.n_queries)\nDB Load Time: \(.db_load_time_ms)ms\n\nCondition Results:"' results/benchmark_grafp_500songs.json 2>/dev/null || true
        jq -r '.grafp.conditions | to_entries[] | "  \(.key):\n    Accuracy: \(.value.accuracy)%\n    Correct: \(.value.correct)/\(.value.total)\n    Avg Query Time: \(.value.avg_query_time_ms)ms\n    Errors: \(.value.errors | length)"' results/benchmark_grafp_500songs.json 2>/dev/null || true

        echo ""
        echo "Timing Breakdown (average per query):"
        jq -r '.grafp.timings | to_entries[] | "  \(.key):\n\(.value | to_entries[] | "    \(.key): \(.value)s")"' results/benchmark_grafp_500songs.json 2>/dev/null || true
    fi
else
    echo "✗ Benchmark failed with exit code: ${BENCH_EXIT:-1}"
    echo "  Indexing: ${INDEX_TIME}s"
    echo "  Benchmark: ${BENCH_TIME}s (FAILED)"
    echo ""
    echo "Partial results may be available in: results/benchmark_grafp_500songs.json"
fi

date
echo "========================================"

exit ${BENCH_EXIT:-0}
