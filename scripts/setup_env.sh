#!/bin/bash
# =============================================================================
# SongFinder Environment Setup
# =============================================================================
# Creates a clean conda environment with all dependencies.
# Usage: bash scripts/setup_env.sh
# =============================================================================

set -e  # Exit on any error

ENV_NAME="songfinder"

echo "=== SongFinder Environment Setup ==="
echo ""

# Step 1: Create conda environment
echo "[1/5] Creating conda environment '$ENV_NAME' with Python 3.11..."
conda create -n $ENV_NAME python=3.11 -y

# Step 2: Install PyTorch with CUDA support
echo ""
echo "[2/5] Installing PyTorch with CUDA support..."
conda run -n $ENV_NAME pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0

# Step 3: Install numpy first (pinned version for compatibility)
echo ""
echo "[3/5] Installing numpy (pinned for scipy/librosa compatibility)..."
conda run -n $ENV_NAME pip install numpy==1.26.4

# Step 4: Install remaining dependencies
echo ""
echo "[4/5] Installing remaining dependencies..."
conda run -n $ENV_NAME pip install -r requirements.txt

# Step 5: Install faiss-gpu via conda (not available on pip)
echo ""
echo "[5/5] Installing faiss-gpu via conda..."
# Remove any pip faiss to avoid conflicts
conda run -n $ENV_NAME pip uninstall faiss faiss-cpu -y 2>/dev/null || true
# Install faiss-gpu from pytorch channel
conda install -n $ENV_NAME -c pytorch faiss-gpu -y

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate the environment, run:"
echo "    conda activate $ENV_NAME"
echo ""
echo "To test the installation, run:"
echo "    python scripts/recognize.py --approach shazam --query ./data/Runaway.flac"
echo ""
