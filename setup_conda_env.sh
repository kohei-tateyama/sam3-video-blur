#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_conda_env.sh
#
# Creates and configures a Conda environment to run SAM3.
#
# Prerequisites:
#   - conda (Miniconda or Anaconda) installed and on PATH
#   - CUDA 12.6+ compatible GPU and driver
#   - HuggingFace authentication for model download:
#       hf auth login   (generate a token at https://huggingface.co/settings/tokens)
#
# Usage:
#   bash setup_conda_env.sh            # creates env named "sam3"
#   bash setup_conda_env.sh myenv      # creates env named "myenv"
# ---------------------------------------------------------------------------
set -euo pipefail

ENV_NAME="${1:-sam3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 1. Create the environment ────────────────────────────────────────────────
echo ">>> Creating conda environment '${ENV_NAME}' with Python 3.12..."
conda create -y -n "${ENV_NAME}" python=3.12

# Activate inside the script
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# ── 2. PyTorch (CUDA 12.8) ───────────────────────────────────────────────────
echo ">>> Installing PyTorch 2.10.0 with CUDA 12.8 support..."
pip install torch==2.10.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# ── 3. SAM3 package ──────────────────────────────────────────────────────────
echo ">>> Installing SAM3 (editable) with notebook and dev extras..."
cd "${SCRIPT_DIR}"
pip install -e ".[notebooks,dev]"

# ── 4. Optional speed-up dependencies ────────────────────────────────────────
echo ">>> Installing einops and ninja (faster kernels)..."
pip install einops ninja

# Uncomment below for Hopper/Ada GPUs (H100, RTX 40xx, etc.) that support FA-3:
# echo ">>> Installing FlashAttention-3..."
# pip install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu128
# pip install git+https://github.com/ronghanghu/cc_torch.git

# ── 5. Done ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Environment '${ENV_NAME}' is ready."
echo ""
echo " Activate it with:"
echo "   conda activate ${ENV_NAME}"
echo ""
echo " Before first use, authenticate with HuggingFace to allow"
echo " automatic model checkpoint download:"
echo "   hf auth login"
echo "============================================================"
