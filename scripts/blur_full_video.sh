#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# blur_full_video.sh — blur a video in a single SAM3 session (best quality).
#
# Best for short clips (≤ 5-10 s at 4K on a 24 GB GPU).
# For longer videos use blur_windowed_video.sh.
#
# Edit the variables below, then run:
#   bash blur_full_video.sh
# ---------------------------------------------------------------------------
set -euo pipefail

# ── Configure here ───────────────────────────────────────────────────────────

INPUT="/workspace/sam3/outputs/chunks/IMG_0006_part030.mp4"
OUTPUT="./outputs/blurred/IMG_0006_part030_driver.mp4"

BLUR_STRENGTH=80                    # Gaussian blur kernel size (larger = heavier)
PROMPTS=("face" "driver" "license plate")   # objects to blur
# GPUS=(0)                         # uncomment to pin to a specific GPU

# ── (no edits needed below) ──────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$(dirname "${OUTPUT}")"

ARGS=(
    --input          "${INPUT}"
    --output         "${OUTPUT}"
    --blur-strength  "${BLUR_STRENGTH}"
    --prompts        "${PROMPTS[@]}"
)
[[ -n "${GPUS+x}" ]] && ARGS+=(--gpus "${GPUS[@]}")

conda run --no-capture-output -n sam3 env PYTHONUNBUFFERED=1 python "${SCRIPT_DIR}/scripts/blur_full_video.py" "${ARGS[@]}"

echo ""
echo "=== Done. Output: ${OUTPUT} ==="
