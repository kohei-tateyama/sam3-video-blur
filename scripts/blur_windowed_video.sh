#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# blur_windowed_video.sh — blur a video using windowed SAM3 tracking.
#
# Splits the video into short windows and runs SAM3 tracking on each window
# separately, keeping GPU state small enough for any video length.
#
# Edit the variables below, then run:
#   bash blur_windowed_video.sh
# ---------------------------------------------------------------------------
set -euo pipefail

# ── Configure here ───────────────────────────────────────────────────────────

INPUT="/workspace/sam3/outputs/chunks/IMG_0006_part001.mp4"
OUTPUT="./outputs/blurred/IMG_0006_part001.mp4"

BLUR_STRENGTH=51                    # Gaussian blur kernel size (larger = heavier)
WINDOW_SECONDS=5                   # tracking window length (shorter = less VRAM)
PROMPTS=("face" "license plate")   # objects to blur
# GPUS=(0)                         # uncomment to pin to a specific GPU

# ── (no edits needed below) ──────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$(dirname "${OUTPUT}")"

ARGS=(
    --input           "${INPUT}"
    --output          "${OUTPUT}"
    --blur-strength   "${BLUR_STRENGTH}"
    --window-seconds  "${WINDOW_SECONDS}"
    --prompts         "${PROMPTS[@]}"
)
[[ -n "${GPUS+x}" ]] && ARGS+=(--gpus "${GPUS[@]}")

conda run --no-capture-output -n sam3 env PYTHONUNBUFFERED=1 python "${SCRIPT_DIR}/scripts/blur_video.py" "${ARGS[@]}"

echo ""
echo "=== Done. Output: ${OUTPUT} ==="
