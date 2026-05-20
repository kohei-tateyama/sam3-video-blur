#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# blur_video.sh — blur faces and number plates in a video using SAM3.
#
# Edit the variables below, then run:
#   bash blur_video.sh
#
# To split a long video into chunks first, use split_video.sh.
# ---------------------------------------------------------------------------
set -euo pipefail

# ── Configure paths and options here ────────────────────────────────────────

INPUT="/workspace/sam3/outputs/chunks/IMG_0006_part035.mp4"  # path to the input video
OUTPUT="./outputs/output_blurred.mp4"         # path for the anonymised output video

BLUR_STRENGTH=51                              # Gaussian blur kernel size (larger = heavier)
WINDOW_SECONDS=5                             # tracking window length in seconds (shorter = less VRAM)
PROMPTS=("face" "license plate")             # objects to blur
# GPUS=(0)                                   # uncomment to pin to specific GPU(s)

# ── (no edits needed below this line) ────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$(dirname "${OUTPUT}")"

ARGS=(
    --input  "${INPUT}"
    --output "${OUTPUT}"
    --blur-strength "${BLUR_STRENGTH}"
    --window-seconds "${WINDOW_SECONDS}"
    --prompts "${PROMPTS[@]}"
)
if [[ -n "${GPUS+x}" ]]; then
    ARGS+=(--gpus "${GPUS[@]}")
fi

conda run --no-capture-output -n sam3 env PYTHONUNBUFFERED=1 python "${SCRIPT_DIR}/scripts/blur_video.py" "${ARGS[@]}"

echo ""
echo "=== Done. Output saved to: ${OUTPUT} ==="
