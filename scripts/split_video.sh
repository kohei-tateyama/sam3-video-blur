#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# split_video.sh — split a video into fixed-length chunks.
#
# Edit the variables below, then run:
#   bash split_video.sh
# ---------------------------------------------------------------------------
set -euo pipefail

# ── Configure paths and options here ────────────────────────────────────────

INPUT="/workspace/sam3/inputs/IMG_0006.MOV"  # path to the input video
OUTPUT_DIR="./outputs/chunks"                # folder where chunks will be saved

CHUNK_SECONDS=5                              # length of each chunk in seconds

# ── (no edits needed below this line) ────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Convert seconds to minutes (split_video.py uses --chunk-minutes)
CHUNK_MINUTES=$(python3 -c "print(${CHUNK_SECONDS} / 60)")

conda run -n sam3 python "${SCRIPT_DIR}/scripts/split_video.py" \
    --input         "${INPUT}" \
    --output-dir    "${OUTPUT_DIR}" \
    --chunk-minutes "${CHUNK_MINUTES}"
