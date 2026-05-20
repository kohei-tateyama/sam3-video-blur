#!/usr/bin/env bash
# split a video into fixed-length chunks for blurring.
#
# Usage:
#   bash scripts/split_video.sh --input INPUT [options]
#
# Options:
#   --input PATH         input video file (required)
#   --output-dir PATH    output folder, default: ./outputs/chunks
#   --chunk-seconds N    chunk length in seconds, default: 5
set -euo pipefail

# Defaults
OUTPUT_DIR="./outputs/chunks"
CHUNK_SECONDS=5

usage() { echo "Usage: $0 --input <path> [--output-dir <path>] [--chunk-seconds N]"; exit 1; }

while [[ $# -gt 0 ]]; do
    case $1 in
        --input)         INPUT="$2";       shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2";  shift 2 ;;
        --chunk-seconds) CHUNK_SECONDS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

[[ -z "${INPUT:-}" ]] && { echo "Error: --input is required"; usage; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHUNK_MINUTES=$(python3 -c "print(${CHUNK_SECONDS} / 60)")

conda run -n sam3 python "${SCRIPT_DIR}/split_video.py" \
    --input         "${INPUT}" \
    --output-dir    "${OUTPUT_DIR}" \
    --chunk-minutes "${CHUNK_MINUTES}"
