#!/usr/bin/env bash
# blur faces and number plates in a video using windowed SAM3 tracking.
# Works on videos of any length by processing short windows at a time.
#
# Usage:
#   bash scripts/blur_windowed_video.sh --input INPUT --output OUTPUT [options]
#
# Options:
#   --input PATH          input video file (required)
#   --output PATH         output video file (required)
#   --blur-strength N     gaussian blur kernel size, default 80
#   --window-seconds N    tracking window length in seconds, default 5
#   --prompts p1 p2 ...   objects to blur, default: face "license plate"
set -euo pipefail

# Defaults
BLUR_STRENGTH=80
WINDOW_SECONDS=5
PROMPTS=("face" "license plate")

usage() { echo "Usage: $0 --input <path> --output <path> [--blur-strength N] [--window-seconds N] [--prompts p1 p2 ...]"; exit 1; }

while [[ $# -gt 0 ]]; do
    case $1 in
        --input)          INPUT="$2";          shift 2 ;;
        --output)         OUTPUT="$2";         shift 2 ;;
        --blur-strength)  BLUR_STRENGTH="$2";  shift 2 ;;
        --window-seconds) WINDOW_SECONDS="$2"; shift 2 ;;
        --prompts)
            shift; PROMPTS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do PROMPTS+=("$1"); shift; done ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

[[ -z "${INPUT:-}" ]]  && { echo "Error: --input is required";  usage; }
[[ -z "${OUTPUT:-}" ]] && { echo "Error: --output is required"; usage; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$(dirname "${OUTPUT}")"

conda run --no-capture-output -n sam3 env PYTHONUNBUFFERED=1 \
    python "${SCRIPT_DIR}/blur_video.py" \
    --input "${INPUT}" --output "${OUTPUT}" \
    --blur-strength "${BLUR_STRENGTH}" \
    --window-seconds "${WINDOW_SECONDS}" \
    --prompts "${PROMPTS[@]}"

echo ""
echo "Done. Output: ${OUTPUT}"
