#!/usr/bin/env bash
# blur faces and number plates in a video (single SAM3 session).
# Best for short clips (≤ 10 s at 4K). For longer videos use blur_windowed_video.sh.
#
# Usage:
#   bash scripts/blur_full_video.sh --input INPUT --output OUTPUT [options]
#
# Options:
#   --input PATH          input video file (required)
#   --output PATH         output video file (required)
#   --blur-strength N     gaussian blur kernel size, default 80
#   --prompts p1 p2 ...   objects to blur, default: face "license plate"
#   --fps-scale N         output speed multiplier, default 1.0 (e.g. 0.5 for half speed)
set -euo pipefail

# Defaults
BLUR_STRENGTH=80
PROMPTS=("face" "license plate")
FPS_SCALE=1.0

usage() { echo "Usage: $0 --input <path> --output <path> [--blur-strength N] [--fps-scale N] [--prompts p1 p2 ...]"; exit 1; }

while [[ $# -gt 0 ]]; do
    case $1 in
        --input)          INPUT="$2";         shift 2 ;;
        --output)         OUTPUT="$2";        shift 2 ;;
        --blur-strength)  BLUR_STRENGTH="$2"; shift 2 ;;
        --fps-scale)      FPS_SCALE="$2";     shift 2 ;;
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
    python "${SCRIPT_DIR}/blur_full_video.py" \
    --input "${INPUT}" --output "${OUTPUT}" \
    --blur-strength "${BLUR_STRENGTH}" \
    --fps-scale "${FPS_SCALE}" \
    --prompts "${PROMPTS[@]}"

echo ""
echo "Done. Output: ${OUTPUT}"
