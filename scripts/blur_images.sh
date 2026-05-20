#!/usr/bin/env bash
# blur faces and number plates in a single image or all images in a folder.
#
# Usage:
#   bash scripts/blur_images.sh --input INPUT --output OUTPUT [options]
#
# Options:
#   --input PATH          image file or folder of images (required)
#   --output PATH         output file or folder (required)
#   --blur-strength N     gaussian blur kernel size, default 80
#   --prompts p1 p2 ...   objects to blur, default: face "license plate"
set -euo pipefail

# Defaults
BLUR_STRENGTH=80
PROMPTS=("face" "license plate")

usage() { echo "Usage: $0 --input <path> --output <path> [--blur-strength N] [--prompts p1 p2 ...]"; exit 1; }

while [[ $# -gt 0 ]]; do
    case $1 in
        --input)         INPUT="$2";          shift 2 ;;
        --output)        OUTPUT="$2";         shift 2 ;;
        --blur-strength) BLUR_STRENGTH="$2";  shift 2 ;;
        --prompts)
            shift; PROMPTS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do PROMPTS+=("$1"); shift; done ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

[[ -z "${INPUT:-}" ]]  && { echo "Error: --input is required";  usage; }
[[ -z "${OUTPUT:-}" ]] && { echo "Error: --output is required"; usage; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

conda run --no-capture-output -n sam3 env PYTHONUNBUFFERED=1 \
    python "${SCRIPT_DIR}/blur_images.py" \
    --input "${INPUT}" --output "${OUTPUT}" \
    --blur-strength "${BLUR_STRENGTH}" \
    --prompts "${PROMPTS[@]}"

echo ""
echo "Done. Output: ${OUTPUT}"
