#!/usr/bin/env bash
# Alias for blur_windowed_video.sh — see that script for full options.
#
# Usage:
#   bash scripts/blur_video.sh --input INPUT --output OUTPUT [options]
exec "$(dirname "${BASH_SOURCE[0]}")/blur_windowed_video.sh" "$@"
