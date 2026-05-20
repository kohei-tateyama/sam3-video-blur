#!/usr/bin/env python3
"""
Split a video into fixed-length chunks using OpenCV.

Usage:
    python scripts/split_video.py --input VIDEO --output-dir DIR [--chunk-minutes N]
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2


def split_video(input_path: str, output_dir: str, chunk_minutes: float = 1.0) -> list[str]:
    """Split *input_path* into chunks of *chunk_minutes* each.

    Returns the list of output file paths.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_chunk = int(round(fps * chunk_minutes * 60))

    duration_s = total_frames / fps
    n_chunks = -(-total_frames // frames_per_chunk)  # ceiling division

    print(
        f"Input : {input_path}\n"
        f"        {total_frames} frames  |  {width}×{height}  |  {fps:.2f} fps  "
        f"|  {duration_s:.0f}s ({duration_s/60:.1f} min)\n"
        f"Chunk : {chunk_minutes:.0f} min  ({frames_per_chunk} frames)\n"
        f"Output: {n_chunks} chunks → {output_dir}"
    )

    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(input_path))[0]
    output_paths: list[str] = []

    chunk_idx = 0
    writer = None
    out_path = None
    frames_written = 0

    while True:
        ret, frame = cap.read()

        # Start a new chunk writer when needed
        if writer is None:
            out_path = os.path.join(output_dir, f"{base}_part{chunk_idx + 1:03d}.mp4")
            for fourcc_str in ("avc1", "mp4v"):
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
                if writer.isOpened():
                    break
            output_paths.append(out_path)
            frames_written = 0

        if not ret:
            break

        writer.write(frame)
        frames_written += 1

        if frames_written >= frames_per_chunk:
            writer.release()
            print(f"  wrote {out_path}  ({frames_written} frames)")
            writer = None
            chunk_idx += 1

    # Flush the last (possibly partial) chunk
    if writer is not None:
        writer.release()
        print(f"  wrote {out_path}  ({frames_written} frames)")

    cap.release()
    print(f"\nDone. {len(output_paths)} chunks saved to {output_dir}")
    return output_paths


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="split_video.py",
        description="Split a video into fixed-length chunks.",
    )
    parser.add_argument("--input", required=True, metavar="VIDEO",
                        help="Path to the input video file.")
    parser.add_argument("--output-dir", required=True, metavar="DIR",
                        help="Directory where the chunks will be saved.")
    parser.add_argument("--chunk-minutes", type=float, default=1.0, metavar="N",
                        help="Length of each chunk in minutes (default: 1).")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    if not os.path.isfile(args.input):
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    split_video(args.input, args.output_dir, args.chunk_minutes)


if __name__ == "__main__":
    main()
