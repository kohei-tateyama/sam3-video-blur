#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Blur faces and number plates in a video using SAM3 — full-video mode.

Loads the entire video into a single SAM3 session for best tracking quality.
Suitable for short clips (up to ~5-10 seconds on a 24 GB GPU at 4K).
For longer videos use blur_windowed_video.sh instead.

Usage
-----
    python scripts/blur_full_video.py \\
        --input  path/to/input.mp4 \\
        --output path/to/output.mp4 \\
        --prompts "face" "license plate"
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

def load_video_frames(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def save_video(frames: List[np.ndarray], output_path: str, fps: float) -> None:
    h, w = frames[0].shape[:2]
    for fourcc_str in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if writer.isOpened():
            break
    else:
        raise RuntimeError("Could not create VideoWriter.")
    for frame in frames:
        writer.write(frame)
    writer.release()


def write_jpeg_folder(frames: List[np.ndarray], folder: str) -> None:
    os.makedirs(folder, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(folder, f"{i}.jpg"), frame)


# ---------------------------------------------------------------------------
# Blurring
# ---------------------------------------------------------------------------

def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def apply_blur_to_mask(frame: np.ndarray, mask: np.ndarray, blur_strength: int) -> np.ndarray:
    k = _odd(blur_strength)
    blurred = cv2.GaussianBlur(frame, (k, k), 0)
    return np.where(np.repeat(mask[:, :, None], 3, axis=2), blurred, frame)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def blur_video(
    input_path: str,
    output_path: str,
    blur_strength: int = 51,
    prompts: Optional[List[str]] = None,
    gpus_to_use=None,
) -> None:
    if prompts is None:
        prompts = ["face", "license plate"]

    print(f"[1/4] Loading video: {input_path}", flush=True)
    frames_bgr, fps = load_video_frames(input_path)
    n_frames = len(frames_bgr)
    h, w = frames_bgr[0].shape[:2]
    print(f"      {n_frames} frames  |  {w}x{h}  |  {fps:.2f} fps", flush=True)

    if gpus_to_use is None:
        gpus_to_use = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else None

    print("[2/4] Loading SAM3 video predictor...", flush=True)
    from sam3.model_builder import build_sam3_video_predictor
    predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

    print("[3/4] Running SAM3 on full video...", flush=True)
    tmp_dir = tempfile.mkdtemp(prefix="sam3_full_")
    try:
        write_jpeg_folder(frames_bgr, tmp_dir)
        response = predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=tmp_dir,
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
            )
        )
        session_id = response["session_id"]
        accumulated: Dict[int, np.ndarray] = {}

        for prompt in prompts:
            print(f"      prompt: \"{prompt}\"", flush=True)
            predictor.handle_request(dict(type="reset_session", session_id=session_id))
            predictor.handle_request(dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=prompt,
            ))
            for resp in predictor.handle_stream_request(
                dict(type="propagate_in_video", session_id=session_id)
            ):
                fidx = resp["frame_index"]
                masks = resp["outputs"].get("out_binary_masks")
                if masks is None or len(masks) == 0:
                    continue
                fm = np.zeros((h, w), dtype=bool)
                for m in masks:
                    m = np.asarray(m, dtype=bool)
                    if m.ndim == 3:
                        m = m.squeeze(0)
                    if m.shape != (h, w):
                        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                    fm |= m
                accumulated[fidx] = accumulated.get(fidx, np.zeros((h, w), dtype=bool)) | fm

        predictor.handle_request(dict(type="close_session", session_id=session_id))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    del predictor
    torch.cuda.empty_cache()

    print("[4/4] Applying blur and writing output...", flush=True)
    blurred, n_blurred = [], 0
    for idx, frame in enumerate(frames_bgr):
        mask = accumulated.get(idx)
        if mask is not None and mask.any():
            frame = apply_blur_to_mask(frame, mask, blur_strength)
            n_blurred += 1
        blurred.append(frame)

    save_video(blurred, output_path, fps)
    print(f"\nDone. {n_blurred}/{n_frames} frames blurred.\nOutput: {output_path}")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(prog="blur_full_video.py")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--blur-strength", type=int, default=51)
    p.add_argument("--prompts", nargs="+", default=["face", "license plate"])
    p.add_argument("--gpus", nargs="*", type=int, default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if not os.path.isfile(args.input):
        print(f"ERROR: {args.input} not found", file=sys.stderr); sys.exit(1)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    blur_video(args.input, args.output, args.blur_strength, args.prompts, args.gpus)


if __name__ == "__main__":
    main()
