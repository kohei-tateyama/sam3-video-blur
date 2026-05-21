#!/usr/bin/env python3
"""
Blur faces and number plates in images using SAM3.

Works on a single image file or all images in a folder.
For images wider than 1920 px (e.g. 4K), tiled inference is used automatically
so that small objects like licence plates are detected at full resolution.

Usage
-----
    # single image
    python scripts/blur_images.py --input photo.jpg --output photo_blurred.jpg

    # whole folder
    python scripts/blur_images.py --input photos/ --output photos_blurred/
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Images wider than this will be processed with a 2×2 tiled grid.
TILE_THRESHOLD_WIDTH = 1920
TILE_OVERLAP = 64   # pixel overlap between adjacent tiles (avoids edge artefacts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def apply_blur(bgr: np.ndarray, mask: np.ndarray, strength: int) -> np.ndarray:
    k = _odd(strength)
    blurred = cv2.GaussianBlur(bgr, (k, k), 0)
    return np.where(mask[:, :, None], blurred, bgr)


def collect_images(path: str):
    """Return list of (input_path, relative_name) pairs."""
    if os.path.isfile(path):
        return [(path, os.path.basename(path))]
    entries = []
    for fname in sorted(os.listdir(path)):
        if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
            entries.append((os.path.join(path, fname), fname))
    return entries


def _run_prompt_on_tile(processor, autocast, pil_tile: Image.Image, prompt: str) -> np.ndarray:
    """Return a boolean mask (H_tile, W_tile) for one tile and one prompt."""
    tw, th = pil_tile.size
    with autocast:
        state = processor.set_image(pil_tile)
        state = processor.set_text_prompt(state=state, prompt=prompt)
    masks = state.get("masks")
    tile_mask = np.zeros((th, tw), dtype=bool)
    if masks is not None and len(masks) > 0:
        for m in masks:
            m_np = m.squeeze(0).cpu().numpy().astype(bool)
            tile_mask |= m_np
    return tile_mask


def _segmentation_mask(processor, autocast, pil_img: Image.Image, prompt: str) -> tuple[np.ndarray, int]:
    """
    Return (full-image boolean mask, instance_count) for one prompt.
    Uses tiled inference when the image is wider than TILE_THRESHOLD_WIDTH.
    """
    w, h = pil_img.size

    if w <= TILE_THRESHOLD_WIDTH:
        # ── Full-image inference ────────────────────────────────────────────
        with autocast:
            state = processor.set_image(pil_img)
            state = processor.set_text_prompt(state=state, prompt=prompt)
        masks = state.get("masks")
        combined = np.zeros((h, w), dtype=bool)
        if masks:
            for m in masks:
                combined |= m.squeeze(0).cpu().numpy().astype(bool)
            return combined, len(masks)
        return combined, 0

    # ── Tiled inference (2×2 grid with overlap) ────────────────────────────
    combined = np.zeros((h, w), dtype=bool)
    total_instances = 0

    cols, rows = 2, 2
    tw = w // cols + TILE_OVERLAP
    th = h // rows + TILE_OVERLAP

    for row in range(rows):
        for col in range(cols):
            x0 = col * (w // cols)
            y0 = row * (h // rows)
            x1 = min(x0 + tw, w)
            y1 = min(y0 + th, h)

            tile = pil_img.crop((x0, y0, x1, y1))
            tile_mask = _run_prompt_on_tile(processor, autocast, tile, prompt)

            # Count non-empty tiles as instances (rough estimate)
            if tile_mask.any():
                total_instances += 1

            # Paste tile mask back at the correct position
            combined[y0:y1, x0:x1] |= tile_mask

    return combined, total_instances


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def load_model():
    print("Loading SAM3 image model...", flush=True)
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=0.5)
    return processor


def blur_image(
    processor,
    input_path: str,
    output_path: str,
    prompts: list[str],
    blur_strength: int,
) -> bool:
    pil_img = Image.open(input_path).convert("RGB")
    w, h = pil_img.size

    tiled = w > TILE_THRESHOLD_WIDTH
    if tiled:
        print(f"  [{w}×{h}] using 2×2 tiled inference", flush=True)

    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    combined_mask = np.zeros((h, w), dtype=bool)
    total_instances = 0

    for prompt in prompts:
        mask, n = _segmentation_mask(processor, autocast, pil_img, prompt)
        combined_mask |= mask
        total_instances += n

    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    if combined_mask.any():
        bgr = apply_blur(bgr, combined_mask, blur_strength)
        print(f"  blurred {total_instances} instance(s)", flush=True)
    else:
        print("  no targets found", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, bgr)
    return combined_mask.any()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(prog="blur_images.py")
    p.add_argument("--input", required=True, help="image file or folder of images")
    p.add_argument("--output", required=True, help="output file (single) or folder (batch)")
    p.add_argument("--blur-strength", type=int, default=80)
    p.add_argument("--prompts", nargs="+", default=["face", "license plate"])
    args = p.parse_args()

    entries = collect_images(args.input)
    if not entries:
        print(f"ERROR: no supported images found in {args.input}", file=sys.stderr)
        sys.exit(1)

    is_single = os.path.isfile(args.input)

    processor = load_model()

    n_blurred = 0
    for i, (src_path, fname) in enumerate(entries):
        dst_path = args.output if is_single else os.path.join(args.output, fname)
        print(f"[{i+1}/{len(entries)}] {fname}", flush=True)
        if blur_image(processor, src_path, dst_path, args.prompts, args.blur_strength):
            n_blurred += 1

    print(f"\nDone. {n_blurred}/{len(entries)} image(s) had targets blurred.")


if __name__ == "__main__":
    main()

