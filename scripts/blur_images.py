#!/usr/bin/env python3
"""
Blur faces and number plates in images using SAM3.

Works on a single image file or all images in a folder.

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

    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    # Encode image once; re-use backbone features for each prompt
    with autocast:
        base_state = processor.set_image(pil_img)

    combined_mask = np.zeros((h, w), dtype=bool)
    total_instances = 0
    for prompt in prompts:
        # set_text_prompt replaces the previous text prompt in-place,
        # so start from a fresh copy of the image state each time
        state = copy.copy(base_state)
        with autocast:
            state = processor.set_text_prompt(state=state, prompt=prompt)
        masks = state.get("masks")
        if masks is None or len(masks) == 0:
            continue
        total_instances += len(masks)
        # masks: (N, 1, H, W) boolean tensor
        for m in masks:
            m_np = m.squeeze(0).cpu().numpy().astype(bool)
            combined_mask |= m_np

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
        if is_single:
            dst_path = args.output
        else:
            dst_path = os.path.join(args.output, fname)

        print(f"[{i+1}/{len(entries)}] {fname}", flush=True)
        if blur_image(processor, src_path, dst_path, args.prompts, args.blur_strength):
            n_blurred += 1

    print(f"\nDone. {n_blurred}/{len(entries)} image(s) had targets blurred.")


if __name__ == "__main__":
    main()
