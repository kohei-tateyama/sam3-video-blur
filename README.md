# Video Face & Number Plate Blurring

Automatically blur faces and number plates in a video using
[SAM3](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)
(Segment Anything with Concepts).

SAM3 is an open-vocabulary segmentation model.  A text prompt such as `"face"`
or `"license plate"` is enough to detect, segment, and track every matching
instance across the entire video.  A Gaussian blur is then applied to each
region in every frame, producing a privacy-safe output video.

---

## Repository layout

```
.
├── sam3/                        # SAM3 model source code
├── scripts/
│   ├── blur_full_video.py       # full-session blurring logic
│   ├── blur_video.py            # windowed blurring logic
│   ├── split_video.py           # video chunking logic
│   ├── blur_full_video.sh       # ← run this for short clips / pre-split chunks
│   ├── blur_windowed_video.sh   # ← run this for longer videos
│   └── split_video.sh           # ← run this to pre-split a long video
├── setup_conda_env.sh           # one-shot environment setup
├── pyproject.toml
└── LICENSE
```

---

## Requirements

| Requirement | Version |
|---|---|
| OS | Linux |
| Python | 3.12+ |
| PyTorch | 2.10.0+ |
| CUDA driver | 12.6+ |
| GPU VRAM | ≥ 24 GB recommended for 4K video |
| GPU arch | Turing (7.5) or newer — Ampere+ preferred for Flash Attention |

A [HuggingFace account](https://huggingface.co) is required to download the
SAM3 model checkpoint (free, one-time).

---

## Installation

### 1. Create the conda environment

```bash
bash setup_conda_env.sh          # creates an env named "sam3"
```

The script installs:
- Python 3.12
- PyTorch 2.10.0 with CUDA 12.8 (`cu128` wheel index)
- SAM3 package in editable mode with all inference dependencies
- `einops` and `ninja` for faster kernel compilation

### 2. Authenticate with HuggingFace

The SAM3 checkpoint is gated.  Run this once:

```bash
conda run -n sam3 huggingface-cli login
```

Generate a read-only access token at <https://huggingface.co/settings/tokens>
and paste it when prompted.  The checkpoint (~3.5 GB) is downloaded
automatically on first inference.

---

## Recommended workflow for long videos

Long videos (e.g. 4K, > 10 s) will run out of GPU memory if processed in a
single session.  The recommended approach is to split first, then blur each
chunk:

### Step 1 — split the video into 5-second chunks

Edit `scripts/split_video.sh` to set `INPUT` and `OUTPUT_DIR`, then:

```bash
bash scripts/split_video.sh
```

Chunks are written to `OUTPUT_DIR` as `<basename>_part001.mp4`,
`<basename>_part002.mp4`, etc.

### Step 2 — blur each chunk

Edit `scripts/blur_full_video.sh` to set `INPUT`, `OUTPUT`, and optionally
`BLUR_STRENGTH` and `PROMPTS`, then:

```bash
bash scripts/blur_full_video.sh
```

To process all chunks in a loop:

```bash
for f in outputs/chunks/*.mp4; do
    name=$(basename "$f")
    sed -i \
        "s|^INPUT=.*|INPUT=\"${f}\"|; s|^OUTPUT=.*|OUTPUT=\"./outputs/blurred/${name}\"|" \
        scripts/blur_full_video.sh
    bash scripts/blur_full_video.sh
done
```

---

## Scripts reference

### `scripts/blur_full_video.sh` — full-session blurring

Opens the entire video in a single SAM3 session.  Best quality; best suited
for short clips (≤ 5–10 s at 4K on a 24 GB GPU).

**Configure in the script:**

| Variable | Default | Description |
|---|---|---|
| `INPUT` | *(required)* | Path to the input video |
| `OUTPUT` | *(required)* | Path for the output video |
| `BLUR_STRENGTH` | `80` | Gaussian blur kernel size (larger = heavier) |
| `PROMPTS` | `face` `license plate` | Objects to blur |
| `GPUS` | *(all)* | Uncomment and set to pin to specific GPUs |

### `scripts/blur_windowed_video.sh` — windowed blurring

Splits the video into short windows internally and runs SAM3 tracking on each
window separately.  Can handle videos of any length, but tracking resets at
each window boundary.

**Additional variable:**

| Variable | Default | Description |
|---|---|---|
| `WINDOW_SECONDS` | `5` | Length of each tracking window in seconds |

### `scripts/split_video.sh` — video splitter

Splits a video into fixed-length MP4 chunks using OpenCV.

| Variable | Default | Description |
|---|---|---|
| `INPUT` | *(required)* | Path to the input video |
| `OUTPUT_DIR` | `./outputs/chunks` | Folder for output chunks |
| `CHUNK_SECONDS` | `5` | Length of each chunk in seconds |

---

## How it works

1. **Split** *(recommended pre-step)* — long videos are split into short chunks
   so each chunk fits in GPU memory.
2. **Load** — all frames of the chunk are decoded from MP4 with OpenCV and
   written to a temporary JPEG folder (the format SAM3 requires).
3. **Segment & track** — for each text prompt:
   - `add_prompt` runs full segmentation on frame 0 to find all instances.
   - `propagate_in_video` tracks those masks forward through every frame using
     SAM3's memory encoder — no re-segmentation per frame.
   - Binary masks are collected per frame.
4. **Merge** — masks from all prompts are OR-ed together into one boolean mask
   per frame.
5. **Blur** — Gaussian blur is applied to every pixel covered by the mask;
   uncovered pixels are left unchanged.
6. **Save** — output frames are re-encoded to an MP4 file.

---

## Tips

- **Blur strength**: `80` is a good starting point for strong anonymisation.
  Increase further (e.g. `151`) for maximum blur, or decrease for a subtler
  effect.  The value is automatically rounded up to the nearest odd number.
- **Custom prompts**: any English description works — e.g. `"face"`,
  `"license plate"`, `"number plate"`, `"hand"`, `"tattoo"`.  Results from
  multiple prompts are merged, so overlapping regions are only blurred once.
- **Output format**: the output video preserves the original FPS and
  resolution exactly.  The codec is `mp4v` (widely compatible).
- **GPU memory**: 4K 5-second clips use approximately 1.8 GB of GPU state.
  If you still hit OOM errors, reduce `CHUNK_SECONDS` in `split_video.sh`.
- **No Flash Attention on Turing GPUs**: Turing (compute 7.5, e.g. RTX 6000)
  is fully supported but runs without Flash Attention.  Expect ~3–7 minutes
  per 5-second 4K chunk.  Ampere or newer GPUs will be significantly faster.

---

## License

This project uses SAM3 which is released under the
[Apache 2.0 License](LICENSE).
