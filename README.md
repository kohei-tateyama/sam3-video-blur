# sam3-video-blur

Blur faces and number plates in video using [SAM3](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/).

---

## Setup

**1. Create the conda environment**
```bash
bash setup_conda_env.sh
```

**2. Log in to HuggingFace** (needed to download the model checkpoint)
```bash
conda run -n sam3 huggingface-cli login
```

---

## Usage

### Blur all images in a folder
```bash
bash scripts/blur_images.sh \
    --input  photos/ \
    --output photos_blurred/
```

### Blur a single image
```bash
bash scripts/blur_images.sh \
    --input  photo.jpg \
    --output photo_blurred.jpg
```

### Blur a short video clip (≤ 10 s)
```bash
bash scripts/blur_full_video.sh \
    --input  footage.mp4 \
    --output footage_blurred.mp4
```

### Blur a longer video
Split into 5-second chunks first, then blur each chunk:
```bash
bash scripts/split_video.sh --input footage.mp4

for f in outputs/chunks/*.mp4; do
    bash scripts/blur_full_video.sh \
        --input  "$f" \
        --output "outputs/blurred/$(basename "$f")"
done
```

### Options

| Script | Options |
|---|---|
| `blur_images.sh` | `--input` (file or folder), `--output` (file or folder), `--blur-strength` (default 80), `--prompts` (default: `face` `license plate`) |
| `blur_full_video.sh` | `--input`, `--output`, `--blur-strength` (default 80), `--prompts` |
| `blur_windowed_video.sh` | same as above + `--window-seconds` (default 5) |
| `split_video.sh` | `--input`, `--output-dir` (default `./outputs/chunks`), `--chunk-seconds` (default 5) |

**Examples:**
```bash
# stronger blur, custom prompts
bash scripts/blur_images.sh \
    --input photos/ --output photos_blurred/ \
    --blur-strength 151 \
    --prompts "face" "number plate" "hand"

# split video into 10-second chunks
bash scripts/split_video.sh --input footage.mp4 --chunk-seconds 10
```

---

## GPU requirements

- NVIDIA GPU with ≥ 24 GB VRAM (tested on RTX 6000)
- CUDA 12.6+
- 5-second 4K clips use ~1.8 GB VRAM. If you hit OOM, reduce `--chunk-seconds`.
- Turing GPUs (RTX 6000, etc.) are supported but run without Flash Attention — expect ~3–7 min per 5 s clip. Ampere+ is faster.

---

## License

[Apache 2.0](LICENSE) — based on [facebookresearch/sam3](https://github.com/facebookresearch/sam3).
