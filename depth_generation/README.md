# PatchFusion — Depth Map Generation

This README describes how to run PatchFusion for high-resolution
depth map generation used in the WIRED wire height estimation
pipeline.

---

## Overview

PatchFusion generates high-resolution pseudo-depth maps from
single RGB images. In the WIRED pipeline, these depth maps serve
as auxiliary input to PDFNet's depth integrity-prior mechanism
for thin wire segmentation. PatchFusion is used in preference
over Depth Anything V2 and V3 because it better preserves local
depth contrast for sub-pixel-width wire structures.

---

## Modified Files

The following files were modified from the original PatchFusion
repository to fix output resolution and pipeline compatibility:

| File | Description |
|---|---|
| `corrected_baseline_pretrain.py` | Baseline pretraining fixes |
| `corrected_patchfusion.py` | Core model corrections |
| `corrected_test.py` | Test script fixes |
| `estimator/tester/tester.py` | Output resolution fix |
| `test_single_forward.py` | Single image forward pass fix |
| `test.py` | Main test entry point |
| `corrected_tester.py` | Ensures output resolution matches input image resolution |

> **Key fix:** `corrected_tester.py` ensures the depth map output
> has the same spatial resolution as the input RGB image, which is
> critical for pixel-accurate wire-to-ground matching in the
> height estimation stage.

---

## Environment Setup


### Step 1 — Navigate and activate environment

```bash
cd PatchFusion/
source patchfusion_env/bin/activate
```

### Step 2 — Set Python path

```bash
export PYTHONPATH="${PYTHONPATH}:/Vaibhav/shivasish1/sam2/PatchFusion"
export PYTHONPATH="${PYTHONPATH}:/Vaibhav/shivasish1/sam2/PatchFusion/external"
```
# Pretrained Model Weights

Download pretrained weights from:

* Coarse Pretrain
  https://huggingface.co/zhyever/PatchFusion/blob/main/depthanything_vitl_u4k/coarse_pretrain/checkpoint_24.pth

* Fine Pretrain
  https://huggingface.co/zhyever/PatchFusion/blob/main/depthanything_vitl_u4k/fine_pretrain/checkpoint_24.pth

* PatchFusion
  https://huggingface.co/zhyever/PatchFusion/blob/main/depthanything_vitl_u4k/patchfusion/checkpoint_16.pth

Place downloaded files inside:

```text
depth_generation/PatchFusion/pretrained_models/
```

---

## Running Depth Generation

```bash
CUDA_VISIBLE_DEVICES=1 python ./tools/test.py \
    configs/patchfusion_depthanything/depthanything_vitl_patchfusion_u4k.py \
    --ckp-path /Vaibhav/shivasish1/sam2/PatchFusion/pretrained_models/patchfusion.pth \
    --cai-mode r64 \
    --cfg-option \
        general_dataloader.dataset.rgb_image_dir='/Vaibhav/shivasish1/sam2/outdoor_dataset' \
    --test-type general \
    --save
```

### Arguments

| Argument | Description |
|---|---|
| `configs/...u4k.py` | Config file for DepthAnything ViT-L + PatchFusion |
| `--ckp-path` | Path to pretrained PatchFusion checkpoint |
| `--cai-mode r64` | Patch sampling mode — r64 gives best quality |
| `--cfg-option` | Override dataset RGB image directory path |
| `--test-type general` | General inference mode for custom images |
| `--save` | Save output depth maps to disk |

---

## Input / Output

| | Details |
|---|---|
| **Input** | RGB images in `outdoor_dataset/` directory |
| **Output** | 16-bit depth maps saved to output directory |
| **Resolution** | Same as input image (fixed by `corrected_tester.py`) |
| **Format** | PNG 16-bit depth maps, values in millimeters |

---

## Depth Map Usage in WIRED Pipeline

After generating depth maps with PatchFusion:

1. Depth maps are loaded and converted from 16-bit to meters
2. Each pixel $(u, v)$ with valid depth $Z > 0$ is back-projected
   to 3D using pinhole camera model
3. Wire and ground pixels from PDFNet and SegFormer masks are
   lifted to 3D point clouds
4. KD-Tree ground matching computes wire height as
   $h = |Z_w - Z_g^*|$

---

## Pretrained Model

Download the pretrained PatchFusion checkpoint:

```bash
mkdir -p pretrained_models
# Place patchfusion.pth in pretrained_models/
```

> The checkpoint used in WIRED:
> `/Vaibhav/shivasish1/sam2/PatchFusion/pretrained_models/patchfusion.pth`

---

## Notes

- Always verify output depth map resolution matches input RGB
  resolution before running height estimation
- Use `CUDA_VISIBLE_DEVICES=1` to select the correct GPU if
  multiple GPUs are available
- The `r64` CAI mode provides the best depth quality for
  high-resolution outdoor images at the cost of higher
  inference time — use `r32` for faster inference if needed
- Invalid depth regions (where $Z = 0$) are automatically
  discarded in the height estimation stage

---

## Reference

```bibtex
@inproceedings{li2024patchfusion,
  title     = {PatchFusion: An End-to-End Tile-Based Framework
               for High-Resolution Monocular Metric Depth
               Estimation},
  author    = {Li, Zhenyu and Bhat, Shariq Farooq and Wonka, Peter},
  booktitle = {CVPR},
  year      = {2024}
}
```
