# PDFNet — Wire Segmentation

This README describes how to train and run PDFNet for overhead
wire segmentation used in the WIRED pipeline. PDFNet uses a
pseudo-depth map alongside RGB features through a Depth Integrity
Prior mechanism to accurately segment extremely thin wire
structures that standard segmentation models fail to detect.

---

## Overview

Two depth source configurations are supported:

| Configuration | Depth Source | Script |
|---|---|---|
| PDFNet + PatchFusion | PatchFusion-generated depth | `corrected_Mydataset.py` |
| PDFNet + Depth Anything V2 | DAM V2-generated depth | `mydatasetoriginal.py` |

### Results Comparison

| Model | $F_{\beta}^{max}$ | $F_{\beta}^{w}$ | $E_{\phi}^{m}$ | $S_{\alpha}$ | MAE |
|---|---|---|---|---|---|
| PDFNet + Depth Anything V2 | 0.746 | 0.655 | 0.901 | 0.834 | 0.011 |
| PDFNet + PatchFusion (Ours) | **0.785** | **0.689** | **0.902** | **0.841** | **0.009** |

PatchFusion consistently outperforms Depth Anything V2 as the
pseudo-depth source across all metrics.

---

## Dataset Structure

Organize your dataset in the following format before training:

```
PDFNet/
└── DATA/
    └── DIS-DATA/
        ├── DIS-TE1/
        │   ├── images/
        │   └── masks/
        ├── DIS-TE2/
        │   ├── images/
        │   └── masks/
        ├── DIS-TE3/
        │   ├── images/
        │   └── masks/
        ├── DIS-TE4/
        │   ├── images/
        │   └── masks/
        ├── DIS-TR/
        │   ├── images/
        │   ├── masks/
        │   ├── depth_large/     ← Depth Anything V2 depths
        │   └── depths/          ← PatchFusion depths
        └── DIS-VD/
            ├── images/
            └── masks/
```

---

## Environment Setup

```bash
conda activate PDFNet
```

---

## Training

### Option A — PDFNet + Depth Anything V2

**Step 1 — Generate depth maps using Depth Anything V2**

```bash
cd PDFNet/DAM_V2/
python Depth-prepare.py
```

Depth maps are saved to:
```
PDFNet/DAM_V2/DATA/DIS-DATA/DIS-TR/depth_large/
```

**Step 2 — Use the correct dataloader**

Use `mydatasetoriginal.py` in the dataloader folder.
This loads depth images from the Depth Anything V2 generated
depth path.

**Step 3 — Attach tmux and train**

```bash
tmux a -t pdfnet
```

```bash
CUDA_VISIBLE_DEVICES=1 python Train_PDFNet.py \
    --data_path /Vaibhav/shivasish1/PDFNet/DAM_V2/DATA/DIS-DATA
```

---

### Option B — PDFNet + PatchFusion (Recommended)

**Step 1 — Generate PatchFusion depth maps**

Follow the PatchFusion README to generate depth maps and place
them in:
```
PDFNet/DATA/DIS-DATA/DIS-TR/depths/
```

**Step 2 — Use the correct dataloader**

Use `corrected_Mydataset.py` in the dataloader folder.
This loads depth images from the PatchFusion generated
depth path.

**Step 3 — Attach tmux and train**

```bash
tmux a -t pdfnet1
```

```bash
python Train_PDFNet.py \
    --data_path /Vaibhav/shivasish1/PDFNet/DATA/DIS-DATA/
```

---

## Inference

**Step 1 — Set input and output paths in `Test.py`**

Open `wire_segmentation/PDFNet/metric_tools/Test.py` and set:

```python
test_dir  = '/path/to/your/input/images'
save_dir  = '/path/to/save/output/masks'
```

**Step 2 — Comment out evaluation line**

At the end of `Test.py`, comment out:

```python
# soc_metrics(file_name)
```

**Step 3 — Run inference**

```bash
python Test.py
```

Output binary wire segmentation masks will be saved to `save_dir`.

---

## Modified Files

| File | Description |
|---|---|
| `corrected_Mydataset.py` | Dataloader for PatchFusion-generated depth maps |
| `mydatasetoriginal.py` | Dataloader for Depth Anything V2-generated depth maps |
| `Depth-prepare.py` | Generates Depth Anything V2 depth maps for dataset |
| `Train_PDFNet.py` | Main training script |
| `Test.py` | Inference script |

---

## Key Notes

- Always match the dataloader file to the depth source being
  used — using the wrong dataloader will load incorrect depth
  paths and silently degrade performance
- Depth maps must be spatially aligned with their corresponding
  RGB images at the same resolution — verify this before training
- The `depth_large/` folder is used by Depth Anything V2 and
  the `depths/` folder is used by PatchFusion — do not mix them
- Comment out `soc_metrics(file_name)` during inference on
  custom data where no ground truth masks are available,
  otherwise the script will throw an error

---

## Output

PDFNet outputs a binary wire segmentation mask
$M_{wire} \in \{0,1\}^{H \times W}$ for each input RGB image,
where white pixels represent detected wire structures and black
pixels represent background. These masks are used downstream in
the WIRED pipeline for 3D back-projection and wire height
computation.

---

## Reference

```bibtex
@article{liu2025pdfnet,
  title   = {High-Precision Dichotomous Image Segmentation via
             Depth Integrity-Prior and Fine-Grained Patch
             Strategy},
  author  = {Liu, Xin and Fu, Keren and Zhao, Qijun},
  journal = {arXiv preprint arXiv:2503.06100},
  year    = {2025}
}
```
