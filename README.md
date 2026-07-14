# Crack Segmentation Tool

Interactive crack segmentation and mask editing tool for preparing training labels from pavement or structure images.

This project combines classical image processing and deep learning model overlays to help create binary crack masks. It supports Sato ridge filtering, CLAHE preprocessing, HC-Unet++ / UNet++ model-assisted prediction, manual erase/draw editing, skeletonization, and final export as black-background / white-crack PNG masks.

## Features

- Sato ridge filter and CLAHE-based crack candidate extraction
- Optional model overlay using HC-Unet++ or UNet++
- UNet++ inference through `segmentation_models_pytorch`
- Adjustable model threshold, hysteresis threshold, morphology, and overlay alpha
- Manual mask editing
  - Erase/restore mode
  - Draw mode for adding thin white crack labels
  - Brush size control, including 1-pixel line drawing
- Optional skeletonization with thickness control
- Resume workflow from the latest saved mask
- Final output as binary grayscale PNG masks
  - `0`: background
  - `255`: crack

## Repository Structure

```text
.
├── sato_clahe_model_edit_add_skeleton_20260710.py  # Main labeling/editing GUI
├── requirements.txt                                # Python dependencies
├── crack_images/                                   # Local input images, not tracked
├── crack_masks/                                    # Local output masks, not tracked
└── model/                                          # Local model weights, not tracked
```


## Installation

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

For GPU inference or training, install the PyTorch build that matches your CUDA version before running the tool.

## Usage

1. Place source images in `crack_images/`.
2. Place model weights in `model/`.
3. Check and update paths near the top of `sato_clahe_model_edit_add_skeleton_20260710.py`:

```python
SRC_DIR = r"...\crack_images"
DST_DIR = r"...\crack_masks"
MODEL_PATH_HC = r"...\model\HC_unetpp_Quebec117_50"
MODEL_PATH_UNETPP = r"...\model\unetpp_204_crack_epoch_300.pth"
```

4. Run the labeling tool:

```bash
python sato_clahe_model_edit_add_skeleton_20260710.py
```

## Main Controls

- `S`: Save current mask and move to next unlabeled image
- `N`: Move to next unlabeled image
- `P`: Move to previous unlabeled image
- `Z`: Undo last manual edit
- `C`: Clear manual edit masks
- `M`: Toggle model overlay
- `D`: Toggle Erase / Draw edit mode
- `H`: Show help
- `Q` or `Esc`: Quit

Mouse behavior depends on edit mode:

- Erase mode
  - Left drag: erase from final mask
  - Right drag: restore erased area
- Draw mode
  - Left drag: draw white crack pixels
  - Right drag: remove manually drawn pixels

## Mask Format

Saved masks are binary grayscale PNG files:

```text
0   = background
255 = crack
```

This format is suitable for segmentation training scripts that read masks in grayscale and binarize them with a threshold such as `mask > 127`.

## Model Weights and Data

Model weights, raw images, and generated masks are intentionally excluded from Git tracking. They can be large and may exceed GitHub file size limits.

Recommended local-only paths:

- `model/`
- `crack_images/`
- `crack_masks/`

If model weights must be shared, use GitHub Releases, cloud storage, or Git LFS instead of committing them directly.

## Notes

- The default UNet++ model configuration uses `segmentation_models_pytorch.UnetPlusPlus` with a ResNet34 encoder.
- Manual drawing uses non-antialiased binary strokes so intermediate gray values are not introduced into training masks.
- Brush size `1` draws a true 1-pixel line.
