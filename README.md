# Brain Tumor Segmentation — 3D U-Net on BraTS2020

A 3D deep learning pipeline for multi-class brain tumor segmentation using the BraTS2020 dataset.
The model is a 3D U-Net that segments three tumor sub-regions simultaneously from four-modality MRI volumes.

---

## Table of Contents

- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Key Design Choices](#key-design-choices)
- [Performance Optimizations](#performance-optimizations)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Configuration](#configuration)
- [Results](#results)
- [Dependencies](#dependencies)

---

## Dataset

**BraTS2020** (Brain Tumor Segmentation Challenge 2020)

- **369 patients**, each with 155 axial slices stored as individual HDF5 files
- **4 MRI modalities** per slice (channels): T1, T1ce (contrast-enhanced), T2, FLAIR
- **3 binary segmentation masks** per slice:
  - Class 0: Necrotic core / non-enhancing tumor (~3% of brain voxels)
  - Class 1: Peritumoral edema (~15% of brain voxels)
  - Class 2: Enhancing tumor — gadolinium uptake on T1ce (~1% of brain voxels)
- Raw data format: `volume_{id}_slice_{s}.h5` with `image (240, 240, 4)` and `mask (240, 240, 3)`

The dataset can be downloaded via [Kaggle](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data).

---

## Model Architecture

**3D U-Net** — the standard architecture for volumetric medical image segmentation.

```
Input  : (B, 4, D, H, W)        four MRI modalities as input channels

enc0   : DoubleConv3D  4  → 32   (D,    H,    W   )   full resolution
enc1   : Down3D  32  → 64        (D/2,  H/2,  W/2 )
enc2   : Down3D  64  → 128       (D/4,  H/4,  W/4 )
enc3   : Down3D  128 → 256       (D/8,  H/8,  W/8 )
bn     : Down3D  256 → 512       (D/16, H/16, W/16)   bottleneck

dec3   : Up3D + skip from enc3
dec2   : Up3D + skip from enc2
dec1   : Up3D + skip from enc1
dec0   : Up3D + skip from enc0

head   : Conv3d(32, 3, 1x1x1)   one logit per class per voxel

Output : (B, 3, D, H, W)        raw logits -> Sigmoid -> binary masks
```

**Building blocks:**
- `DoubleConv3D`: two `Conv3d(3×3×3)` + `InstanceNorm3d` + `LeakyReLU(0.01)` units
- `Down3D`: `MaxPool3d(2)` followed by `DoubleConv3D`
- `Up3D`: `ConvTranspose3d(2×2×2)` (learnable upsampling), concatenate skip, `DoubleConv3D`

**Why InstanceNorm instead of BatchNorm?**
MRI intensities vary dramatically between patients and scanners. InstanceNorm normalizes each
sample independently, keeping training stable even with `batch_size=1`.

**Parameter count:** approximately 11 million (with `FEATURES=[32, 64, 128, 256]`)

---

## Key Design Choices

### Patch-based Training
Full BraTS volumes are `(4, 155, 240, 240)` — too large for GPU memory in one pass.
Training uses random patches of size `(64, 160, 160)` extracted on-the-fly.

**Tumor-centered sampling (60%):** 60% of patches are centered on a random tumor voxel.
This guarantees that the model sees positive examples at every step, since tumor voxels
represent less than 1% of the total volume if sampled uniformly.

### Per-Channel Z-Score Normalization
Applied only to non-zero (brain) voxels to avoid inflating statistics with background.
Background pixels are restored to exactly 0 after normalization so the network can
still detect brain/background boundaries.

```python
mu  = ch[brain_mask].mean()
std = ch[brain_mask].std() + 1e-8
normalized = (ch - mu) / std
normalized[~brain_mask] = 0.0
```

### Weighted Dice Loss + Focal BCE Loss
**Weighted Dice** — handles class imbalance at the region level:
- Necrosis weight: 1.5 (small, irregular)
- Edema weight: 1.0 (large, easier to detect)
- Enhancing tumor weight: 2.0 (tiny, clinically critical)

**Focal BCE** (γ=2) — focuses gradient on hard, ambiguous voxels near tumor boundaries.

Combined loss: `0.5 × WeightedDice + 0.5 × FocalBCE`

### Gradient Accumulation
`ACCUM_STEPS=2` with `BATCH_SIZE=2` gives an **effective batch size of 4**
without additional VRAM cost — the gradients from 2 forward passes are summed before
each optimizer step.

### LR Schedule: Warmup + Cosine Annealing
- Linear warmup for 5 epochs: prevents large gradient updates when weights are random
- Cosine annealing for the remaining epochs: decays LR smoothly to `LR × 0.01`

### Sliding Window Inference
At evaluation time, the model cannot process a full `(4, 155, 240, 240)` volume at once.
A sliding window with 50% overlap tiles the volume into patches, runs inference on each,
and averages the predicted probabilities where patches overlap. This removes boundary
artifacts and improves segmentation quality.

---

## Performance Optimizations

### Pre-Cache System (critical on Windows)
Windows Defender scans each file on first read. A single patient volume consists of
**155 HDF5 files**, leading to 15–25 seconds of antivirus overhead per volume.

The pre-cache step (run once, ~10–15 minutes) converts each patient into two NumPy arrays:

| File | Shape | dtype | Size |
|------|-------|-------|------|
| `v{id}_img.npy` | `(4, 155, 240, 240)` | float16 | ~72 MB |
| `v{id}_mask.npy` | `(3, 155, 240, 240)` | uint8 | ~27 MB |

| Loading method | Time per volume |
|---------------|-----------------|
| 155 × H5 (cold, antivirus active) | 15–25 s |
| 155 × H5 (warm filesystem cache) | ~2.3 s |
| Single .npy file | ~0.05 s |
| **Speedup** | **~400×** |

### RAM LRU Cache
An `OrderedDict`-based LRU cache keeps the most recently used volumes in RAM.
When the same patient is sampled multiple times within an epoch, no disk I/O occurs.

### Automatic Mixed Precision (AMP)
`torch.cuda.amp.autocast` runs convolutions in float16 on the GPU, halving VRAM usage
and increasing throughput via tensor cores. Master weights stay in float32.

### Windows Multiprocessing Fix
`DataLoader(num_workers=0)` avoids multiprocessing deadlocks that occur when using
`num_workers > 0` inside Jupyter on Windows.

---

## Project Structure

```
CancerDetection/
├── tumor_detection_3d.ipynb   main training notebook (19 cells)
├── dataset.ipynb              dataset download via kagglehub
├── README.md                  this file
├── checkpoints/
│   └── best_model.pth         saved when val Dice improves
├── volume_cache_240/          pre-cached .npy volumes (240x240, ~35 GB)
├── sample_slice.png           visualization of one MRI slice + masks
├── training_curves.png        loss and dice curves over epochs
├── predictions.png            GT vs predicted masks (patch-level)
└── 3views_segmentation.png    axial / sagittal / coronal views
```

**Notebook cells:**

| Cell | Content |
|------|---------|
| 1 | Imports |
| 2 | Configuration (hyperparameters, paths) |
| 3 | Dataset exploration |
| 4 | Slice visualization |
| 5 | Pre-cache system (H5 → NPY) |
| 6 | Cache vs H5 loading benchmark |
| 7 | `BraTS3DDataset` class |
| 8 | Train / Val / Test splits and DataLoaders |
| 9 | `UNet3D` architecture |
| 10 | Loss functions and Dice metric |
| 11 | `train_epoch` / `eval_epoch` functions |
| 12 | Training loop with gradient accumulation + AMP |
| 13 | Training curves (loss, mean Dice, per-class Dice) |
| 14 | Test set evaluation |
| 15 | Qualitative prediction visualization |
| 16 | Sliding window full-volume inference |
| 17 | Three-plane anatomical view (axial / sagittal / coronal) |
| 18 | Architecture summary (markdown) |

---

## How to Run

### 1. Install dependencies
```bash
pip install torch torchvision h5py numpy pandas matplotlib scipy tqdm
```

### 2. Download the dataset
Open `dataset.ipynb` and run the kagglehub download cell, or run:
```bash
python -c "import kagglehub; kagglehub.dataset_download('awsaf49/brats2020-training-data')"
```

### 3. Update `DATA_DIR` in cell 2
Set `DATA_DIR` to the path where the `.h5` files were downloaded.

### 4. Run cells in order
- **Cell 5** (pre-cache): run once, takes ~10–15 minutes, creates `volume_cache_240/`
- **Cell 12** (training): ~1–2 minutes per epoch on GPU, longer on CPU
- **Cells 13–17**: evaluation and visualization (requires a trained checkpoint)

---

## Configuration

Key hyperparameters in **cell 2**:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `PATCH_SIZE` | `(64, 160, 160)` | Training patch size (D, H, W) |
| `FEATURES` | `[32, 64, 128, 256]` | Feature map counts per encoder level |
| `BATCH_SIZE` | `2` | Samples per forward pass |
| `ACCUM_STEPS` | `2` | Gradient accumulation steps (effective batch = 4) |
| `LR` | `2e-4` | Peak learning rate after warmup |
| `WARMUP_EP` | `5` | Linear warmup epochs |
| `EPOCHS` | `50` | Total training epochs |
| `CACHE_H/W` | `240` | Cache resolution (full native resolution) |
| `CLASS_WEIGHTS` | `[1.5, 1.0, 2.0]` | Dice loss weights (necrosis, edema, enhancing) |
| `NUM_WORKERS` | `0` | Keep at 0 on Windows to avoid deadlocks |

---

## Results

The model is evaluated using the **Dice Similarity Coefficient (DSC)** per tumor sub-region,
which is the standard metric in the BraTS challenge.

A Dice score of 1.0 means perfect overlap; 0.0 means no overlap.
The three classes differ significantly in difficulty:

- **Edema** is the largest region and easiest to segment (typically Dice > 0.80)
- **Necrosis** is irregular and variable in size (typically Dice 0.60–0.80)
- **Enhancing tumor** is the smallest and clinically most important (typically Dice 0.70–0.85)

Training curves are saved to `training_curves.png` after training completes.

---

## Dependencies

| Library | Purpose |
|---------|---------|
| PyTorch | Model training and inference |
| h5py | Reading BraTS HDF5 files |
| NumPy | Array operations and pre-cache |
| SciPy | Spatial resampling (`ndimage.zoom`) |
| Pandas | Reading dataset metadata CSVs |
| Matplotlib | Visualization |
| tqdm | Progress bars |

Python 3.10+ and CUDA 11.x+ recommended for GPU training.
The code falls back to CPU automatically if no GPU is detected.
