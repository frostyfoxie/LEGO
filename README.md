# LEGO: Deep Spatial Restoration Network

| original image | degraded image | lego's output |
| :--- | :--- | :--- |
| ![sample](images/original.png) | ![sample](images/gaussian_blur_input.png) | ![sample](images/gaussian_blur_LEGO.png) |

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-Supported-76B900.svg?style=flat&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Interim_Checkpoint_(Stage_2)-orange.svg)]()

LEGO (**DeepSpatialRestorationNet**) is a residual image restoration network designed to remove composite, real-world image degradations (deblurring, denoising, demosaicing, and JPEG deblocking) in a single feed-forward pass.

It couples **Modulated Deformable Convolutions (DCNv2)** for spatially adaptive geometric feature alignment with **Restormer-style Transposed Cross-Covariance Attention Blocks** for long-range contextual aggregation.

---

## ⚠️ Transparent Training Status & Disclosures

We believe in scientific transparency. The released checkpoint (`LEGO_latest.pth`) is an **interim model** and is **not fully converged**. Please review the planned versus executed training history before interpreting any benchmark claim.

| Training Stage | Objective & Data | Planned Budget | Actual Executed Status | Checkpoint Export |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1: Pixel Pretraining** | DIV2K (800 HR images), Charbonnier + FFT + Edge Loss. | 100 Epochs | **Halted at Epoch 51** (~51% complete) | Transferred to Stage 2 |
| **Stage 2: Generalized Diversity** | Streamed COCO + Flickr30k mixed with 50% DIV2K replay, L2-SP anchor. | 60 Epochs | **Halted at Epoch 17** (~28% complete) | **`LEGO_latest.pth`** |
| **Stage 3: Perceptual GAN Fine-Tuning** | VGG19 Perceptual Loss + Spectral Norm U-Net Discriminator. | 80 Epochs | **Not Executed (Pending)** | N/A |

### What this means for performance:
- **Strong Capabilities**: Good structural denoising, JPEG artifact suppression, and moderate motion/defocus deblurring. Color constancy and high-frequency structure are preserved comparatively well.
- **Current Limitations**: Because Stage 3 (adversarial perceptual loss) was not executed, the model functions as a primarily L1/Charbonnier/frequency-regressed network. Fine textures and adversarially challenging cases remain limited.

---

## 🏛️ Model Architecture

```text
Input Image (RGB, [-1, 1])
  │
  ├── Initial Feature Extraction: Conv2d(3, 128, kernel=3, pad=1)
  │
  ├── [Encoder Level 1] (128 Channels)
  │    ├── Modulated Deformable Conv (DCNv2: offset_conv + mask_conv + regular_conv)
  │    └── Transposed Channel Attention (4 Heads, depthwise 3x3 projection, temperature scaling)
  │
  ├── Downsampling 1: Conv2d(128, 256, kernel=4, stride=2, pad=1)
  │
  ├── [Encoder Level 2] (256 Channels)
  │    ├── Modulated Deformable Conv
  │    └── Transposed Channel Attention
  │
  ├── Downsampling 2: Conv2d(256, 512, kernel=4, stride=2, pad=1)
  │
  ├── [Bottleneck] (512 Channels)
  │    └── 4x Cascaded Transposed Channel Attention Blocks
  │
  ├── Upsampling 2: ConvTranspose2d(512, 256, kernel=2, stride=2)
  │
  ├── [Decoder Level 2] (Concatenation with Encoder 2 -> 512 Channels -> 256 Channels)
  │    ├── Modulated Deformable Conv
  │    └── Transposed Channel Attention
  │
  ├── Upsampling 1: ConvTranspose2d(256, 128, kernel=2, stride=2)
  │
  ├── [Decoder Level 1] (Concatenation with Encoder 1 -> 256 Channels -> 128 Channels)
  │    ├── Modulated Deformable Conv
  │    └── Transposed Channel Attention
  │
  ├── Final Projection: Conv2d(128, 3, kernel=3, pad=1)
  │
Residual Addition: Output = Input + Final_Residual
```

---

## 📊 Benchmark & Empirical Evaluation

The benchmark data below is taken directly from the evaluation report for the `LEGO_latest.pth` checkpoint.

### Family-level summary

| Degradation family | ΔPSNR (dB) | ΔSSIM | ΔMAE | Helped | Hurt | Tests | Verdict |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| TRAINING-MATCHED CONTROL | +2.8854 | +0.0320 | -0.0176 | 3 | 0 | 3 | GENERALLY HELPFUL |
| GAUSSIAN BLUR | -2.7273 | -0.0057 | +0.0056 | 0 | 1 | 1 | GENERALLY HARMFUL |
| STRONG GAUSSIAN BLUR | +4.2706 | +0.0365 | -0.0192 | 1 | 0 | 1 | GENERALLY HELPFUL |
| MOTION BLUR | -1.5651 | -0.0035 | +0.0038 | 0 | 1 | 2 | GENERALLY HARMFUL |
| DEFOCUS / OUT-OF-FOCUS | +5.2696 | +0.0199 | -0.0152 | 1 | 0 | 1 | GENERALLY HELPFUL |
| SENSOR NOISE | +0.7134 | +0.0084 | -0.0131 | 1 | 0 | 1 | GENERALLY HELPFUL |
| JPEG COMPRESSION | -8.1109 | -0.0201 | +0.0178 | 0 | 1 | 1 | GENERALLY HARMFUL |
| BLUR + SENSOR NOISE | +3.3094 | +0.0147 | -0.0166 | 1 | 0 | 1 | GENERALLY HELPFUL |
| BLUR + NOISE + JPEG | -2.2435 | -0.0042 | +0.0028 | 0 | 1 | 1 | GENERALLY HARMFUL |
| DOWNSAMPLE + BLUR + NOISE + JPEG | +0.4543 | +0.0021 | -0.0018 | 0 | 0 | 1 | MIXED |

### Unseen generalization

| Metric | Value |
| :--- | ---: |
| mean ΔPSNR (dB) | -0.2195 |
| mean ΔSSIM | +0.00445 |
| helped rate | 0.4000 |
| hurt rate | 0.4000 |
| verdict | LIMITED / MIXED GENERALIZATION |

---

## 🚀 Quickstart & Inference

### 1. Installation

```bash
git clone https://github.com/your-username/lego-restoration.git
cd lego-restoration
pip install -r requirements.txt
```

### 2. Download Checkpoint
Download `LEGO_latest.pth` and place it in the `checkpoints/` directory.

### 3. Run Inference on a Single Image
```bash
python scripts/infer.py \
  --checkpoint checkpoints/LEGO_latest.pth \
  --input path/to/blurry_image.png \
  --output path/to/restored_image.png \
  --tile-size 256 \
  --overlap 64 \
  --fp16
```

### 4. Run Batch Inference on a Folder
```bash
python scripts/infer.py \
  --checkpoint checkpoints/LEGO_latest.pth \
  --input path/to/input_folder/ \
  --output path/to/output_folder/ \
  --fp16
```
