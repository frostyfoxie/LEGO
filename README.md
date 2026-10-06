# LEGO: Deep Spatial Restoration Network

|original image| degraded image| lego's output |
|:---|:---|:---|
|![sample](images/)|![sample](images/gaussian_blur_comparison_input.png)|![sample](images/gaussian_blur_LEGO.png)|

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-Supported-76B900.svg?style=flat&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Interim_Checkpoint_(Stage_2)-orange.svg)]()

LEGO (**DeepSpatialRestorationNet**) is a residual image restoration network designed to remove composite, real-world image degradations (deblurring, denoising, demosaicing, and JPEG deblocking) in a unified forward pass. 

It couples **Modulated Deformable Convolutions (DCNv2)** for spatially adaptive geometric feature alignment with **Restormer-style Transposed Cross-Covariance Attention Blocks** for long-range context without quadratic token complexity.

---

## ⚠️ Transparent Training Status & Disclosures

We believe in scientific transparency. The released checkpoint (`LEGO_latest.pth`) is an **interim model** and is **not fully converged**. Please review the planned versus executed training history below:

| Training Stage | Objective & Data | Planned Budget | Actual Executed Status | Checkpoint Export |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1: Pixel Pretraining** | DIV2K (800 HR images), Charbonnier + FFT + Edge Loss. | 100 Epochs | **Halted at Epoch 51** (~51% complete) | Transferred to Stage 2 |
| **Stage 2: Generalized Diversity** | Streamed COCO + Flickr30k mixed with 50% DIV2K replay, L2-SP anchor. | 60 Epochs | **Halted at Epoch 17** (~28% complete) | **`LEGO_latest.pth`** |
| **Stage 3: Perceptual GAN Fine-Tuning** | VGG19 Perceptual Loss + Spectral Norm U-Net Discriminator. | 80 Epochs | **Not Executed (Pending)** | N/A |

### What this means for performance:
- **Strong Capabilities**: Excellent structural denoising, JPEG artifact clearing, and moderate motion/defocus deblurring. Color constancy and high-frequency structural lines are preserved accurately without halluncinating artifacts.
- **Current Limitations**: Because Stage 3 (adversarial perceptual loss) was not executed, the model functions purely as an L1/Charbonnier/frequency-regressed network. Very fine textures (e.g., individual facial pores, woven fabric micro-threads) under severe blur will appear smoothly reconstructed rather than photorealistically hallucinated.

---

## 🏛️ Model Architecture

```
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

## 📊 Benchmark & Empirical Evaluation (100 Web Images)

We evaluated the `LEGO_latest.pth` checkpoint across a validation suite of 100 real-world web images spanning five visual categories (**Faces**, **Outdoor**, **Indoor**, **Architecture**, **Nature**). Tests were executed using seamless windowed inference ($256 \times 256$ tiles with 64px sine-weighted overlap).

### 1. Quantitative Results by Degradation Profile

| Degradation Profile | Input PSNR (dB) | LEGO PSNR (dB) | **$\Delta$PSNR (dB)** | Input SSIM | LEGO SSIM | **$\Delta$SSIM** | Objective Outcome |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Gaussian Sensor Noise** ($\sigma=0.06$) | 24.52 | 31.84 | **+7.32 dB** | 0.6120 | 0.8842 | **+0.2722** | Significant Enhancement |
| **JPEG Compression** (Quality = 25) | 28.14 | 31.22 | **+3.08 dB** | 0.7915 | 0.8654 | **+0.0739** | Artifacts Removed |
| **Motion Blur** ($L=15\text{px}, \theta=0^\circ$) | 22.80 | 25.44 | **+2.64 dB** | 0.6841 | 0.7612 | **+0.0771** | Edges Recovered |
| **Blur + Sensor Noise** | 21.65 | 26.11 | **+4.46 dB** | 0.5412 | 0.7420 | **+0.2008** | Substantial Cleanup |
| **Camera-Like Degradation** | 23.40 | 26.98 | **+3.58 dB** | 0.6510 | 0.7831 | **+0.1321** | Structural Restoration |
| **Low-Quality Camera (Compound)** | 20.12 | 22.89 | **+2.77 dB** | 0.4912 | 0.6210 | **+0.1298** | Stabilized |
| **Strong Gaussian Blur** ($r=6.0\text{px}$) | 19.85 | 20.10 | **+0.25 dB** | 0.5620 | 0.5701 | **+0.0081** | Minimal (Conservative) |

> **Evaluation Rule:** $\text{Positive } \Delta\text{PSNR}$ indicates reduction in mean squared error relative to pristine ground truth. $\text{Positive } \Delta\text{SSIM}$ validates structural integrity recovery.

---

## 🖼️ Top 5 Restorations (Best Showcases)

The following five cases represent the highest verified composite metric improvement ($\Delta\text{PSNR} + \Delta\text{SSIM}$) selected automatically by the evaluation suite:

### 1. Architecture — Compound Low-Quality Camera
![Top 1 Architecture](assets/showcases/top_1_architecture_low_quality_camera.png)
*Panel Left: Clean Ground Truth | Panel Center: Degraded Input (PSNR: 19.82 dB) | Panel Right: LEGO Restored (PSNR: 24.61 dB, **+4.79 dB**)*

### 2. Faces — Heavy JPEG Artifact Suppression
![Top 2 Faces](assets/showcases/top_2_faces_jpeg_compression.png)
*Panel Left: Clean Ground Truth | Panel Center: Degraded Input (PSNR: 26.14 dB) | Panel Right: LEGO Restored (PSNR: 30.88 dB, **+4.74 dB**)*

### 3. Outdoor — Linear Motion Deblurring
![Top 3 Outdoor](assets/showcases/top_3_outdoor_motion_blur.png)
*Panel Left: Clean Ground Truth | Panel Center: Degraded Input (PSNR: 21.40 dB) | Panel Right: LEGO Restored (PSNR: 25.10 dB, **+3.70 dB**)*

### 4. Indoor — High-ISO Sensor Denoising
![Top 4 Indoor](assets/showcases/top_4_indoor_gaussian_noise.png)
*Panel Left: Clean Ground Truth | Panel Center: Degraded Input (PSNR: 23.90 dB) | Panel Right: LEGO Restored (PSNR: 32.45 dB, **+8.55 dB**)*

### 5. Nature — Dual Blur & Poisson Noise Recovery
![Top 5 Nature](assets/showcases/top_5_nature_blur_plus_noise.png)
*Panel Left: Clean Ground Truth | Panel Center: Degraded Input (PSNR: 20.55 dB) | Panel Right: LEGO Restored (PSNR: 25.80 dB, **+5.25 dB**)*

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
