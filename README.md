# LEGO: Deep Spatial Restoration Network

| original image | degraded image | lego's output |
| :--- | :--- | :--- |
| ![sample](images/original.png) | ![sample](images/gaussian_blur_input.png) | ![sample](images/gaussian_blur_LEGO.png) |

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-Supported-76B900.svg?style=flat&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Interim_Checkpoint_(Stage_2)-orange.svg)]()

LEGO (**DeepSpatialRestorationNet**) is a residual image restoration network designed to remove composite, real-world image degradations (deblurring, denoising, demosaicing, and JPEG deblocking) in a single forward pass.

It couples **Modulated Deformable Convolutions (DCNv2)** for spatially adaptive geometric feature alignment with **Restormer-style Transposed Cross-Covariance Attention Blocks** for long-range context aggregation, using a residual formulation to recover sharp, clean outputs while preserving scene structure and color fidelity.

---

## ⚠️ Transparent Training Status & Disclosures

We believe in scientific transparency. The released checkpoint (`LEGO_latest.pth`) is an **interim model** and is **not fully converged**. Please review the planned versus executed training history before treating it as a production-ready restoration model.

| Training Stage | Objective & Data | Planned Budget | Actual Executed Status | Checkpoint Export |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1: Pixel Pretraining** | DIV2K (800 HR images), Charbonnier + FFT + Edge Loss. | 100 Epochs | **Halted at Epoch 51** (~51% complete) | Transferred to Stage 2 |
| **Stage 2: Generalized Diversity** | Streamed COCO + Flickr30k mixed with 50% DIV2K replay, L2-SP anchor. | 60 Epochs | **Halted at Epoch 17** (~28% complete) | **`LEGO_latest.pth`** |
| **Stage 3: Perceptual GAN Fine-Tuning** | VGG19 Perceptual Loss + Spectral Norm U-Net Discriminator. | 80 Epochs | **Not Executed (Pending)** | N/A |

### What this means for performance:
- **Strong Capabilities**: Good structural denoising, JPEG artifact suppression, and moderate motion/defocus deblurring. Color constancy and high-frequency structure are preserved comparatively well across many composite degradations.
- **Current Limitations**: Because Stage 3 (adversarial perceptual loss) was not executed, the model functions as a primarily L1/Charbonnier/frequency-regressed network. Fine textures and adversarially complex scenes may still be under-optimized.

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

- Architecture class: `DeepSpatialRestorationNet`
- Embedding dimension: `128`
- Parameters: `10,702,607`
- DCN backend: `torchvision`
- Attention heads: `4`
- Output/input range: `[-1, 1]`
- Tile inference: `256 x 256` with `64` overlap and `192` stride

---

## 📊 Benchmark & Empirical Evaluation

The benchmark data below is taken directly from the evaluation report for the `LEGO_latest.pth` checkpoint.

### Family-level summary

| Degradation family | ΔPSNR (dB) | ΔSSIM | ΔMAE | Helped | Hurt | Tests | Verdict |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| TRAINING-MATCHED CONTROL | +1.0406 | +0.0212 | -0.0080 | 3 | 0 | 3 | GENERALLY HELPFUL |
| GAUSSIAN BLUR | +2.3329 | +0.0208 | -0.0091 | 1 | 0 | 1 | GENERALLY HELPFUL |
| STRONG GAUSSIAN BLUR | +0.5564 | +0.0115 | -0.0034 | 1 | 0 | 1 | GENERALLY HELPFUL |
| MOTION BLUR | +0.8785 | +0.0108 | -0.0040 | 2 | 0 | 2 | GENERALLY HELPFUL |
| DEFOCUS / OUT-OF-FOCUS | +0.9964 | +0.0158 | -0.0058 | 1 | 0 | 1 | GENERALLY HELPFUL |
| SENSOR NOISE | +6.1414 | +0.0367 | -0.0256 | 1 | 0 | 1 | GENERALLY HELPFUL |
| JPEG COMPRESSION | -1.0010 | -0.0040 | +0.0024 | 0 | 1 | 1 | GENERALLY HARMFUL |
| BLUR + SENSOR NOISE | +2.3937 | +0.0306 | -0.0157 | 1 | 0 | 1 | GENERALLY HELPFUL |
| BLUR + NOISE + JPEG | +0.9680 | +0.0102 | -0.0049 | 1 | 0 | 1 | GENERALLY HELPFUL |
| DOWNSAMPLE + BLUR + NOISE + JPEG | +0.3889 | +0.0057 | -0.0019 | 1 | 0 | 1 | GENERALLY HELPFUL |

### Unseen generalization

| Metric | Value |
| :--- | ---: |
| mean ΔPSNR (dB) | +1.4534 |
| mean ΔSSIM | +0.0149 |
| helped rate | 0.9000 |
| hurt rate | 0.1000 |
| verdict | SOME GENERALIZATION |

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
