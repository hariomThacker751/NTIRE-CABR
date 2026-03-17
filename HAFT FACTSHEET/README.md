# HAFT: Hybrid Aperture-conditioned Feature Transformer for Controllable Bokeh Rendering

[![NTIRE 2026](https://img.shields.io/badge/NTIRE-2026-blue.svg)](https://cvlai.net/ntire/2026/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **HAFT**, the DVision team's solution for the **NTIRE 2026 Controllable Aperture Bokeh Rendering (CABR) Challenge**. HAFT is a robust architecture designed for generating high-resolution, photorealistic bokeh effects with precise focal tracking and aperture control.

---

## 🔬 Architectural Overview

HAFT (Hybrid Aperture-conditioned Feature Transformer) introduces a multi-stage pipeline for controllable bokeh synthesis:

1.  **Aperture Encoding**: The scalar f-stop value is encoded into a 64-dimensional embedding using Fourier positional features (8 frequency bands) and a two-layer MLP. This embedding guides the entire network via **FiLM (Feature-wise Linear Modulation)** layers.
2.  **Backbone U-Net**: A modified U-Net architecture that processes RGB images combined with a physics-based **Circle-of-Confusion (CoC)** map and positional encodings. 
3.  **Aperture-Aware Attention**: The Deep Feature Extraction (DFE) bottleneck utilizes decomposed attention with dynamic relative positional bias. The attention field's decay range scales dynamically with the aperture value, contracting for wide apertures and expanding for narrow ones.
4.  **Depth-guided Refinement**: A lightweight head fuses the base prediction with depth and focal priors to predict a focus-weighted residual, effectively correcting artifacts at foreground-background transitions.

---

## 🛠️ Installation & Dependencies

To set up the environment and install all necessary dependencies for training and inference, follow these steps:

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA support

### Dependencies
The project requires the following Python libraries:
- `torch >= 2.0.0`
- `torchvision`
- `numpy`
- `opencv-python`
- `Pillow`
- `tqdm`
- `lpips`

### Quick Set-up
```bash
git clone https://github.com/hariomThacker751/NTIRE-CABR.git
cd NTIRE-CABR

# Create virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install all dependencies
pip install torch torchvision numpy opencv-python Pillow tqdm lpips
```

---

## 📂 Dataset Management

### Training
Models were fine-tuned using the **RealBokeh_3MP** dataset.
- **Dataset Link**: [RealBokeh_3MP (Hugging Face)](https://huggingface.co/datasets/timseizinger/RealBokeh_3MP)

### Inference & NTIRE Testing
For challenge evaluation and generating submission files, ensure the testing dataset is structured under:
`HAFT FACTSHEET/dataset/Bokeh_NTIRE2026`

---

## 🚀 Execution Guide

### Training Pipeline
Execute the joint fine-tuning of the backbone and HAFT modules:
```bash
python "HAFT FACTSHEET/train_haft_small.py"
```
*Note: The script uses differential learning rates ($10^{-5}$ for backbone, $5 \times 10^{-4}$ for HAFT heads) and a composite Charbonnier-FFT-LPIPS loss.*

### Submission & Inference
To generate the final Codabench-ready `.zip` submission:
```bash
python "HAFT FACTSHEET/submit_ntire.py"
```
- **Input**: The script reads from `HAFT FACTSHEET/dataset/Bokeh_NTIRE2026`.
- **Output**: Individual prediction images and the required `readme.txt` are stored in the `outputs/` directory.
- **Archive**: A submission-compliant ZIP file is automatically generated in the root directory upon completion.

---

## 📚 References & Citation

Our work builds upon several foundational architectures:
- **Restormer**: Efficient Transformer for High-Resolution Image Restoration (CVPR 2022)
- **NAFNet**: Simple Baselines for Image Restoration (ECCV 2022)
- **Bokeh Rendering from Defocus Estimation** (ECCV 2022)



---
*Developed by the DVision Team (SVNIT Surat) for the NTIRE 2026 Challenge.*
