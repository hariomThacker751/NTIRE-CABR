# HAFT: Hybrid Aperture and Focal Tracking for Controllable Bokeh Rendering

[![NTIRE 2026](https://img.shields.io/badge/NTIRE-2026-blue.svg)](https://cvlai.net/ntire/2026/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the **DVision** team's professional solution for the **NTIRE 2026 Controllable Aperture Bokeh Rendering (CABR) Challenge**. Our method, **HAFT**, introduces a robust architecture for generating high-resolution, photorealistic bokeh effects with precise focal tracking and aperture control.

---

## 🔬 Project Overview

The **HAFT (Hybrid Aperture and Focal Tracking)** architecture is designed to address the complexities of rendering bokeh from monocular images. By integrating depth-guided refinement heads and an aperture-aware embedding space, HAFT achieves state-of-the-art results in producing controllable blur while maintaining sharp in-focus regions.

### Key Features
- **Controllable Aperture**: Seamlessly adjust the bokeh strength using F-number embeddings.
- **Focal Tracking**: Hybrid depth and mask integration for accurate focus preservation.
- **HPC Optimized**: Includes pre-configured Slurm scripts for large-scale training and inference.

---

## 🛠️ Dependencies & Installation

This project is built on **PyTorch** and requires the following libraries:

### Core Requirements
- `torch >= 2.0.0`
- `torchvision`
- `numpy`
- `opencv-python (cv2)`
- `Pillow`
- `tqdm`
- `lpips`

### Quick Start Installation
```bash
git clone https://github.com/hariomThacker751/NTIRE-CABR.git
cd NTIRE-CABR

# Setup environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install torch torchvision tqdm opencv-python Pillow lpips
```

---

## 📂 Dataset Preparation

### Training Dataset
We utilize the **RealBokeh_3MP** dataset for fine-tuning the model. 
- **Download**: [RealBokeh_3MP on Hugging Face](https://huggingface.co/datasets/timseizinger/RealBokeh_3MP)

### Inference & Testing (NTIRE 2026)
For generating challenge submissions, ensure the testing dataset is placed in:
`HAFT FACTSHEET/dataset/Bokeh_NTIRE2026`

---

## 🚀 Execution Guide

### Training
To reproduce our training pipeline or fine-tune the HAFT model:
```bash
python "HAFT FACTSHEET/train_haft_small.py"
```
*Configurable parameters (batch size, learning rates, epochs) can be found in the `CONFIG` dictionary within the script.*

### Inference & Submission Creation
To generate the final `.zip` submission file for the NTIRE 2026 leaderboard:
```bash
python "HAFT FACTSHEET/submit_ntire.py"
```
- **Inputs**: Read from `HAFT FACTSHEET/dataset/Bokeh_NTIRE2026`.
- **Outputs**: Generated images and the metadata `readme.txt` are stored in the `outputs/` folder.
- **Archive**: A submission-ready ZIP file is automatically created in the root directory.

---

## 📊 Performance and Architecture

HAFT employs a sophisticated refinement pass that utilizes synthesized CoC (Circle of Confusion) maps and positional encodings to ensure spatial consistency and realistic blur transitions.

| Component | Description |
|-----------|-------------|
| **Backbone** | Pretrained ViT-based encoder-decoder |
| **Aperture Encoder** | Projecting F-numbers into a high-dimensional embedding space |
| **Refinement Head** | Depth-guided residual blocks for high-frequency detail preservation |

---

## 📝 Citations

If this codebase contributes to your research, please cite our work and the challenge report:

```bibtex
@InProceedings{NTIRE2026_CABR_Report,
    author    = {Seizinger, Tim and Vasluianu, Florin-Alexandru and Conde, Marcos V. and Wu, Zongwei and Zhou, Zhuyun and Timofte, Radu},
    title     = {NTIRE 2026 Challenge on Controllable Aperture Bokeh Rendering Report},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    year      = {2026}
}

@inproceedings{HAFT_DVision_2026,
    title     = {HAFT: Hybrid Aperture and Focal Tracking for Robust Bokeh Synthesis},
    author    = {DVision Team},
    booktitle = {CVPR Workshops},
    year      = {2026}
}
```

---
*Developed by the DVision Team for NTIRE 2026.*
