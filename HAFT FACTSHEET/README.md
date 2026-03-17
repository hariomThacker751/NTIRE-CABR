# HAFT: Hybrid Aperture-conditioned Feature Transformer for Controllable Bokeh Rendering

<p align="center">
  <img src="https://img.shields.io/badge/NTIRE-2026-blue.svg" alt="NTIRE 2026">
  <img src="https://img.shields.io/badge/Task-Bokeh%20Rendering-orange.svg" alt="Task">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

Official implementation of **HAFT**, the **CV SVNIT** team's solution for the **NTIRE 2026 Controllable Aperture Bokeh Rendering (CABR) Challenge**. HAFT is a professional architecture designed for generating high-resolution, photorealistic bokeh effects with precise focal tracking and aperture control.

---

## 🔬 Architectural Overview

**HAFT (Hybrid Aperture-conditioned Feature Transformer)** introduces a robust multi-stage pipeline for controllable bokeh synthesis:

1.  **Aperture Encoding**: Encodes scalar f-stop values into a 64-dimensional embedding via Fourier positional features and a two-layer MLP. This guides the network through **FiLM** layers.
2.  **Backbone U-Net**: A modified U-Net processing RGB images with a physics-based **Circle-of-Confusion (CoC)** map.
3.  **Aperture-Aware Attention**: Utilizes decomposed attention with dynamic relative positional bias that scales with the aperture value.
4.  **Depth-guided Refinement**: A lightweight head that fuses base predictions with depth and focal priors to correct artifacts at transitions.

---

## 🛠️ Installation & Setup

Follow these steps to set up the environment and install all dependencies for training and inference.

### 📋 Prerequisites
- Python 3.8+
- NVIDIA GPU + CUDA

### ⚙️ Quick Start
```bash
git clone https://github.com/hariomThacker751/NTIRE-CABR.git
cd NTIRE-CABR

# Setup environment
python -m venv venv
# On Windows: .\venv\Scripts\activate | On Linux: source venv/bin/activate
source venv/bin/activate 

# Install dependencies
pip install -r requirements.txt
```

---

## 📂 Dataset Management

### 📥 Training Dataset
Models were fine-tuned using the **RealBokeh_3MP** dataset.
- **Source**: [RealBokeh_3MP on Hugging Face](https://huggingface.co/datasets/timseizinger/RealBokeh_3MP)

### 🧪 Inference & NTIRE Testing
For challenge evaluation, ensure the dataset is structured under:
`HAFT FACTSHEET/dataset/Bokeh_NTIRE2026`

---

## 🚀 Execution Guide

### 🏋️ Training
Execute the joint fine-tuning of the backbone and HAFT modules:
```bash
python "HAFT FACTSHEET/train_haft_small.py"
```

### 🎯 Submission & Inference
Generate the final Codabench-ready `.zip` submission:
```bash
python "HAFT FACTSHEET/submit_ntire.py"
```
*Outputs (images and metadata) are stored in the `outputs/` directory, and a submission ZIP is created in the root.*

---

## 📚 References & Citations

Our work is based on the **Bokehlicious** architecture and several foundational papers in image restoration.

### 🔗 Reference Repository
- **Bokehlicious**: [https://github.com/TimSeizinger/Bokehlicious](https://github.com/TimSeizinger/Bokehlicious)

### 📄 Citation
If this repository helps your research, please cite the following:

```bibtex
@InProceedings{NTIRE2026_CABR_Report,
    author    = {Seizinger, Tim and Vasluianu, Florin-Alexandru and Conde, Marcos V. and Wu, Zongwei States, Zhuyun and Timofte, Radu},
    title     = {NTIRE 2026 Challenge on Controllable Aperture Bokeh Rendering Report},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    year      = {2026}
}

@inproceedings{HAFT_DVision_2026,
    title     = {HAFT: Hybrid Aperture-conditioned Feature Transformer for Controllable Bokeh Rendering},
    author    = {CV SVNIT Team (Divyavardhan Singh, Hariom Thacker, Aanchal Maurya, Hammad Mohammad)},
    booktitle = {NTIRE 2026 Workshop @ CVPR 2026},
    year      = {2026}
}

@misc{Seizinger2023Bokehlicious,
  author = {Seizinger, Tim},
  title = {Bokehlicious: Bokeh Rendering from Defocus Estimation},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/TimSeizinger/Bokehlicious}}
}
```

---
<p align="center">
  <i>Developed by the CV SVNIT Team (SVNIT Surat) for NTIRE 2026.</i>
</p>
