# HAFT: Hybrid Aperture-conditioned Feature Transformer for Controllable Bokeh Rendering

[![NTIRE 2026](https://img.shields.io/badge/NTIRE-2026-blue.svg)](https://cvlai.net/ntire/2026/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **HAFT**, developed by team **CV SVNIT** for the **NTIRE 2026 Controllable Aperture Bokeh Rendering (CABR) Challenge**. HAFT is a high-resolution architecture designed for photorealistic bokeh synthesis with precise focal tracking and aperture control.

---

## Architectural Overview

HAFT (Hybrid Aperture-conditioned Feature Transformer) utilizes a multi-stage pipeline to achieve controllable bokeh rendering from monocular inputs:

1.  **Aperture Encoding**: Scalar f-stop values are mapped into a 64-dimensional embedding via Fourier positional features (8 frequency bands) and a two-layer MLP. This embedding modulates the network features through **FiLM (Feature-wise Linear Modulation)** layers.
2.  **Backbone U-Net**: An optimized U-Net backbone processes RGB inputs concatenated with physics-based **Circle-of-Confusion (CoC)** maps and spatial positional encodings.
3.  **Aperture-Aware Attention**: The Deep Feature Extraction (DFE) bottleneck incorporates decomposed attention with dynamic relative positional bias. The attention span scales inversely with the aperture value, enabling adaptive spatial context for varying blur intensities.
4.  **Depth-guided Refinement**: A specialized refinement head fuses base predictions with depth and focal priors to predict focus-weighted residuals, effectively mitigating artifacts at foreground-background transitions.

---

## Pre-trained Models

The trained HAFT checkpoint used for our challenge submission is available for download:

*   **HAFT Checkpoint (Google Drive)**: [Download Link](https://drive.google.com/file/d/1QDwx-tDu7TBYsyYjiBCO-98suTFRbLSq/view?usp=drive_link)

---

## Installation Guide

The environment requires Python 3.8+ and a CUDA-compatible NVIDIA GPU.

### Dependency Installation
All required Python packages are listed in `requirements.txt`.

```bash
git clone https://github.com/hariomThacker751/NTIRE-CABR.git
cd NTIRE-CABR

# Environment Setup
python -m venv venv
# Windows: .\venv\Scripts\activate | Linux: source venv/bin/activate
source venv/bin/activate 

# Install dependencies
pip install -r requirements.txt
```

---

## Dataset Management

### Training Data
Models were trained using the **RealBokeh_3MP** dataset.
*   **Access**: [RealBokeh_3MP on Hugging Face](https://huggingface.co/datasets/timseizinger/RealBokeh_3MP)

### Inference and Testing
For challenge evaluation, the dataset should be organized according to the structure expected by the inference scripts. The testing data path is typically:
`HAFT FACTSHEET/dataset/Bokeh_NTIRE2026`

---

## Execution Instructions

### Training
To initialize training for the backbone and HAFT modules:
```bash
python "HAFT FACTSHEET/train_haft_small.py"
```
The training process employs differential learning rates ($10^{-5}$ for the backbone, $5 \times 10^{-4}$ for HAFT modules) and a composite Charbonnier-FFT-LPIPS loss function.

### Inference and Submission Generation
To generate predictions for the NTIRE 2026 challenge and create a Codabench-compliant submission:
```bash
python "HAFT FACTSHEET/submit_ntire.py"
```
*Input images are read from the specified directory, and results are stored in the `outputs/` folder. A submission archive is automatically generated in the root directory.*

---

## References and Citations

The HAFT architecture incorporates elements from the **Bokehlicious** project and other recent advancements in image restoration.

### Reference Repository
*   **Bokehlicious**: [https://github.com/TimSeizinger/Bokehlicious](https://github.com/TimSeizinger/Bokehlicious)

### Citation
Please cite the following works if you use this codebase in your research:

```bibtex
@InProceedings{NTIRE2026_CABR_Report,
    author    = {Seizinger, Tim and Vasluianu, Florin-Alexandru and Conde, Marcos V. and Wu, Zongwei and Zhou, Zhuyun and Timofte, Radu},
    title     = {NTIRE 2026 Challenge on Controllable Aperture Bokeh Rendering Report},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
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
*Developed by the CV SVNIT Team (SVNIT Surat) for the NTIRE 2026 Challenge.*
