# HAFT: Hybrid Aperture and Focal Tracking for Controllable Bokeh Rendering

This repository contains the **DVision** team's solution to the **NTIRE 2026 Controllable Aperture Bokeh Rendering (CABR) Challenge**. Our approach, **HAFT**, is designed to generate realistic and controllable bokeh effects from monocular images by leveraging depth-guided refinement and aperture-aware embeddings.

---

## Installation

To set up the environment, follow these steps:

```bash
# Clone the repository
git clone https://github.com/hariomThacker751/NTIRE-CABR.git
cd NTIRE-CABR

# Create and activate a virtual environment
python3 -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install core dependencies
pip install torch torchvision tqdm opencv-python Pillow lpips
```

## Dataset Setup

The project is configured to work with the official NTIRE 2026 Bokeh Challenge dataset. Place your dataset in the `dataset/` directory.

### Structure
```
dataset/
└── Bokeh_NTIRE2026/
    ├── validation/   # Used for 'dev' phase evaluation
    └── test/         # Used for 'test' phase submission
```

### Automatic Setup
The `submit_ntire.py` script includes logic to automatically detect and unpack dataset archives (e.g., `Bokeh_NTIRE2026_Development_Inputs.zip`) if they provide the required directory structure.

## Inference (Submission Generation)

To generate a ready-to-upload `.zip` archive for Codabench, run:

```bash
python submit_ntire.py
```

### Configuration
Inference parameters are managed within the `Config` class in `submit_ntire.py`:

- **Phase**: Set `phase = 'dev'` (validation) or `phase = 'test'` (final leaderboard).
- **Checkpoints**: Ensure `checkpoint_path` points to your trained weights (e.g., `checkpoints/haft_large_best_psnr.pth`).
- **Optimization**: `max_inference_dim` (default: 1024) helps manage memory on lower-VRAM GPUs.

## Training

To training the HAFT model (backbone + refinement heads):

```bash
python train_haft_small.py
```

### Configuration
Training hyperparameters are defined in the `CONFIG` dictionary:

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `img_size`| 512           | Spatial size of training patches |
| `batch_size`| 2            | Samples per GPU (adjusted for VRAM) |
| `lr_backbone`| 1e-5       | Learning rate for the pretrained backbone |
| `lr_haft` | 5e-4          | Learning rate for HAFT refinement heads |
| `epochs`  | 100           | Total training epochs |

## HPC Usage (Slurm)

If you are using an HPC environment, Slurm job scripts are provided for convenience:

- **Training**: `sbatch haft_job.sh`
- **Inference**: `sbatch submit_job.sh`

## Citations

If this repository contributes to your research, please consider citing:

```bibtex
@InProceedings{NTIRE2026_CABR_Report,
    author    = {Seizinger, Tim and Vasluianu, Florin-Alexandru and Conde, Marcos V. and Wu, Zongwei and Zhou, Zhuyun and Timofte, Radu},
    title     = {NTIRE 2026 Challenge on Controllable Aperture Bokeh Rendering Report},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    year      = {2026}
}

@inproceedings{HAFT_Bokeh2024,
    title     = {Hybrid Aperture and Focal Tracking for Realistic Bokeh Rendering},
    author    = {DVision Team},
    booktitle = {CVPR Workshops},
    year      = {2024}
}
```
