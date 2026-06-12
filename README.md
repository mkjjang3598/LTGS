# LTGS: Long-Term Gaussian Scene Chronology From Sparse View Updates

**CVPR 2026 (Findings)**

[Minkwan Kim](https://mkjjang3598.github.io) · [Seungmin Lee](https://veldic.github.io) · [Junho Kim](https://www.junhokim.xyz) · [Young Min Kim](http://3d.snu.ac.kr/members)

Seoul National University

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2510.09881)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://mkjjang3598.github.io/LTGS)
[![Dataset](https://img.shields.io/badge/Dataset-Google%20Drive-green)](https://drive.google.com/file/d/1qcsMhZKjr0nCiK2chGHzVUmMG5DX6gFr/view?usp=sharing)

---

## Overview

LTGS models the long-term evolution of an indoor 3D Gaussian Splatting scene from sparse, casually captured update images. Given an initial 3DGS reconstruction and a series of sparse-view image sets captured at later time steps, LTGS detects changed objects, initializes 3D Gaussian templates for newly introduced items, and integrates them into a chronological scene representation — without re-training from scratch at each step.

![Teaser](docs/assets/teaser.png)

---

## Installation

```bash
git clone --recursive https://github.com/mkjjang3598/LTGS.git
cd LTGS
```

Create the conda environment (tested on RTX 4090, CUDA 12.1):

```bash
conda create -n ltgs python=3.10 -y
conda activate ltgs

# PyTorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 3DGS rasterization backends
cd gaussian-splatting
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/fused-ssim
pip install submodules/flashsplat-rasterization
pip install opencv-python joblib tqdm
cd ..

# Hierarchical Localization (HLoc)
cd submodules/Hierarchical-Localization
python -m pip install -e .
cd ../..

# MAst3R / DUSt3R
cd submodules/mast3r
pip install -r requirements.txt
pip install -r dust3r/requirements.txt
pip install future consoleprinter
# (Optional) compile CUDA RoPE kernels for faster inference
cd dust3r/croco/models/curope
python setup.py build_ext --inplace
cd ../../../../..

# Common dependencies
pip install open3d pycolmap==3.10.0 numpy==1.26.4 timm

# TEASER++ (robust 3D registration)
sudo apt install cmake libeigen3-dev libboost-all-dev
cd submodules/TEASER-plusplus
mkdir build && cd build
cmake -DTEASERPP_PYTHON_VERSION=3.10 .. && make teaserpp_python
cd python && pip install .
cd ../../../..
```

> **Note:** If you encounter a `trunc_normal_` import error from DINO, add to the affected file:
> ```python
> import sys
> from pathlib import Path
> sys.path.append(str(Path.home() / ".cache/torch/hub"))
> from facebookresearch_dino_main.utils import trunc_normal_
> ```

---

## Dataset

Download the **LTGS** dataset from Google Drive: **[Download Dataset](https://drive.google.com/file/d/1qcsMhZKjr0nCiK2chGHzVUmMG5DX6gFr/view?usp=sharing)**

The dataset contains 5 indoor scenes captured across multiple time steps with sparse-view image sets:

| Scene | Description |
|-------|-------------|
| `livingroom` | Living room with furniture changes |
| `cafe` | Cafe table setup |
| `diningroom` | Dining room |
| `lab` | Laboratory space |
| `hall` | Hallway |
 
Place the downloaded data under `data/ltgs_dataset/`:
```
data/ltgs_dataset/
├── livingroom/
│   ├── images/          # Per-timestep image folders
│   └── ...
├── cafe/
└── ...
```

---

## Usage

### Pretrained 3DGS Models (Optional)

We provide pretrained initial 3DGS checkpoints for all 5 scenes. Download from Google Drive: **[Download Pretrained Models](https://drive.google.com/file/d/14jfAYUy-8tpr0IAWI_8WEtyOfD3f6__D/view?usp=sharing)**

Place them under `gaussian-splatting/output/`:
```
gaussian-splatting/output/
├── livingroom_first_state/
├── cafe_first_state/
├── diningroom_first_state/
├── lab_first_state/
└── hall_first_state/
```

If using the pretrained models, skip Step 1 below.

---

### 1. Train Initial 3DGS

```bash
GPU=0
OUTPUT=output/${SCENE}_first_state

cd gaussian-splatting
CUDA_VISIBLE_DEVICES=${GPU} python train.py -s ../${IMAGE_PATH} \
    -m ${OUTPUT} -r 2 --eval
# Optional: generate renderings and compute metrics
CUDA_VISIBLE_DEVICES=${GPU} python render.py -m ${OUTPUT}
CUDA_VISIBLE_DEVICES=${GPU} python torch_metrics.py -m ${OUTPUT}
cd ..
```

### 2. Run Long-Term Update Pipeline

Use the provided scene-specific scripts to run the full LTGS pipeline (localization → change detection → instance matching → PCD initialization → long-term update):

```bash
bash scripts/${SCENE}.sh
```

Available scenes: `livingroom`,`cafe`, `diningroom`, `lab`, `hall`

### 3. Evaluate

```bash
CUDA_VISIBLE_DEVICES=${GPU} python metrics.py -m output/${SCENE}
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{kim2026ltgs,
  title={LTGS: Long-Term Gaussian Scene Chronology From Sparse View Updates},
  author={Kim, Minkwan and Lee, Seungmin and Kim, Junho and Kim, Young Min},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={488--497},
  year={2026}
}
```

---

## Acknowledgements

This project builds on [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), [MASt3R](https://github.com/naver/mast3r), [Hierarchical-Localization](https://github.com/cvg/Hierarchical-Localization), [TEASER++](https://github.com/MIT-SPARK/TEASER-plusplus), and [GeSCF](https://github.com/AutoCompSysLab/towards-generalizable-scene-change-detection).
