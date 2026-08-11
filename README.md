# ECE484 Final Project: Vision-Based Drone Control

A vision-based drone control system using NeRF rendering, Neural Pose Estimator (NPE), and Extended Kalman Filter (EKF) for closed-loop trajectory tracking.

## Overview

This project implements a complete vision control pipeline:
1. **Data Generation**: Render training images from NeRF scenes
2. **NPE Training**: Train neural network to predict drone pose from images
3. **EKF Fusion**: Fuse NPE predictions with dynamics model to reduce jitter
4. **Closed-loop Simulation**: Real-time drone control using visual feedback

## Core Files

```
├── train_npe.py                    # NPE model training
├── finetune_npe.py                 # NPE model fine-tuning
├── generate_image.py               # NeRF image generation
├── auto_dataset_generator.py       # Automatic dataset generation
└── scripts/
    ├── ece484_vision_controller.py # Vision controller (NPE + EKF)
    ├── ece484_vision_closed_loop.py# Closed-loop simulation
    ├── ekf_state_estimator.py      # EKF state estimator
    ├── drone_dynamics.py           # Drone dynamics model
    └── ns-renderer.py              # NeRF rendering interface
```

## Key Contributions

- **Neural Pose Estimator (NPE)**: ResNet50 backbone, predicts [x, y, z, yaw] from single image
- **EKF State Fusion**: Fuses visual observations with dynamics predictions for smoother output
- **Gate-focused Fine-tuning**: Increased training samples near gates to improve accuracy in critical regions

## Performance

| Model | Mean Position Error | Yaw Error |
|:---|:---|:---|
| NPE (Base) | 10.3 cm | 1.0 deg |
| NPE (Fine-tuned) | 8.9 cm | 1.0 deg |
| EKF Fusion | ~40% jitter reduction | - |

## Usage

```bash
# Closed-loop simulation
python scripts/ece484_vision_closed_loop.py --track lemniscate --ekf
```

## External Assets

Large generated assets are intentionally not included in this repository:

- `outputs/`: trained NeRF/GSplat scenes and nerfstudio checkpoints
- `npe_models/`: trained Neural Pose Estimator checkpoints
- `npe_datasets/`: rendered training datasets
- `train_gate/`, `train_ellipse/`: gate detector/refiner datasets and weights
- `closed_loop/`, `videos/`: generated evaluation outputs

Place these directories at the repository root before running the full closed-loop
pipeline. The committed code and small configuration files are sufficient to show
the implementation structure, but the large model/data artifacts must be supplied
separately.

## Dependencies

- PyTorch, torchvision
- nerfstudio
- OpenCV, NumPy, SciPy

## Reproduction status

The recovery work now has an asset-independent, strict visual-control baseline.
It runs the submitted final planner/controller against an injectable pose
observation, fixes the body/world-frame dynamics contract, and evaluates two
ordered laps without the original evaluator's double-counting behavior.

See [REPRODUCTION.md](REPRODUCTION.md) for the exact command, current numerical
baseline, coordinate conventions, and the boundary between the verified control
core and the still-missing NeRF/NPE assets.
