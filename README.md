# D2L Homework 6

GitHub: https://github.com/silverenternal/d2l6

This repository contains the code and report for the sixth D2L homework. The implementation is written in Rust with `tch-rs`.

## Contents

- `ParallelModule`: takes two submodules, runs both on the same input, and concatenates their outputs.
- Shared-parameter MLP: reuses one linear layer twice and prints parameter and gradient information during training.
- Calculation questions: parameter count, convolution output size, convolution parameter count, pooling output size.
- Scene classification CNN: trains a custom convolutional network on the Intel Image Classification dataset.

## Project Files

- `src/main.rs`: main homework implementation.
- `report.md`: homework report.
- `scripts/download_intel_scenes.sh`: downloads the scene classification dataset.
- `scripts/run_with_rocm.sh`: runs the Rust program with ROCm PyTorch libraries.
- `scripts/visualize_dataset.py`: generates dataset and training visualizations.
- `plots/`: generated figures used by the report.
- `data/README.md`: dataset notes.

## Environment

The project uses:

- Rust
- `tch = 0.23`
- LibTorch / PyTorch C++ libraries
- ROCm GPU runtime for the local GPU run
- Python with `matplotlib`, `numpy`, and `Pillow` for visualization

On this machine, the GPU run uses ROCm libraries under `/usr/lib` and `/opt/rocm/lib`. The helper script sets the required environment variables.

## Dataset

The scene classification part uses the Intel Image Classification dataset with 6 classes:

```text
buildings, forest, glacier, mountain, sea, street
```

Download and extract it with:

```bash
bash scripts/download_intel_scenes.sh
```

Expected dataset path:

```text
data/intel-scenes/
  seg_train/seg_train/
  seg_test/seg_test/
```

The dataset directory and zip file are ignored by Git.

## Run

For the ROCm GPU run:

```bash
./scripts/run_with_rocm.sh
```

For a normal Cargo run:

```bash
cargo run
```

The normal run depends on the local LibTorch setup. On this machine, the ROCm script is the tested command.

## Visualizations

Generate the figures used in the report:

```bash
python scripts/visualize_dataset.py
```

The generated figures are:

- `plots/class_distribution.png`
- `plots/sample_grid.png`
- `plots/training_curves.png`

All plot text is in English to avoid Chinese font rendering problems.

## Current Scene Classification Result

The final CNN uses the full training and test sets, `128 x 128` inputs, 5 convolution blocks, AdamW, data augmentation, and learning rate decay.

Best recorded test accuracy:

```text
91.37%
```

Training samples: `14034`  
Test samples: `3000`
