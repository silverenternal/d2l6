#!/usr/bin/env python3
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "intel-scenes"
PLOTS_DIR = ROOT / "plots"
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

TRAIN_LOG = [
    (1, 0.8202, 77.97, 77.93),
    (2, 0.5550, 79.75, 78.97),
    (3, 0.4768, 81.34, 81.47),
    (4, 0.4185, 83.91, 83.07),
    (5, 0.3879, 84.36, 82.80),
    (6, 0.3766, 81.27, 80.57),
    (7, 0.3485, 80.74, 79.70),
    (8, 0.3166, 83.15, 81.13),
    (9, 0.3101, 84.25, 82.43),
    (10, 0.2961, 90.20, 88.20),
    (11, 0.2259, 93.67, 90.13),
    (12, 0.2121, 93.41, 90.10),
    (13, 0.2109, 92.88, 90.23),
    (14, 0.1982, 94.21, 90.53),
    (15, 0.1856, 94.68, 91.23),
    (16, 0.1580, 95.80, 91.20),
    (17, 0.1570, 95.85, 91.23),
    (18, 0.1472, 96.06, 91.37),
]


def list_images(path: Path) -> list[Path]:
    return sorted(
        p
        for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def count_split(split: str) -> OrderedDict[str, int]:
    split_root = DATA_ROOT / split / split
    return OrderedDict((cls, len(list_images(split_root / cls))) for cls in CLASSES)


def save_class_distribution() -> None:
    train_counts = count_split("seg_train")
    test_counts = count_split("seg_test")
    x = np.arange(len(CLASSES))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    ax.bar(x - width / 2, list(train_counts.values()), width, label="Train", color="#3b82f6")
    ax.bar(x + width / 2, list(test_counts.values()), width, label="Test", color="#f97316")
    ax.set_title("Intel Scene Dataset Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Images")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "class_distribution.png")
    plt.close(fig)


def save_training_curves() -> None:
    epochs = [row[0] for row in TRAIN_LOG]
    losses = [row[1] for row in TRAIN_LOG]
    train_acc = [row[2] for row in TRAIN_LOG]
    test_acc = [row[3] for row in TRAIN_LOG]

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=160)

    ax_loss.plot(epochs, losses, marker="o", color="#2563eb")
    ax_loss.set_title("Training Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross Entropy Loss")
    ax_loss.grid(alpha=0.25)

    ax_acc.plot(epochs, train_acc, marker="o", label="Train", color="#16a34a")
    ax_acc.plot(epochs, test_acc, marker="s", label="Test", color="#dc2626")
    ax_acc.set_title("Classification Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_ylim(75, 98)
    ax_acc.legend()
    ax_acc.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "training_curves.png")
    plt.close(fig)


def save_sample_grid() -> None:
    fig, axes = plt.subplots(len(CLASSES), 4, figsize=(8, 11), dpi=160)
    train_root = DATA_ROOT / "seg_train" / "seg_train"

    for row, cls in enumerate(CLASSES):
        files = list_images(train_root / cls)[:4]
        for col in range(4):
            ax = axes[row, col]
            ax.axis("off")
            img = Image.open(files[col]).convert("RGB").resize((128, 128))
            ax.imshow(img)
            if col == 0:
                ax.set_ylabel(cls, rotation=0, ha="right", va="center", labelpad=42)

    fig.suptitle("Sample Images from Each Scene Class", y=0.995)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "sample_grid.png")
    plt.close(fig)


def main() -> None:
    if not DATA_ROOT.exists():
        raise SystemExit("Dataset not found. Run scripts/download_intel_scenes.sh first.")
    PLOTS_DIR.mkdir(exist_ok=True)
    save_class_distribution()
    save_training_curves()
    save_sample_grid()
    print(f"Saved visualizations to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
