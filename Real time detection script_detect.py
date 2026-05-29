"""
=============================================================
Anti-Drone System — YOLOv8 Training Script
=============================================================
USAGE:
    python src/train.py
    python src/train.py --yaml path/to/data.yaml --epochs 100

WHAT IT DOES:
    - Loads a YOLOv8n (Nano) pretrained model
    - Fine-tunes it on your custom drone dataset
    - Saves best weights to runs/detect/anti_drone_final/weights/best.pt
    - Plots and saves training metrics
=============================================================
"""

import os
import argparse
import yaml
import torch
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO


# ──────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ──────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for drone detection")
    parser.add_argument("--yaml",    type=str, default="configs/data.yaml",
                        help="Path to dataset YAML file")
    parser.add_argument("--model",   type=str, default="yolov8n.pt",
                        help="Pretrained model: yolov8n.pt / yolov8s.pt / yolov8m.pt")
    parser.add_argument("--epochs",  type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--imgsz",   type=int, default=416,
                        help="Input image size (must be multiple of 32)")
    parser.add_argument("--batch",   type=int, default=8,
                        help="Batch size (reduce to 4 if OOM on CPU)")
    parser.add_argument("--workers", type=int, default=2,
                        help="DataLoader workers")
    parser.add_argument("--name",    type=str, default="anti_drone_final",
                        help="Training run name (saved under runs/detect/)")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────
# DEVICE SELECTION
# ──────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        device = 0
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"\n✓ GPU detected: {gpu_name} ({vram:.1f} GB VRAM)")
        print("  Training will use CUDA for maximum speed.\n")
    else:
        device = "cpu"
        print("\n⚠ No GPU found — training on CPU.")
        print("  Tip: Use Google Colab (free GPU) for faster training.")
        print("  Expected time on CPU: ~2–4 hours for 50 epochs.\n")
    return device


# ──────────────────────────────────────────────────────────
# VALIDATE YAML
# ──────────────────────────────────────────────────────────

def validate_yaml(yaml_path):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(
            f"\n❌ YAML file not found: {yaml_path}\n"
            "  1. Download dataset from Roboflow\n"
            "  2. Update the path in configs/data.yaml\n"
            "  3. Re-run this script\n"
        )
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    print(f"✓ YAML loaded: {yaml_path}")
    print(f"  Classes   : {cfg.get('names', 'Not found')}")
    print(f"  Num classes: {cfg.get('nc', 'Not found')}")
    return cfg


# ──────────────────────────────────────────────────────────
# PLOT TRAINING RESULTS
# ──────────────────────────────────────────────────────────

def plot_results(run_name):
    """
    Parse YOLOv8 results CSV and plot training metrics.
    """
    results_csv = os.path.join("runs", "detect", run_name, "results.csv")

    if not os.path.exists(results_csv):
        print("⚠ results.csv not found — skipping plot.")
        return

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("YOLOv8 Training Metrics — Anti-Drone System", fontsize=14)

    metrics = [
        ("train/box_loss",      "Train Box Loss",       "red"),
        ("train/cls_loss",      "Train Class Loss",     "orange"),
        ("train/dfl_loss",      "Train DFL Loss",       "purple"),
        ("metrics/mAP50(B)",    "mAP@0.5",              "green"),
        ("metrics/precision(B)","Precision",            "blue"),
        ("metrics/recall(B)",   "Recall",               "teal"),
    ]

    for ax, (col, title, color) in zip(axes.flatten(), metrics):
        if col in df.columns:
            ax.plot(df["epoch"], df[col], color=color, linewidth=2)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)
        else:
            ax.set_title(f"{title} (not found)", color="grey")
            ax.axis("off")

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    save_path = "outputs/training_metrics.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Training metrics plot saved: {save_path}")
    plt.show()


# ──────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ──────────────────────────────────────────────────────────

def train(args):
    print("=" * 60)
    print("  ANTI-DRONE SYSTEM — YOLOv8 TRAINING")
    print("=" * 60)

    # Validate dataset
    validate_yaml(args.yaml)

    # Device
    device = get_device()

    # Load pretrained YOLOv8 model
    print(f"\n📦 Loading model: {args.model}")
    model = YOLO(args.model)
    print(f"   Parameters: {sum(p.numel() for p in model.model.parameters()):,}")

    # Training configuration summary
    print("\n📋 Training Configuration:")
    print(f"   Dataset YAML  : {args.yaml}")
    print(f"   Epochs        : {args.epochs}")
    print(f"   Image size    : {args.imgsz}×{args.imgsz}")
    print(f"   Batch size    : {args.batch}")
    print(f"   Device        : {'GPU' if device == 0 else 'CPU'}")
    print(f"   Run name      : {args.name}")
    print("\n🚀 Starting training...\n")

    # Train
    results = model.train(
        data      = args.yaml,
        epochs    = args.epochs,
        imgsz     = args.imgsz,
        batch     = args.batch,
        workers   = args.workers,
        device    = device,
        name      = args.name,

        # Augmentation
        fliplr    = 0.5,           # horizontal flip
        mosaic    = 1.0,           # mosaic augmentation
        hsv_h     = 0.015,         # hue jitter
        hsv_s     = 0.7,           # saturation jitter
        hsv_v     = 0.4,           # value jitter

        # Training behaviour
        patience  = 10,            # early stopping patience
        cache     = False,         # set True if RAM > 8GB
        save      = True,
        verbose   = True,
    )

    # Best model path
    best_pt = os.path.join("runs", "detect", args.name, "weights", "best.pt")
    print("\n" + "=" * 60)
    print("  ✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n📁 Best model saved: {best_pt}")
    print("\n💡 Next step:")
    print(f"   python src/detect.py --model {best_pt}")

    # Plot metrics
    plot_results(args.name)

    return best_pt


# ──────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = get_args()
    train(args)
