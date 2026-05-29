"""
=============================================================
Anti-Drone System — Model Evaluation Script
=============================================================
USAGE:
    python scripts/evaluate_model.py --model best.pt --yaml data.yaml

WHAT IT DOES:
    - Runs YOLOv8 validation on the validation set
    - Prints mAP@0.5, Precision, Recall, F1
    - Saves confusion matrix and PR curve
    - Saves per-class metrics table
=============================================================
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 drone detection model")
    parser.add_argument("--model", type=str,
                        default="runs/detect/anti_drone_final/weights/best.pt",
                        help="Path to trained model weights (.pt)")
    parser.add_argument("--yaml",  type=str,
                        default="configs/data.yaml",
                        help="Path to dataset YAML file")
    parser.add_argument("--imgsz", type=int, default=416,
                        help="Inference image size")
    parser.add_argument("--conf",  type=float, default=0.5,
                        help="Confidence threshold for evaluation")
    parser.add_argument("--iou",   type=float, default=0.5,
                        help="IoU threshold for mAP calculation")
    return parser.parse_args()


def evaluate(args):
    print("=" * 60)
    print("  ANTI-DRONE — MODEL EVALUATION")
    print("=" * 60)

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not os.path.exists(args.yaml):
        raise FileNotFoundError(f"YAML not found: {args.yaml}")

    print(f"\n📦 Loading: {args.model}")
    model = YOLO(args.model)

    print(f"\n🔍 Running validation on dataset: {args.yaml}")
    metrics = model.val(
        data  = args.yaml,
        imgsz = args.imgsz,
        conf  = args.conf,
        iou   = args.iou,
        verbose = True,
    )

    # ── Extract key metrics ──────────────────────────────────
    map50  = metrics.box.map50
    map50_95 = metrics.box.map
    precision = metrics.box.mp     # mean precision
    recall    = metrics.box.mr     # mean recall
    f1        = (2 * precision * recall / (precision + recall + 1e-9))

    print("\n" + "=" * 60)
    print("  📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"  mAP@0.5       : {map50:.4f}")
    print(f"  mAP@0.5:0.95  : {map50_95:.4f}")
    print(f"  Precision     : {precision:.4f}")
    print(f"  Recall        : {recall:.4f}")
    print(f"  F1 Score      : {f1:.4f}")
    print("=" * 60)

    # ── Interpretation ───────────────────────────────────────
    print("\n💡 Interpretation:")
    if map50 >= 0.85:
        print("  ✅ Excellent detection performance (mAP@0.5 ≥ 0.85)")
    elif map50 >= 0.70:
        print("  ⚠ Good performance — consider more training data or epochs")
    else:
        print("  ❌ Low mAP — review dataset quality and training config")

    if precision < 0.7:
        print("  ⚠ Low Precision — model may produce false positives")
        print("     Fix: Increase --conf threshold or clean dataset")
    if recall < 0.7:
        print("  ⚠ Low Recall — model may miss drones")
        print("     Fix: Lower --conf threshold or add more drone images")

    # ── Save metrics table ───────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    metrics_dict = {
        "Metric": ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1"],
        "Value":  [f"{map50:.4f}", f"{map50_95:.4f}",
                   f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"]
    }
    df = pd.DataFrame(metrics_dict)
    csv_path = "outputs/evaluation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Metrics saved: {csv_path}")

    # ── Bar chart ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = ["steelblue", "navy", "green", "orange", "red"]
    bars    = ax.bar(
        ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1 Score"],
        [map50, map50_95, precision, recall, f1],
        color=colors, edgecolor="white", width=0.5
    )
    ax.set_ylim(0, 1.0)
    ax.set_title("YOLOv8 Model Performance — Anti-Drone System", fontsize=13, pad=12)
    ax.set_ylabel("Score (0–1)")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="0.5 baseline")
    ax.axhline(0.8, color="green", linestyle="--", linewidth=1, label="0.8 target")
    ax.legend(fontsize=9)

    for bar, val in zip(bars, [map50, map50_95, precision, recall, f1]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    chart_path = "outputs/evaluation_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"✓ Chart saved: {chart_path}")
    plt.show()

    return metrics


if __name__ == "__main__":
    args = get_args()
    evaluate(args)
