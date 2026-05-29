"""
=============================================================
Anti-Drone System — Environment Setup Script
=============================================================
USAGE:
    python src/setup.py

WHAT IT DOES:
    - Checks Python version (3.8+ required)
    - Installs all required libraries
    - Verifies webcam connection
    - Creates project folder structure
    - Prints next steps
=============================================================
"""

import subprocess
import sys
import os


def check_python():
    v = sys.version_info
    print(f"\n[1] Python version: {v.major}.{v.minor}.{v.micro}")
    if v.major < 3 or (v.major == 3 and v.minor < 8):
        print("  ❌ Python 3.8+ required. Please upgrade.")
        sys.exit(1)
    else:
        print("  ✓ Python version OK")


def install(package):
    print(f"    Installing {package}...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", package, "--quiet"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"    ✓ {package}")


def install_libraries():
    print("\n[2] Installing required libraries...")
    libraries = [
        "ultralytics",      # YOLOv8
        "roboflow",         # dataset download
        "opencv-python",    # webcam + image processing
        "pyserial",         # Arduino serial communication
        "matplotlib",       # plots
        "seaborn",          # styled plots
        "pandas",           # CSV data
        "numpy",            # array ops
        "Pillow",           # image I/O
        "tqdm",             # progress bars
        "pyyaml",           # YAML config parsing
    ]
    for lib in libraries:
        try:
            install(lib)
        except Exception as e:
            print(f"    ⚠ Failed: {lib} — {e}")
    print("  ✓ All libraries installed")


def check_webcam():
    print("\n[3] Checking webcam...")
    try:
        import cv2
        for idx in range(3):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    h, w = frame.shape[:2]
                    print(f"  ✓ Webcam found at index {idx} — {w}×{h}")
                    cap.release()
                    return
                cap.release()
        print("  ⚠ No webcam detected. Connect a USB webcam and re-run.")
    except Exception as e:
        print(f"  ⚠ Webcam check failed: {e}")


def check_torch():
    print("\n[4] Checking PyTorch + CUDA...")
    try:
        import torch
        cuda = torch.cuda.is_available()
        print(f"  PyTorch version : {torch.__version__}")
        print(f"  CUDA available  : {cuda}")
        if cuda:
            print(f"  GPU             : {torch.cuda.get_device_name(0)}")
        else:
            print("  ℹ Tip: Use Google Colab for free GPU training")
    except ImportError:
        print("  ⚠ PyTorch not yet installed — it will be installed via ultralytics")


def create_folders():
    print("\n[5] Creating project folder structure...")
    folders = [
        "dataset/images/train",
        "dataset/images/val",
        "dataset/labels/train",
        "dataset/labels/val",
        "models",
        "outputs",
        "runs",
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"  ✓ {folder}/")


def main():
    print("=" * 60)
    print("  ANTI-DRONE YOLOV8 SYSTEM — ENVIRONMENT SETUP")
    print("=" * 60)

    check_python()
    install_libraries()
    check_webcam()
    check_torch()
    create_folders()

    print("\n" + "=" * 60)
    print("  ✅ SETUP COMPLETE!")
    print("=" * 60)
    print("""
Next Steps:
  1. Download dataset:
       Open notebooks/02_dataset_preparation.ipynb

  2. Train YOLOv8:
       python src/train.py
       (or open notebooks/03_yolov8_training.ipynb)

  3. Run detection:
       python src/detect.py --model runs/detect/anti_drone_final/weights/best.pt

  4. Flash Arduino:
       Open arduino/anti_drone_controller.ino in Arduino IDE
    """)


if __name__ == "__main__":
    main()
