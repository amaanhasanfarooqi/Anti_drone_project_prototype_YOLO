# 🛡️ AI-Based Drone Detection, Tracking & Neutralisation System
### YOLOv8 + Arduino + OpenCV | Real-Time Computer Vision

<p align="center">
  <img src="assets/banner.png" alt="Anti Drone System" width="800"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-blue?style=flat-square&logo=pytorch" />
  <img src="https://img.shields.io/badge/Python-3.8%2B-yellow?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv" />
  <img src="https://img.shields.io/badge/Arduino-Uno-teal?style=flat-square&logo=arduino" />
  <img src="https://img.shields.io/badge/License-MIT-red?style=flat-square" />
</p>

---

## 📌 Project Overview

This is a **real-time AI-powered anti-drone surveillance prototype** that:

1. **Detects** drones from a live webcam feed using a custom-trained **YOLOv8** model
2. **Tracks** the drone's position using **coordinate-based visual servoing**
3. **Controls** a pan-tilt mechanism (MG996R servo motors) via **Arduino Uno** over serial communication
4. **Neutralises** the target by activating a spark-generation module when TARGET LOCK is achieved

> 🎓 **B.Tech Final Year Project** — Department of Mechanical Engineering, Jamia Millia Islamia, New Delhi (2025–26)  
> 👨‍💻 Submitted by: Amaan Hasan Farooqi and team
> 🧑‍🏫 Supervisor: Prof. Mohd Suhaib

---

## 🧠 ML / Computer Vision Focus

This project is heavily focused on the **machine learning pipeline**:

| Stage | Technology | Details |
|-------|-----------|---------|
| Dataset | Roboflow | Custom drone dataset, YOLO-format annotations |
| Model | YOLOv8n (Nano) | Ultralytics, transfer learning on drone class |
| Training | PyTorch + CUDA | 50–100 epochs, imgsz 416, batch 4–16 |
| Inference | OpenCV + Python | Real-time webcam, 18–25 FPS |
| Tracking | Coordinate-based | Error minimisation → servo commands |
| Hardware | Arduino Uno | Serial communication, PWM servo control |

---

## 📁 Repository Structure

```
anti-drone-yolov8/
│
├── 📂 notebooks/
│   ├── 01_environment_setup.ipynb       # Install libs, check webcam, create folders
│   ├── 02_dataset_preparation.ipynb     # Roboflow download, annotation visualisation
│   ├── 03_yolov8_training.ipynb         # Full training pipeline with metrics
│   └── 04_realtime_detection.ipynb      # Live detection + Arduino integration
│
├── 📂 src/
│   ├── setup.py                         # Auto-install all dependencies
│   ├── dataset_utils.py                 # Dataset loading, stats, visualisation
│   ├── train.py                         # YOLOv8 training script (standalone)
│   ├── detect.py                        # Standalone real-time detection
│   └── tracking_controller.py          # Error calc + servo command logic
│
├── 📂 arduino/
│   └── anti_drone_controller.ino       # Full Arduino code (servo + FIRE logic)
│
├── 📂 configs/
│   ├── data.yaml                        # YOLO dataset config template
│   └── training_config.yaml            # Training hyperparameters
│
├── 📂 docs/
│   ├── SYSTEM_ARCHITECTURE.md          # Detailed system design
│   ├── YOLO_DEEP_DIVE.md               # YOLOv8 architecture explained
│   └── HARDWARE_SETUP.md               # Wiring + hardware guide
│
├── 📂 scripts/
│   └── evaluate_model.py               # mAP, precision/recall evaluation
│
├── requirements.txt
├── .gitignore
└── README.md                            ← You are here
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/anti-drone-yolov8.git
cd anti-drone-yolov8
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
Or run the setup script:
```bash
python src/setup.py
```

### 3. Download Dataset (Roboflow)
Edit `configs/data.yaml` with your Roboflow API key, then run:
```bash
python src/dataset_utils.py
```

### 4. Train YOLOv8
```bash
python src/train.py
```
Or open `notebooks/03_yolov8_training.ipynb` in Jupyter.

### 5. Run Real-Time Detection
```bash
python src/detect.py --model runs/detect/anti_drone_final/weights/best.pt
```

### 6. Flash Arduino
Open `arduino/anti_drone_controller.ino` in Arduino IDE and upload to your board.

---

## 🔬 YOLOv8 Model Architecture

```
Input Frame (640×640)
        │
        ▼
┌──────────────────┐
│   Backbone       │  CSPDarknet — Feature Extraction
│   (CSPDarknet)   │  Extracts hierarchical features
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Neck           │  PANNet — Feature Pyramid
│   (PAN-FPN)      │  Multi-scale feature fusion
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Head           │  Anchor-free detection head
│   (Detect)       │  Predicts: boxes + classes + conf
└────────┬─────────┘
         │
         ▼
  Bounding Boxes + Confidence Scores
```

**Why YOLOv8 for Drone Detection?**
- Single-pass inference → real-time speed (18–25 FPS on CPU)
- Improved small-object detection vs. previous YOLO versions
- Anchor-free head → better generalisation on custom datasets
- Seamless Roboflow integration for dataset management

---

## 📊 Training Results

| Metric | Value |
|--------|-------|
| mAP@0.5 | ~0.85+ |
| Confidence Score (avg) | 0.82 – 0.95 |
| FPS (real-time) | 18 – 25 FPS |
| Detection Delay | < 1 second |
| False Positive Rate | Low |

Training parameters:
```yaml
epochs:     10–100
imgsz:      416
batch:      4–16
patience:   5
fliplr:     0.5
device:     GPU (CUDA) / CPU fallback
```

---

## ⚙️ System Workflow

```
📷 HD Camera (1080p)
        │  Live video feed
        ▼
🐍 Python + YOLOv8
        │  Drone detected → bounding box + (cx, cy)
        ▼
📐 Error Calculation
        │  error_x = cx - frame_center_x
        │  error_y = cy - frame_center_y
        ▼
🔌 Serial Communication (USB)
        │  Commands: LEFT / RIGHT / UP / DOWN / FIRE / SCAN / TRACK
        ▼
🤖 Arduino Uno
        │  Controls MG996R pan servo + tilt servo
        ▼
🎯 TARGET LOCK (|error_x| < 20 AND |error_y| < 20)
        │
        ▼
⚡ FIRE → Relay → Spark Module Activated
```

---

## 🔧 Hardware Components

| Component | Specification | Role |
|-----------|-------------|------|
| HD USB Camera | 1080p Webcam | Live video input |
| Arduino Uno | ATmega328P | Embedded controller |
| MG996R Servo × 2 | High-torque servo | Pan + Tilt movement |
| Relay Module | 5V | Electrical switching |
| Spark Generator | High-voltage unit | Neutralisation |
| Power Supply | 5V / 12V DC | System power |

---

## 🧪 Detection Modes

| Mode | Trigger | Behaviour |
|------|---------|-----------|
| `SCAN` | No drone detected | Servos sweep left-right automatically |
| `TRACK` | Drone detected | Servos align with drone centre |
| `TARGET LOCK` | Error < 20px | "TARGET LOCKED" overlay on frame |
| `FIRE` | Lock + cooldown elapsed | Relay activated → spark module fires |

---

## 📦 Requirements

```
ultralytics>=8.0.0
opencv-python>=4.8.0
pyserial>=3.5
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
Pillow>=9.5.0
tqdm>=4.65.0
pyyaml>=6.0
roboflow>=1.1.0
torch>=2.0.0
```

---

## 🎬 Demo

> 📹 See `docs/demo_video_guide.md` for instructions on recording and uploading your system demo video.

---

## 🗺️ Future Scope

- [ ] Replace Arduino with Raspberry Pi / Jetson Nano for edge AI
- [ ] Multi-drone simultaneous tracking
- [ ] Thermal camera integration for night operation
- [ ] Trajectory prediction with Kalman Filter
- [ ] RF + Vision sensor fusion
- [ ] IoT dashboard for remote monitoring
- [ ] YOLOv8-Pose for attitude estimation

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 📚 Citation

If you use this work, please cite:

```
Amaan Hasan Farooqi and team (2026).
AI-Based Drone Detection, Tracking and Neutralisation Prototype using YOLOv8 and Arduino.
B.Tech Project Report, Dept. of Mechanical Engineering, Jamia Millia Islamia, New Delhi.
```

---

## 🙌 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Roboflow](https://roboflow.com) for dataset management
- [OpenCV](https://opencv.org)
- Prof. Mohd Suhaib, Dept. of Mechanical Engineering, JMI
