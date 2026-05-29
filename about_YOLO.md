# 🧠 YOLOv8 Deep Dive — Anti-Drone Detection

This document explains the YOLOv8 computer vision pipeline in detail, written specifically for understanding how it works in this drone detection project.

---

## 1. What Is YOLO?

**YOLO (You Only Look Once)** is a family of real-time object detection algorithms. Unlike older two-stage detectors (like Faster R-CNN, which first proposes regions then classifies them), YOLO processes the entire image **in a single forward pass** through the neural network — making it extremely fast.

| Algorithm       | Approach    | Speed   | Accuracy |
|----------------|------------|--------|---------|
| Faster R-CNN   | Two-stage   | Slow   | Very High |
| SSD            | One-stage   | Fast   | Medium |
| YOLOv8         | One-stage   | Very Fast | High |

For drone detection, **speed is critical** — a 1-second delay between detection and servo response means the drone has already moved. YOLOv8 achieves 18–25 FPS even on CPU, making it ideal for this system.

---

## 2. YOLOv8 Architecture

YOLOv8 (developed by [Ultralytics](https://github.com/ultralytics/ultralytics)) has three main components:

```
┌─────────────────────────────────────────────────────────────────┐
│                       INPUT IMAGE (416×416)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        BACKBONE                                  │
│                      (CSPDarknet)                                │
│                                                                  │
│  Layer 1 → 2 → 3 → 4 → 5 → 6 → ... → N                        │
│  Convolutional layers extract hierarchical features:             │
│    • Early layers  → edges, textures                             │
│    • Middle layers → shapes, parts                               │
│    • Deep layers   → "drone-like" patterns                       │
│                                                                  │
│  Output: C3, C4, C5 feature maps at different scales             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          NECK                                    │
│                       (PAN-FPN)                                  │
│                                                                  │
│  Feature Pyramid Network (FPN): top-down pathway                 │
│    • Combines deep semantic info with spatial detail             │
│                                                                  │
│  Path Aggregation Network (PAN): bottom-up pathway               │
│    • Improves small-object detection                             │
│    • Critical for drones (small, fast-moving objects)            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        DETECTION HEAD                            │
│                      (Anchor-Free)                               │
│                                                                  │
│  Three detection scales (small, medium, large objects)           │
│  For each grid cell, predicts:                                   │
│    • (x, y, w, h) → bounding box                                │
│    • objectness score                                            │
│    • class probabilities                                         │
│                                                                  │
│  YOLOv8 is ANCHOR-FREE (unlike v5):                              │
│    → no hand-tuned anchor boxes needed                           │
│    → better generalisation on custom datasets                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
             Bounding Boxes + Confidence Scores
```

---

## 3. How the Drone Dataset Works

### YOLO Annotation Format

Every training image has a corresponding `.txt` label file:

```
drone_001.jpg
drone_001.txt   ← annotation
```

Each line in the `.txt` file is one object:
```
<class_id> <cx> <cy> <width> <height>
```

All values are **normalised** (0.0 to 1.0) relative to image dimensions.

**Example:**
```
0 0.512 0.345 0.124 0.089
```
- Class 0 = `drone`
- Centre X = 51.2% from left
- Centre Y = 34.5% from top
- Box width = 12.4% of image width
- Box height = 8.9% of image height

### Why Normalised Coordinates?

Normalisation makes the labels **resolution-independent**. The same label works whether the image is 416×416 or 640×640, which is essential when training with `imgsz` augmentation.

---

## 4. Transfer Learning (Why We Use `yolov8n.pt`)

Training from scratch requires millions of images and weeks of compute. Instead, we use **transfer learning**:

1. `yolov8n.pt` is pre-trained on **COCO** (80 classes, 330,000+ images)
2. The model has already learned to detect general objects
3. We **fine-tune** it on our small drone dataset (~300–500 images)
4. Only the final detection head needs to adapt to "drone" as a class

This gives high accuracy even with limited data — critical for a B.Tech project where collecting thousands of images isn't feasible.

---

## 5. Training Process Explained

```python
model = YOLO("yolov8n.pt")

results = model.train(
    data    = "configs/data.yaml",
    epochs  = 50,          # full passes through dataset
    imgsz   = 416,         # resize all images to 416×416
    batch   = 8,           # 8 images per gradient update
    fliplr  = 0.5,         # augmentation: random horizontal flip
    mosaic  = 1.0,         # combine 4 images per sample
    patience= 10,          # stop early if no improvement
)
```

### Key Training Parameters

| Parameter | What it Does | Recommended Value |
|-----------|-------------|------------------|
| `epochs`  | Passes through dataset | 50–100 |
| `imgsz`   | Input resolution | 416 (CPU) / 640 (GPU) |
| `batch`   | Samples per update | 4 (CPU) / 16 (GPU) |
| `patience`| Early stopping | 10 |
| `fliplr`  | Horizontal flip aug | 0.5 |
| `mosaic`  | 4-image mosaic aug | 1.0 |

---

## 6. Data Augmentation

YOLOv8 applies augmentation **on-the-fly** during training to increase effective dataset size and prevent overfitting:

| Augmentation | Effect | Why It Helps |
|-------------|--------|-------------|
| Horizontal flip | Mirror image | Drones can fly left or right |
| HSV jitter | Change hue/brightness | Handle different lighting |
| Mosaic | 4 images in 1 | Exposes model to small objects |
| Scale jitter | Zoom in/out | Handle drones at different distances |
| Random crop | Partial objects | Handle partial views |

---

## 7. Inference: How Detection Works in Real-Time

```python
results = model.predict(source=frame, conf=0.7, verbose=False)
boxes   = results[0].boxes

for box in boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])   # pixel coordinates
    cx = (x1 + x2) // 2                        # drone centre X
    cy = (y1 + y2) // 2                        # drone centre Y
    conf = float(box.conf[0])                  # confidence score
```

**Non-Maximum Suppression (NMS)** is applied automatically:
- If multiple overlapping boxes detect the same drone, only the highest-confidence one is kept
- Threshold: IoU > 0.45 → suppress duplicate

**Confidence threshold = 0.7:**
- Only detections with ≥ 70% confidence are kept
- Reduces false positives (non-drone objects mistakenly detected)
- Lower = more detections but more false positives

---

## 8. Visual Servoing: Closing the Loop

The pixel coordinates from YOLOv8 are used directly for servo control:

```
Frame (640×480 pixels)
┌────────────────────────────────────────────────┐
│                                                │
│                                                │
│          [drone detected]                      │
│          centre: (cx=420, cy=180)              │
│                                                │
│                    × ← frame centre (320, 240) │
│                                                │
└────────────────────────────────────────────────┘

error_x = cx - frame_cx = 420 - 320 = +100   → too far RIGHT → send "RIGHT"
error_y = cy - frame_cy = 180 - 240 = -60    → too far UP    → send "UP"
```

**TARGET LOCK** is achieved when:
```
|error_x| < 20 AND |error_y| < 20
```
meaning the drone is within a 40×40 pixel "bullseye" at the frame centre.

---

## 9. Model Metrics Explained

| Metric | Formula | What It Means |
|--------|--------|--------------|
| **Precision** | TP / (TP + FP) | Of all "drone" detections, how many are actually drones? |
| **Recall** | TP / (TP + FN) | Of all actual drones, how many were detected? |
| **mAP@0.5** | Mean AP at IoU=0.5 | Overall detection quality — main benchmark |
| **F1 Score** | 2 × P×R / (P+R) | Balanced metric for unequal classes |

**Our results:**
- Confidence: 0.82–0.95 (high confidence, low false positives)
- FPS: 18–25 (real-time on CPU)
- Detection delay: <1 second

---

## 10. Why YOLOv8 vs Other Detectors?

| System | Speed | Accuracy | Ease of Use | Notes |
|--------|-------|----------|------------|-------|
| **YOLOv8** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **Used in this project** |
| Faster R-CNN | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | Too slow for real-time |
| SSD | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Lower accuracy on small objects |
| YOLOv5 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Previous generation |

For **real-time embedded drone detection**, YOLOv8 Nano is the optimal choice.
