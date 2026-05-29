# 🏗️ System Architecture — Anti-Drone Detection System

## Full System Block Diagram

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          DETECTION & PROCESSING UNIT                        │
│                                                                              │
│  ┌──────────────┐     ┌────────────────┐     ┌───────────────────────────┐ │
│  │  HD Camera   │────▶│  Python Host   │────▶│   YOLOv8 Inference        │ │
│  │  (1080p USB) │     │  (OpenCV)      │     │   model.predict(frame)    │ │
│  └──────────────┘     └────────────────┘     └───────────┬───────────────┘ │
│                                                           │                 │
│                               ┌───────────────────────────┘                │
│                               ▼                                             │
│                      Bounding Box (x1,y1,x2,y2)                            │
│                      Centre (cx, cy)                                        │
│                      Confidence score                                       │
└───────────────────────────────────┬────────────────────────────────────────┘
                                    │
                                    │  Serial (USB, 115200 baud)
                                    │  Commands: LEFT / RIGHT / UP / DOWN
                                    │            FIRE / SCAN / TRACK
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                          TRACKING & CONTROL UNIT                            │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                     Arduino Uno (ATmega328P)                          │  │
│  │                                                                       │  │
│  │  if command == "SCAN"  → pan servo sweeps left/right automatically   │  │
│  │  if command == "LEFT"  → panPos  -= 2°                               │  │
│  │  if command == "RIGHT" → panPos  += 2°                               │  │
│  │  if command == "UP"    → tiltPos += 2°                               │  │
│  │  if command == "DOWN"  → tiltPos -= 2°                               │  │
│  │  if command == "FIRE"  → digitalWrite(RELAY_PIN, HIGH) 500ms         │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│             │ (PWM)                      │ (PWM)         │ (Digital)        │
│             ▼                            ▼               ▼                  │
│    ┌──────────────────┐      ┌────────────────┐   ┌────────────┐           │
│    │  Pan Servo       │      │  Tilt Servo    │   │  Relay 5V  │           │
│    │  (MG996R, Pin 9) │      │  (MG996R,Pin10)│   │  (Pin 7)   │           │
│    └────────┬─────────┘      └───────┬────────┘   └─────┬──────┘           │
│             │                        │                   │                  │
└─────────────┼────────────────────────┼───────────────────┼──────────────────┘
              │                        │                   │
              ▼                        ▼                   ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                       PAN-TILT MECHANISM + NEUTRALISATION                   │
│                                                                              │
│   ┌──────────────────────────────────────────┐   ┌──────────────────────┐  │
│   │         Pan-Tilt Assembly                 │   │  Spark Generator     │  │
│   │  ┌────────────┐  ┌────────────────────┐  │   │  (Flyback + Gap)     │  │
│   │  │ Pan Servo  │  │    Camera          │  │   │  Activated by relay  │  │
│   │  │ (H-axis)   │  │    (1080p)         │  │   └──────────────────────┘  │
│   │  └────────────┘  └────────────────────┘  │                             │
│   │  ┌────────────┐  ┌────────────────────┐  │                             │
│   │  │ Tilt Servo │  │  Launcher Module   │  │                             │
│   │  │ (V-axis)   │  │  (aligned w/cam)   │  │                             │
│   │  └────────────┘  └────────────────────┘  │                             │
│   └──────────────────────────────────────────┘                             │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
1. Camera captures frame (640×480 @ ~25 FPS)
2. YOLOv8 predicts: [x1, y1, x2, y2, class=drone, conf=0.87]
3. Python computes: cx = (x1+x2)/2, cy = (y1+y2)/2
4. Error: error_x = cx - 320, error_y = cy - 240
5. Decision:
   ├─ error_x > 30  → serial.write("RIGHT\n")
   ├─ error_x < -30 → serial.write("LEFT\n")
   ├─ error_y > 30  → serial.write("DOWN\n")
   ├─ error_y < -30 → serial.write("UP\n")
   └─ |error_x|<20 AND |error_y|<20 → serial.write("FIRE\n")
6. Arduino receives command, updates servo angle
7. Servo moves camera+launcher → repeat from step 1
```

---

## Software Architecture

```
anti-drone-yolov8/
│
├── src/
│   ├── setup.py              → Environment & dependency setup
│   ├── dataset_utils.py      → Roboflow download, dataset stats, visualisation
│   ├── train.py              → YOLOv8 fine-tuning pipeline
│   ├── detect.py             → Real-time webcam detection loop
│   └── tracking_controller.py → Visual servo controller (error → commands)
│
├── scripts/
│   └── evaluate_model.py     → Validation: mAP, Precision, Recall
│
├── notebooks/
│   ├── 01_environment_setup.ipynb
│   ├── 02_dataset_preparation.ipynb
│   ├── 03_yolov8_training.ipynb
│   └── 04_realtime_detection.ipynb
│
├── arduino/
│   └── anti_drone_controller.ino  → Pan/tilt servo + relay fire logic
│
└── configs/
    ├── data.yaml            → YOLOv8 dataset config
    └── training_config.yaml → Hyperparameter reference
```

---

## State Machine (System Modes)

```
         ┌─────────────────────────────────────┐
         │              BOOT                    │
         │  Servos → 90° (centre)               │
         │  Relay  → OFF                        │
         └──────────────────┬──────────────────┘
                            │
                            ▼
         ┌─────────────────────────────────────┐
         │              SCAN                    │◀─────────────┐
         │  Pan servo sweeps L↔R               │              │
         │  Waiting for drone detection         │   No drone   │
         └──────────────────┬──────────────────┘              │
                            │ Drone detected (conf ≥ 0.7)     │
                            ▼                                  │
         ┌─────────────────────────────────────┐              │
         │              TRACK                   │──────────────┘
         │  Compute error_x, error_y            │
         │  Send LEFT/RIGHT/UP/DOWN to servos   │
         │  Update every frame (~25 FPS)        │
         └──────────────────┬──────────────────┘
                            │ |error_x|<20 AND |error_y|<20
                            ▼
         ┌─────────────────────────────────────┐
         │           TARGET LOCK                │
         │  Display "TARGET LOCKED" on screen   │
         │  Check fire cooldown timer           │
         └──────────────────┬──────────────────┘
                            │ Cooldown elapsed
                            ▼
         ┌─────────────────────────────────────┐
         │               FIRE                   │
         │  Send FIRE command to Arduino        │
         │  Relay ON → Spark module activated  │
         │  500ms → Relay OFF                   │
         │  Reset cooldown timer (3s)           │
         └─────────────────────────────────────┘
```

---

## Wiring Diagram (Arduino)

```
Arduino Uno
┌──────────────────────────────────────────────────────┐
│                                                      │
│  Pin 9  ────────────► Pan Servo MG996R (Signal)      │
│  Pin 10 ────────────► Tilt Servo MG996R (Signal)     │
│  Pin 7  ────────────► Relay Module (IN)              │
│                                                      │
│  5V     ────────────► Relay VCC                      │
│  GND    ────────────► Relay GND                      │
│                                                      │
│  USB    ◄───────────► Computer (Python serial)       │
│                                                      │
└──────────────────────────────────────────────────────┘

External Power (5V, 2A+):
  Servo VCC ─── External 5V (do NOT power servos from Arduino 5V)
  Servo GND ─── Arduino GND (common ground)

Relay Output:
  NO  ──► Spark Generator +
  COM ──► External Battery +
  External Battery – ──► Spark Generator –
```
