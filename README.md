#ML_Fuel_Gauge
# ⛽ Fuel Gauge Reading ML Platform

An **industry-grade Machine Learning platform** for automatic fuel gauge reading from vehicle dashboard images.  
Supports **analog and digital fuel gauges** with a modular, scalable, and deployment-ready architecture.

---
## 🧠 ML Pipeline Architecture

```
Dashboard Image
      ↓
YOLOv11s – Gauge Detection
      ↓
Gauge Crop
      ↓
YOLOv8n – Gauge Classification
      ↓
┌─────────────────────┬─────────────────────┐
│ Analog Gauge        │ Digital Gauge       │
│ YOLOv8n (Pose)      │ EfficientNet-B0     │
│ Needle + E/F + Pivot│ Filled Bar Analysis │
└─────────────────────┴─────────────────────┘
      ↓
Fuel Percentage/Fuel Fraction
      ↓
JSON | CSV | Visual Output
```

---

## 🧩 Models Used

| Task | Model |
|---|---|
| Gauge Detection | YOLOv11s |
| Gauge Classification | YOLOv8n |
| Analog Gauge Reading | YOLOv8n (Pose Estimation) |
| Digital Gauge Reading | EfficientNet-B0 |

---

## 🚀 Key Capabilities

- 🔍 Fuel gauge detection from dashboard images
- 🧠 Gauge classification (Analog / Digital / Readable / Non-readable)
- 📈 Analog needle angle to fuel percentage conversion
- 🔢 Digital bar / numeric fuel level estimation
- 🧩 Modular ML pipeline architecture
- ♻ Independent model retraining
- 🏭 Industry-ready for production deployment

---

## 🏗️ Repository Structure

```text
M_FUEL_GAUGE/
│
├── src/
│   ├── components/                     # Model training modules
│   │   ├── model_trainer/               # Shared training utilities
│   │   ├── classifier_trainer/          # Gauge type classifier training
│   │   ├── Detection_trainer/           # Gauge detection model training
│   │   └── digital_trainer/             # Digital gauge model training
│   │
│   ├── data/                            # Datasets & inference outputs
│   │   ├── odometer_2k_images/
│   │   └── output/
│   │
│   ├── pipeline/                        # Production inference pipeline
│   │   ├── analogreader.py              # Analog gauge reading logic
│   │   ├── classification.py            # Gauge classification inference
│   │   ├── digitalreader1.py            # Digital gauge reading logic
│   │   └── finalfuelgauge.py            # End-to-end pipeline orchestrator
│   │
│   ├── exception.py                     # Custom exception handling
│   ├── logger.py                        # Centralized logging
│   └── utils.py                         # Shared helper utilities
│
├── venv/                                # Virtual environment (ignored)
├── requirements.txt                     # Python dependencies
├── .gitignore                           # Git ignore rules
└── README.md                            # Project documentation

## 🧠 Component Overview
-🔹 Detection Trainer

-Trains YOLO-based models to detect fuel gauges

-Outputs bounding boxes

-🔹 Classifier Trainer

-Classifies gauges as:

-Analog / Digital

-Readable / Non-readable

-🔹 Analog Reader

-Detects needle geometry

-Converts angle into calibrated fuel percentage

-🔹 Digital Reader

-Detects bars or digits

-Computes visible fuel percentage

-🔹 Final Pipeline

-Single entry point for inference

-Orchestrates detection → classification → reading

## ▶️ Running Inference
python src/pipeline/finalfuelgauge.py --image path/to/image.jpg

## Example Output
{
  "gauge_type": "analog",
  "fuel_level": "65%",
  "confidence": 0.94
}

## 🧪 Model Training

Each model is trained independently:

python src/components/model trainer/Detection_trainer/gauge_v2.py
python src/components/model trainer/classifier_trainer/Classifier_type_trainer.py
python src/components/model trainer/digital_trainer/digitalfuel_trainer_v2.py
python src/components/model trainer/analog_trainer/analog_gauge_training.py

✔ Enables modular upgrades
✔ No pipeline refactor needed

