import os
import json
import csv
import torch
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from multiprocessing import freeze_support

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = r"D:\FuelGauge\Classification_Model\dataset_cls_new"
MODEL_PATH = r"D:\FuelGauge\Classification_Model\yolov8n-cls.pt"

SAVE_DIR = "runs/classification"
METRICS_JSON = "classification_metrics.json"
METRICS_CSV = "classification_metrics.csv"

BATCH = 16
EPOCHS = 50
IMG_SIZE = 640
NUM_WORKERS = 4   # SAFE after fix
DEVICE = 0 if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    "NonReadable_analog",
    "NonReadable_digital",
    "Readable_analog",
    "Readable_digital"
]

# ============================================================
# MAIN
# ============================================================
def main():
    print("\n========================")
    print(" TRAINING STARTED ")
    print("========================\n")

    model = YOLO(MODEL_PATH)

    model.train(
        data=DATA_DIR,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        workers=NUM_WORKERS,
        device=DEVICE,

        hsv_h=0.0,
        hsv_s=0.2,
        hsv_v=0.2,
        degrees=10,
        translate=0.05,
        scale=0.1,
        shear=5,

        project=SAVE_DIR,
        name="fuel_gauge_classifier",
        exist_ok=True
    )

    print("✅ Training completed")

    # ----------------------------
    # Validation (manual metrics)
    # ----------------------------
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    val_ds = datasets.ImageFolder(
        os.path.join(DATA_DIR, "val"),
        transform=transform
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    y_true, y_pred = [], []
    model.eval()

    for imgs, labels in val_loader:
        imgs_np = [(img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                   for img in imgs]

        preds = model.predict(
            imgs_np,
            imgsz=IMG_SIZE,
            device=DEVICE,
            verbose=False
        )

        for p in preds:
            y_pred.append(int(p.probs.top1))

        y_true.extend(labels.numpy().tolist())

    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    conf_mat = confusion_matrix(y_true, y_pred)

    metrics = model.val(data=DATA_DIR, imgsz=IMG_SIZE, device=DEVICE)

    final_output = {
        "top1_accuracy": float(metrics.top1),
        "top5_accuracy": float(metrics.top5),
        "classification_report": report,
        "confusion_matrix": conf_mat.tolist()
    }

    with open(METRICS_JSON, "w") as f:
        json.dump(final_output, f, indent=4)

    print("\n🎉 DONE — Training + Evaluation complete!")


# ============================================================
# WINDOWS ENTRY POINT (MANDATORY)
# ============================================================
if __name__ == "__main__":
    freeze_support()
    main()
