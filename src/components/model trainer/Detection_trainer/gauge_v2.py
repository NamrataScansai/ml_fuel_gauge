import gc
import torch
from ultralytics import YOLO
import multiprocessing

# ======================================
# GAUGE DETECTION TRAINING SCRIPT (FINAL)
# ======================================
# Model: YOLO11 Nano (Fastest)
# Hardware: GTX 1650
# Resolution: 960px (High detail for needles)
# ======================================

MODEL_PATH = "yolo11n.pt"     
DATA_YAML = r"D:\FuelGauge\Detection_Model\dataset_det\data.yaml"


EPOCHS = 50                 
IMG_SIZE = 960

# BATCH = 8 is safer for GTX 1650 with 960px images in FP32 mode.
# If you get "CUDA Out of Memory", change this to 4 or 6.
BATCH = 4                   
WORKERS = 2                   

if __name__ == '__main__':
    # This line prevents the recursive process crash on Windows
    multiprocessing.freeze_support()
   # 1. Clear GPU Cache manually before starting
    torch.cuda.empty_cache()
    gc.collect()

    print("--- Starting Training on GTX 1650 ---")
    
    model = YOLO(MODEL_PATH)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        workers=WORKERS,
        project="runs/gauge_v2",
        name="gauge_detection_model",

        # ---- Augmentations for Gauges ----
        mosaic=1.0,      # Good for detection
        mixup=0.1,
        
        # CRITICAL: Set fliplr to 0.0. 
        # Fuel gauges are not symmetrical (Empty is Left, Full is Right).
        # Flipping them confuses the model.
        fliplr=0.0,      
        flipud=0.0,
        
        hsv_s=0.7,       # Help with different lighting conditions
        hsv_v=0.7,
        scale=0.15,      # Slight zooming in/out

        # ---- Hardware Settings ----
        cache='ram',     # Keeps images in RAM for speed
        amp=True,        # Will auto-disable if GPU doesn't support it (as seen in logs)
        patience=10,     # Stop early if no improvement
        exist_ok=True    # Overwrite existing run folder if needed
    )

    print("Training finished successfully.")
    print(f"Best model saved at: runs/gauge_v2/gauge_detection_model/weights/best.pt")