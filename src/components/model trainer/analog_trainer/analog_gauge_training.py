import torch
from ultralytics import YOLO

def train():
    # --- GPU OPTIMIZATION ---
   
    model = YOLO('yolov8n-pose.pt') 

    # Check if CUDA is actually available
    if torch.cuda.is_available():
        print(f"🚀 GPU Mode: Active - {torch.cuda.get_device_name(0)}")
        device_config = 0 # Use GPU 0
    else:
        print("⚠️ GPU not found. Falling back to CPU.")
        device_config = 'cpu'

    results = model.train(
        data='analog_gauge_data.yaml',     
        epochs=50,
        
        # --- VRAM SAVING SETTINGS (4GB LIMIT) ---
        imgsz=960,            # Reduced from 960 to 640 to fit in memory
        batch=4,              # Low batch size prevents Out Of Memory errors
        device=device_config, # Force GPU usage
        workers=0,            # Keep at 0 for Windows stability
        
        # --- CRITICAL ACCURACY SETTINGS (Kept from your config) ---
        mosaic=0.0,           # DISABLE THIS! (Don't shrink the needle)
        fliplr=0.0,           # DISABLE THIS! (Gauges aren't symmetric)
        mixup=0.0,            # Ensure no ghosting
        copy_paste=0.0,       # Ensure no ghosting
        
        # --- ROBUSTNESS SETTINGS ---
        degrees=10.0,         
        hsv_v=0.4,            
        hsv_s=0.6,            
        scale=0.1,            
        translate=0.1,        
        
        patience=10,          
        augment=True,
        amp=True,             # Automatic Mixed Precision (Faster + Uses less VRAM)
        project='analog_gauge_project',
        name='analog_gauge_gpu_training'
    )

if __name__ == '__main__':
    train()