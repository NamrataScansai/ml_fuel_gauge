import os
import csv
import json  
import shutil
import cv2
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw
from ultralytics import YOLO

# ==========================================
# IMPORT MODULES
# ==========================================
from classification import GaugeClassifier
from analogreader import AnalogGaugeReader
from digitalreader1 import DigitalGaugeReader

# ==========================================
# CONFIGURATION
# ==========================================
# INPUT / OUTPUT
IMAGE_PATH = r"D:\Scanai_modules\FUelGauge_Gemini2\odometer_2k_imgs"
OUTPUT_DIR = r"D:\FuelGauge\Final_1\fuelgauge_utility_output3.1"

# MODEL PATHS
PATH_DETECTOR    = r"D:\FuelGauge\Detection_Model\runs\gauge_v2\gauge_detection_model\weights\best.pt"
PATH_CLASSIFIER  = r"D:\FuelGauge\Classification_Model\runs\classify\train\fuel_gauge_cls\weights\best.pt"
PATH_ANALOG_POSE = r"D:\FuelGauge\Analog_Gauge Model\analogtraining\analog_gauge_project\analog_gauge_gpu_training2\weights\best.pt"
PATH_DIGITAL_EXP = r"D:\FuelGauge\Digital_Gauge Model\fuel_expert_v2.pth"
PATH_DIG_CLASSES = r"D:\FuelGauge\Digital_Gauge Model\classes.txt"

DETECTION_CONF_THRESHOLD = 0.30

class FuelGaugePipeline:
    def __init__(self):
        self.setup_directories()
        self.init_stats()  # <--- Initialize Counters
        self.load_models()
        self.init_csv()

    def setup_directories(self):
        # ===== CATEGORY FOLDERS =====
        self.dirs = {
            "Readable_Digital": os.path.join(OUTPUT_DIR, "readable_digital"),
            "Readable_Analog": os.path.join(OUTPUT_DIR, "readable_analog"),
            "NonReadable_Analog": os.path.join(OUTPUT_DIR, "non_readable_analog"),
            "NonReadable_Digital": os.path.join(OUTPUT_DIR, "non_readable_digital"),
            "no_gauge": os.path.join(OUTPUT_DIR, "no_gauge_found")
        }

        # ===== CENTRALIZED FOLDERS =====
        self.dirs["all_crops"] = os.path.join(OUTPUT_DIR, "all_crops")
        self.dirs["all_bboxes"] = os.path.join(OUTPUT_DIR, "all_bboxes")
        self.dirs["all_jsons"] = os.path.join(OUTPUT_DIR, "all_jsons") 
        self.dirs["final_reading"] = os.path.join(OUTPUT_DIR, "final_reading_vis")

        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)

    def init_stats(self):
        """Initialize the counter dictionary"""
        self.stats = {
            "Total_Processed": 0,
            "Readable_Digital": 0,
            "Readable_Analog": 0,
            "NonReadable_Digital": 0,
            "NonReadable_Analog": 0,
            "No_Gauge_Detected": 0
        }

    def init_csv(self):
        self.csv_file = os.path.join(OUTPUT_DIR, "fuelgauge_summary3.csv")
        self.header = [
            "image_name", 
            "detected", 
            "class_name", 
            "reading_value",  
            "fuel_fraction",  
            "reading_status",
            "crop_path", 
            "bbox_path",
            "det_conf",
            "cls_conf"
        ]
        
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.header)

    def load_models(self):
        print("\n=== LOADING MODELS ===")
        self.detector = YOLO(PATH_DETECTOR)
        self.classifier = GaugeClassifier(PATH_CLASSIFIER)
        self.analog_reader = AnalogGaugeReader(PATH_ANALOG_POSE)
        self.digital_reader = DigitalGaugeReader(PATH_DIGITAL_EXP, PATH_DIG_CLASSES)
        print("=== MODELS READY ===\n")

    def process_all(self):
        image_files = [f for f in os.listdir(IMAGE_PATH) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        total_files = len(image_files)
        print(f"Total images to process: {total_files}")

        for img_name in image_files:
            self.process_single(img_name)

        # After loop finishes, save summary
        self.save_count_summary()

    def process_single(self, img_name):
        img_path = os.path.join(IMAGE_PATH, img_name)
        self.stats["Total_Processed"] += 1 # Increment total

        # Initialize JSON Data structure
        json_data = {
            "filename": img_name,
            "timestamp": datetime.now().isoformat(),
            "detection": {"found": False},
            "classification": None,
            "reading": None,
            "paths": {}
        }

        try:
            im = Image.open(img_path).convert("RGB")
            W, H = im.size
        except Exception as e:
            print(f"Error opening image: {e}")
            return

        # 1. DETECTION
        det_results = self.detector(img_path, verbose=False)
        best_box = None
        best_conf = 0.0

        for box in det_results[0].boxes:
            conf = float(box.conf[0])
            if conf > DETECTION_CONF_THRESHOLD and conf > best_conf:
                best_conf = conf
                best_box = box

        # CASE: NO GAUGE
        if best_box is None:
            self.stats["No_Gauge_Detected"] += 1 # Increment No Gauge
            
            save_path = os.path.join(self.dirs["no_gauge"], img_name)
            shutil.copy(img_path, save_path)
            
            self.log_csv([
                img_name, "no", "N/A", "N/A", "N/A", "No Gauge", "", "", 0.0, 0.0
            ])
            self.save_json(img_name, json_data)
            
            print(f" → {img_name}: No gauge detected")
            return

        # Update JSON with Detection info
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        json_data["detection"] = {
            "found": True,
            "confidence": round(best_conf, 4),
            "bbox": [x1, y1, x2, y2]
        }

        # 2. CROP
        pad = 15
        x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
        x2p, y2p = min(W, x2 + pad), min(H, y2 + pad)
        crop_pil = im.crop((x1p, y1p, x2p, y2p))

        # 3. CLASSIFICATION
        cls_name, cls_conf = self.classifier.predict(crop_pil)
        
        # Increment Classification Stats
        if cls_name in self.stats:
            self.stats[cls_name] += 1
        
        json_data["classification"] = {
            "class": cls_name,
            "confidence": round(cls_conf, 4)
        }

        # 4. SAVE FILES
        target_dir = self.dirs.get(cls_name, self.dirs["no_gauge"])
        shutil.copy(img_path, os.path.join(target_dir, img_name))

        base_name = os.path.splitext(img_name)[0]
        crop_path = os.path.join(self.dirs["all_crops"], f"{base_name}_crop.jpg")
        crop_pil.save(crop_path)

        bbox_img = im.copy()
        draw = ImageDraw.Draw(bbox_img)
        draw.rectangle((x1, y1, x2, y2), outline="green", width=5)
        bbox_path = os.path.join(self.dirs["all_bboxes"], f"{base_name}_bbox.jpg")
        bbox_img.save(bbox_path)

        json_data["paths"] = {
            "crop": crop_path,
            "bbox": bbox_path
        }

        # 5. READING
        crop_cv2 = cv2.cvtColor(np.array(crop_pil), cv2.COLOR_RGB2BGR)
        
        # Defaults
        reading_val = "N/A" 
        fuel_frac = "N/A"
        vis_reading = None
        read_status = "Skipped"

        if "NonReadable" in cls_name:
            read_status = "Blurry/Dark"
            reading_val = "NonReadable"
        
        elif "Analog" in cls_name:
            # Analog: value="1/8", frac="17.0%"
            pct, lbl, _, vis_reading = self.analog_reader.predict(crop_cv2)
            if pct is not None:
                reading_val = lbl                
                fuel_frac = f"{pct:.1f}%"         
                read_status = "Success"
            else:
                read_status = "Analog Read Failed"
                reading_val = "ReadFail"

        elif "Digital" in cls_name:
            # Digital: value="1", frac="100.0%"
            res = self.digital_reader.predict(crop_cv2, img_name)
            
            if res["success"]:
                reading_val = res["fraction_text"]          
                fuel_frac = f"{res['visible_percentage']:.1f}%" 
                vis_reading = res["vis_img"]
                read_status = res["status"]
            else:
                read_status = "Digital Read Failed"
                reading_val = "ReadFail"
                vis_reading = res["vis_img"]

        # Save Visualisation
        if vis_reading is not None:
            vis_path = os.path.join(self.dirs["final_reading"], f"{base_name}_read.jpg")
            cv2.imwrite(vis_path, vis_reading)
            json_data["paths"]["reading_vis"] = vis_path

        # Update JSON
        json_data["reading"] = {
            "value": reading_val,
            "percentage": fuel_frac,
            "status": read_status
        }

        # 6. SAVE JSON & LOG CSV
        self.save_json(img_name, json_data)
        
        self.log_csv([
            img_name,               # image_name
            "yes",                  # detected
            cls_name,               # class_name
            reading_val,            # reading_value
            fuel_frac,              # fuel_fraction
            read_status,            # reading_status
            crop_path,              # crop_path
            bbox_path,              # bbox_path
            round(best_conf, 4),    # det_conf
            round(cls_conf, 4)      # cls_conf
        ])
        
        print(f" → {img_name} | {cls_name} | {reading_val} | {fuel_frac}")

    def save_count_summary(self):
        """Prints counts to console and saves to CSV"""
        csv_path = os.path.join(OUTPUT_DIR, "countsummary.csv")
        
        print("\n" + "="*30)
        print(" PIPELINE PROCESSING SUMMARY ")
        print("="*30)
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Category", "Count"])
            
            for category, count in self.stats.items():
                print(f"{category:<20}: {count}")
                writer.writerow([category, count])
                
        print("="*30)
        print(f"Summary saved to: {csv_path}")

    def save_json(self, img_name, data):
        json_name = os.path.splitext(img_name)[0] + ".json"
        json_path = os.path.join(self.dirs["all_jsons"], json_name)
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)

    def log_csv(self, row_data):
        with open(self.csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row_data)

if __name__ == "__main__":
    pipeline = FuelGaugePipeline()
    pipeline.process_all()
    
    print("\n===== DONE =====")
    print(f"Results saved in: {OUTPUT_DIR}")