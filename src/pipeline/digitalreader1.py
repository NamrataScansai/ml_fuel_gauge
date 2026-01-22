import torch
import timm
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

class DigitalGaugeReader:
    def __init__(self, model_path, classes_file_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # print(f"   [Digital] Loading EfficientNet on {self.device}...")
        
        # Load Class Names
        with open(classes_file_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines() if line.strip()]

        # Load EfficientNetV2 Expert
        self.model = timm.create_model('tf_efficientnetv2_s', pretrained=False, num_classes=len(self.class_names))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()

        # The Precision Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # print("   [Digital] Model Loaded.")

    def map_fuel_level(self, p):
        """Maps percentage to standard 1/8th fractions (Same as Analog Logic)"""
        if p <= 6:    return "0"
        elif p <= 18: return "1/8"
        elif p <= 31: return "1/4"
        elif p <= 43: return "3/8"
        elif p <= 56: return "1/2"
        elif p <= 68: return "5/8"
        elif p <= 81: return "3/4"
        elif p <= 93: return "7/8"
        else:         return "1"

    def predict(self, crop_bgr, image_name="Unknown"):
        try:
            # Prepare Image
            crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
            input_tensor = self.transform(crop_pil).unsqueeze(0).to(self.device)

            # Inference
            with torch.no_grad():
                output = self.model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                conf, pred_idx = torch.max(probs, 1)

            # Parsing Logic
            class_str = self.class_names[pred_idx.item()]
            
            # Defaults
            bar_text = class_str
            pct = 0.0

            # Parse "Filled_Total" (e.g., "12_12")
            if "_" in class_str:
                parts = class_str.split('_')
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    filled = int(parts[0])
                    total = int(parts[1])
                    if total > 0:
                        pct = (filled / total) * 100
                    bar_text = f"{filled}/{total}"
            
            # Get Standard Fraction (0, 1/8, ... 1) based on percentage
            fraction_str = self.map_fuel_level(pct)
            
            # Logic for Status
            status = "Verified" if conf.item() > 0.60 else "Review Required"

            # Visualization
            vis_img = crop_bgr.copy()
            # Overlay: "12/12 (1)"
            cv2.putText(vis_img, f"{bar_text} ({fraction_str})", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            
            return {
                "bar_count": bar_text,         # e.g. "12/12"
                "visible_percentage": pct,     # e.g. 100.0 (float)
                "fraction_text": fraction_str, # e.g. "1" or "1/2"
                "confidence": conf.item(),
                "status": status,
                "vis_img": vis_img,
                "success": True
            }

        except Exception as e:
            print(f"   [Digital] Error: {e}")
            return {
                "bar_count": "Error",
                "visible_percentage": 0.0,
                "fraction_text": "Error",
                "confidence": 0.0,
                "status": "Error",
                "vis_img": crop_bgr,
                "success": False
            }