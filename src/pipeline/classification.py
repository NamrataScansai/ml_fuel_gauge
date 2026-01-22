from ultralytics import YOLO

class GaugeClassifier:
    def __init__(self, model_path):
        print(f"   [Classifier] Loading YOLO: {model_path}...")
        self.model = YOLO(model_path)
        self.names = self.model.names
        print(f"   [Classifier] Loaded classes: {self.names}")

    def predict(self, pil_crop):
        """
        Input: PIL Image object
        Output: class_name (str), confidence (float)
        """
        # Verbose=False prevents YOLO from printing every prediction to console
        results = self.model(pil_crop, verbose=False)
        
        # Extract Top-1 prediction
        top1_index = int(results[0].probs.top1)
        conf = float(results[0].probs.top1conf)
        class_name = self.names[top1_index]
        
        return class_name, conf