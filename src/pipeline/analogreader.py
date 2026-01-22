import cv2
import math
import numpy as np
import torch
from ultralytics import YOLO

class AnalogGaugeReader:
    def __init__(self, pose_model_path, conf_thres=0.30):
        print(f"   [Analog] Loading YOLO Pose...")
        self.model = YOLO(pose_model_path)
        self.conf_thres = conf_thres
        
        # Constants
        self.KP_PIVOT = 0
        self.KP_TIP = 1
        self.KP_EMPTY = 2
        self.KP_FULL = 3

    def calculate_pose_percentage(self, pivot, p_empty, p_full, p_tip):
        # (Your exact geometry logic)
        def get_angle(v):
            return (math.degrees(math.atan2(v[1], v[0])) + 360) % 360

        a_e = get_angle(p_empty - pivot)
        a_f = get_angle(p_full  - pivot)
        a_t = get_angle(p_tip   - pivot)

        diff = (a_f - a_e) % 360
        is_clockwise = diff < 180
        
        def normalize(angle, anchor):
            res = (angle - anchor) % 360
            if res > 180: res -= 360 
            return res

        if is_clockwise:
            range_val = normalize(a_f, a_e)
            needle_val = normalize(a_t, a_e)
        else:
            range_val = normalize(a_e, a_f)
            needle_val = normalize(a_e, a_t)

        if abs(range_val) < 1e-3: return 0.0
        pct = needle_val / range_val
        return max(0.0, min(100.0, pct * 100.0))

    def map_fuel_level(self, p):
        if p <= 6:    return "0"
        elif p <= 18: return "1/8"
        elif p <= 31: return "1/4"
        elif p <= 43: return "3/8"
        elif p <= 56: return "1/2"
        elif p <= 68: return "5/8"
        elif p <= 81: return "3/4"
        elif p <= 93: return "7/8"
        else:         return "1"

    def predict(self, crop_img):
        """
        Input: BGR numpy array
        Output: percent (float), label (str), confidence (dummy), visualization (img)
        """
        results = self.model(crop_img, conf=self.conf_thres, verbose=False)
        
        if not results or not results[0].keypoints or results[0].keypoints.data.shape[0] == 0:
            return None, "Pose Fail", 0.0, crop_img

        kp = results[0].keypoints.data[0].cpu().numpy()
        
        if kp.shape[0] < 4:
            return None, "Missing Pts", 0.0, crop_img

        p_pivot = kp[self.KP_PIVOT][:2]
        p_tip   = kp[self.KP_TIP][:2]   
        p_empty = kp[self.KP_EMPTY][:2]
        p_full  = kp[self.KP_FULL][:2]

        pct = self.calculate_pose_percentage(p_pivot, p_empty, p_full, p_tip)
        label = self.map_fuel_level(pct)
        
        # Visualisation
        vis = crop_img.copy()
        cv2.line(vis, tuple(p_pivot.astype(int)), tuple(p_tip.astype(int)), (0, 0, 255), 3, cv2.LINE_AA)
        cv2.circle(vis, tuple(p_empty.astype(int)), 5, (255, 0, 0), -1) 
        cv2.circle(vis, tuple(p_full.astype(int)), 5, (0, 255, 0), -1)
        cv2.putText(vis, f"{label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return pct, label, 0.99, vis