# detection/vehicle_detector.py
import os
from ultralytics import YOLO

MODEL_DIR = "models"
YOLONAME = os.path.join(MODEL_DIR, "yolov8n.pt")

class VehicleDetector:
    def __init__(self):
        self.model = None

    def _load(self):
        if self.model is not None:
            return
        if os.path.exists(YOLONAME):
            self.model = YOLO(YOLONAME)
        else:
            # allow ultralytics to auto-download if allowed
            self.model = YOLO("yolov8n")

    def detect(self, frame, conf=0.25):
        """
        frame: BGR numpy image
        returns list of dicts: {'bbox':(x1,y1,x2,y2), 'score':score, 'label':label}
        """
        self._load()
        results = self.model.predict(frame, imgsz=640, conf=conf, verbose=False)
        res = results[0]
        out = []
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            score = float(box.conf[0])
            cls = int(box.cls[0])
            coco_map = {
                0: "person", 1: "bicycle", 2: "car", 3: "motorbike",
                5: "bus", 7: "truck"
            }
            label = coco_map.get(cls, f"cls_{cls}")
            if label in ["car", "motorbike", "bus", "truck", "bicycle"]:
                out.append({"bbox": (x1, y1, x2, y2), "score": score, "label": label})
        return out
