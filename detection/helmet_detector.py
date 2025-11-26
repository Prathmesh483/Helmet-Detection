# detection/helmet_detector.py
import os
from ultralytics import YOLO
import cv2

MODEL_DIR = "models"
SPECIAL_HELMET_MODEL = os.path.join(MODEL_DIR, "helmet_yolo.pt")  # optional specialized helmet model
FALLBACK_DETECTOR_NAME = "yolov8n.pt"  # we rely on download_models.py to supply this

class HelmetDetector:
    def __init__(self):
        self.model = None
        self.loaded_with = None

    def _load(self):
        if self.model is not None:
            return
        # Prefer a specialized helmet model if present
        if os.path.exists(SPECIAL_HELMET_MODEL):
            try:
                self.model = YOLO(SPECIAL_HELMET_MODEL)
                self.loaded_with = SPECIAL_HELMET_MODEL
                return
            except Exception as e:
                print("Failed to load specialized helmet model:", e)
        # Fallback: load general-purpose yolov8n for person/motorbike detection
        yolopath = os.path.join(MODEL_DIR, FALLBACK_DETECTOR_NAME)
        if os.path.exists(yolopath):
            try:
                self.model = YOLO(yolopath)
                self.loaded_with = yolopath
                return
            except Exception as e:
                print("Failed to load local yolov8n:", e)
        # Final fallback: let ultralytics auto-download model by name (this requires net access on the host)
        try:
            self.model = YOLO("yolov8n")  # ultralytics can auto-download
            self.loaded_with = "yolov8n(hub)"
        except Exception as e:
            print("Couldn't auto-load yolov8n:", e)
            raise RuntimeError("No detection model available. Run download_models.py or allow network download.")

    def detect(self, frame, conf=0.25):
        """
        Input: BGR numpy frame (OpenCV)
        Output: list of helmet detection dicts (for now: those are aligned to vehicle/person bboxes)
        Example:
            [{'bbox':(x1,y1,x2,y2), 'score':0.9, 'label':'motorbike', 'helmet':'unknown'|'yes'|'no'}]
        """
        self._load()
        results = self.model.predict(frame, imgsz=640, conf=conf, verbose=False)
        res = results[0]
        outputs = []
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            score = float(box.conf[0])
            cls = int(box.cls[0])
            # COCO class mapping for common vehicles/persons
            coco_map = {
                0: "person", 1: "bicycle", 2: "car", 3: "motorbike",
                5: "bus", 7: "truck"
            }
            label = coco_map.get(cls, f"cls_{cls}")
            # Helmet decision:
            # If we had a specialized helmet model we'd use it to confirm helmets.
            # For now we set 'unknown' for helmets from generic COCO detections.
            helmet_status = "unknown"
            outputs.append({"bbox": (x1, y1, x2, y2), "score": score, "label": label, "helmet": helmet_status})
        return outputs
