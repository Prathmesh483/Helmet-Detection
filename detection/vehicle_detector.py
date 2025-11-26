from ultralytics import YOLO
import cv2

class VehicleDetector:
    def __init__(self):
        self.model = YOLO("models/yolo_vehicle.pt")

    def detect(self, img):
        results = self.model.predict(img)[0]
        detections = []

        for box in results.boxes.data.tolist():
            x1, y1, x2, y2, score, cls = box
            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "label": "Vehicle",
                "confidence": float(score)
            })
        return detections
