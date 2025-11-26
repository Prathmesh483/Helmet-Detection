from ultralytics import YOLO

class HelmetDetector:
    def __init__(self):
        # Use pretrained model from Ultralytics Hub
        self.model = YOLO("yolov8n.pt")  # small, cloud-friendly

    def detect(self, img):
        results = self.model.predict(img)[0]
        detections = []

        for box in results.boxes.data.tolist():
            x1, y1, x2, y2, score, cls = box

            # Dummy helmet logic (for demo)
            label = "Helmet" if float(score) > 0.5 else "No Helmet"

            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "label": label,
                "confidence": float(score)
            })
        return detections
