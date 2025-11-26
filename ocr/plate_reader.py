# ocr/plate_reader.py
import easyocr
import cv2
import numpy as np

class PlateReader:
    def __init__(self):
        # GPU disabled by default so it works on CPU hosts
        try:
            self.reader = easyocr.Reader(['en'], gpu=False)
        except Exception as e:
            print("EasyOCR init failed:", e)
            self.reader = None

    def read_plates(self, frame):
        """
        frame: BGR image
        returns: list of detected text strings (best-effort)
        """
        if self.reader is None:
            return []
        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.reader.readtext(rgb, detail=0)
        return results
