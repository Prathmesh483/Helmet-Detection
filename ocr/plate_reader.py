import easyocr
import cv2

class PlateReader:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])

    def read_plates(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results = self.reader.readtext(gray)

        plates = []
        for bbox, text, score in results:
            if len(text) >= 4:  
                plates.append({"plate": text, "confidence": score})
        return plates
