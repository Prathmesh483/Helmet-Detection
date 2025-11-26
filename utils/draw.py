# INSERT drawing utils HERE
import cv2

def draw_boxes(img, vehicles, helmets, plates):
    img = img.copy()

    for v in vehicles:
        x1, y1, x2, y2 = v["bbox"]
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img, v["label"], (x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    for h in helmets:
        x1, y1, x2, y2 = h["bbox"]
        color = (0,255,0) if h["label"]=="Helmet" else (0,0,255)
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        cv2.putText(img, h["label"], (x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    y = 30
    for p in plates:
        cv2.putText(img, p["plate"], (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        y += 25

    return img
