# utils/draw.py
import cv2

def draw_boxes(frame, vehicles, helmets=None, plates=None):
    """
    frame: BGR numpy image
    vehicles: list of dicts {bbox,label,score}
    helmets: list/dict aligned to vehicles (optional)
    plates: list of OCR strings (optional)
    returns: RGB image (uint8)
    """
    img = frame.copy()
    # draw vehicles
    for v in vehicles:
        x1, y1, x2, y2 = v["bbox"]
        label = v.get("label", "")
        score = v.get("score", 0.0)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        txt = f"{label} {score:.2f}"
        cv2.putText(img, txt, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    # helmets overlay (if any)
    if helmets:
        for h in helmets:
            x1, y1, x2, y2 = h.get("bbox", (0,0,0,0))
            helmet_status = h.get("helmet", "unknown")
            color = (0,255,0) if helmet_status == "yes" else ((0,165,255) if helmet_status == "unknown" else (0,0,255))
            cv2.putText(img, f"Helmet:{helmet_status}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    # plates overlay
    if plates:
        y = 30
        for p in plates:
            cv2.putText(img, f"Plate: {p}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
            y += 24
    # convert BGR -> RGB for Streamlit
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb
