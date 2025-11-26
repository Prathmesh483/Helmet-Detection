# app.py
import streamlit as st
import cv2
import numpy as np
import tempfile
import torch
from ultralytics import YOLO

# utils for drawing boxes
def draw_boxes(frame, vehicles=[], helmets=[], plates=[]):
    img = frame.copy()
    # Draw vehicles in blue
    for box in vehicles:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, "Vehicle", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
    # Draw helmets in green
    for box in helmets:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, "Helmet", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    # Plates in red
    for box in plates:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, "Plate", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    return img

# Streamlit page config
st.set_page_config(page_title="Traffic AI System", layout="wide")
st.title("🚦 Traffic AI Analysis System (Demo)")

# Function to load models safely
@st.cache_resource
def get_models():
    helmet_model = None
    vehicle_model = None
    try:
        # Allowlist Sequential class and disable weights_only
        with torch.serialization.add_safe_globals([torch.nn.modules.container.Sequential]):
            helmet_model = YOLO("yolov8n.pt", weights_only=False)
    except Exception as e:
        st.error("HelmetDetector initialization error: " + str(e))
        helmet_model = None

    try:
        with torch.serialization.add_safe_globals([torch.nn.modules.container.Sequential]):
            vehicle_model = YOLO("yolov8n.pt", weights_only=False)
    except Exception as e:
        st.error("VehicleDetector initialization error: " + str(e))
        vehicle_model = None

    return helmet_model, vehicle_model

# Load models
helmet_model, vehicle_model = get_models()

# Sidebar settings
st.sidebar.header("Settings")
conf = st.sidebar.slider("Detection confidence", 0.1, 0.9, 0.25)

# Image uploader
uploaded = st.file_uploader("Upload an image (jpg, png)", type=["jpg","jpeg","png"])
if uploaded is None:
    st.info("Upload an image to run detection.")
else:
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tmp_file.write(uploaded.getbuffer())
    tmp_file.flush()
    tmp_path = tmp_file.name

    frame = cv2.imread(tmp_path)
    if frame is None:
        st.error("Failed to read the uploaded image.")
    else:
        # Run vehicle detection
        vehicles = []
        if vehicle_model:
            try:
                result = vehicle_model(frame, conf=conf)
                vehicles = result[0].boxes.xyxy.tolist()
            except Exception as e:
                st.error("Vehicle detection error: " + str(e))
                vehicles = []

        # Run helmet detection
        helmets = []
        if helmet_model:
            try:
                result = helmet_model(frame, conf=conf)
                helmets = result[0].boxes.xyxy.tolist()
            except Exception as e:
                st.warning("Helmet detection failed: " + str(e))
                helmets = []

        # Plates detection (optional, leave empty for now)
        plates = []

        # Draw boxes
        out_img = draw_boxes(frame, vehicles=vehicles, helmets=helmets, plates=plates)
        st.image(out_img, use_column_width=True, caption="Processed Image")

        # Show results
        st.subheader("Results")
        st.write(f"Vehicles detected: {len(vehicles)}")
        st.write(f"Helmets detected: {len(helmets)}")
        st.write(f"Plates detected: {len(plates)} (OCR not implemented)")
