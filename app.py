# app.py
import streamlit as st
import cv2
import numpy as np
import tempfile

# Ultralytics YOLO & PyTorch
import torch
import ultralytics.nn.tasks
from ultralytics import YOLO

# Utils
from utils.draw import draw_boxes

# Streamlit page setup
st.set_page_config(page_title="Traffic AI System", layout="wide")
st.title("🚦 Traffic AI Analysis System (College Demo)")

# =====================
# Model initialization
# =====================
@st.cache_resource
def get_models():
    helmet_model = None
    vehicle_model = None

    try:
        with torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel]):
            helmet_model = YOLO("yolov8n.pt")  # cloud download
    except Exception as e:
        st.error("HelmetDetector initialization error: " + str(e))
        helmet_model = None

    try:
        with torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel]):
            vehicle_model = YOLO("yolov8n.pt")  # cloud download
    except Exception as e:
        st.error("VehicleDetector initialization error: " + str(e))
        vehicle_model = None

    # PlateReader, EmissionCalculator, TrafficCounter not used in this version
    return helmet_model, vehicle_model, None, None, None


# Load models
helmet_model, vehicle_model, _, _, _ = get_models()

# =====================
# Sidebar settings
# =====================
st.sidebar.header("Settings")
conf = st.sidebar.slider("Detection confidence", 0.1, 0.9, 0.25)

# =====================
# Image upload
# =====================
uploaded = st.file_uploader("Upload an image (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload an image to run detection.")
else:
    # Save uploaded image to temp file
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tmp_file.write(uploaded.getbuffer())
    tmp_file.flush()
    tmp_path = tmp_file.name

    # Read image with OpenCV
    frame = cv2.imread(tmp_path)
    if frame is None:
        st.error("Failed to read the uploaded image.")
    else:
        # =====================
        # Vehicle detection
        # =====================
        vehicles = []
        if vehicle_model:
            try:
                result = vehicle_model(frame, conf=conf)
                vehicles = result[0].boxes.xyxy.tolist()  # list of bounding boxes
            except Exception as e:
                st.error("Vehicle detection error: " + str(e))
                vehicles = []

        # =====================
        # Helmet detection
        # =====================
        helmets = []
        if helmet_model:
            try:
                result = helmet_model(frame, conf=conf)
                helmets = result[0].boxes.xyxy.tolist()
            except Exception as e:
                st.warning("Helmet detection failed: " + str(e))
                helmets = []

        # =====================
        # Draw bounding boxes
        # =====================
        out_img = draw_boxes(frame, vehicles=vehicles, helmets=helmets, plates=[])
        st.image(out_img, use_column_width=True, caption="Processed image")
