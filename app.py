# app.py
import streamlit as st
import cv2
import numpy as np
import tempfile

from ultralytics import YOLO

# utils
from utils import draw_boxes

st.set_page_config(page_title="Traffic AI System", layout="wide")
st.title("🚦 Traffic AI Analysis System (College Demo)")

# Lazy instantiate detectors inside try/except to avoid app crash
@st.cache_resource
def get_models():
    helmet = None
    vehicle = None

    # Use cloud-pretrained YOLOv8 models
    try:
        helmet = YOLO("yolov8n.pt")  # generic detection; replace with helmet-specific if trained and hosted
    except Exception as e:
        st.error("HelmetDetector initialization error: " + str(e))
        helmet = None

    try:
        vehicle = YOLO("yolov8n.pt")  # generic object detection
    except Exception as e:
        st.error("VehicleDetector initialization error: " + str(e))
        vehicle = None

    # Optional: placeholder for PlateReader, TrafficCounter, EmissionCalculator
    plate_reader = None
    traffic_counter = None
    emission_calc = None

    return helmet, vehicle, plate_reader, emission_calc, traffic_counter

helmet_model, vehicle_model, plate_reader, emission_calc, traffic_counter = get_models()

st.sidebar.header("Settings")
conf = st.sidebar.slider("Detection confidence", 0.1, 0.9, 0.25)

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
        # Run vehicle detection (safe if vehicle_model is None)
        vehicles = []
        if vehicle_model:
            try:
                result = vehicle_model(frame, conf=conf)
                vehicles = result[0].boxes.xyxy.tolist()  # list of bounding boxes
            except Exception as e:
                st.error("Vehicle detection error: " + str(e))
                vehicles = []

        # Run helmet detection (safe)
        helmets = []
        if helmet_model:
            try:
                result = helmet_model(frame, conf=conf)
                helmets = result[0].boxes.xyxy.tolist()
            except Exception as e:
                st.warning("Helmet detection failed: " + str(e))
                helmets = []

        # Draw bounding boxes
        out_img = draw_boxes(frame, vehicles=vehicles, helmets=helmets, plates=[])
        st.image(out_img, use_column_width=True, caption="Processed image")
