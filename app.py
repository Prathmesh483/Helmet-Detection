# app.py
import streamlit as st
import cv2
import numpy as np
import tempfile
import os

from detection import HelmetDetector, VehicleDetector
from ocr import PlateReader
from analysis import TrafficCounter, EmissionCalculator
from utils import draw_boxes


# from detection.helmet_detector import HelmetDetector
# from detection.vehicle_detector import VehicleDetector
# from ocr.plate_reader import PlateReader
# from analysis.traffic_counter import TrafficCounter
# from analysis.emission_calculator import EmissionCalculator
# from utils.draw import draw_boxes

st.set_page_config(page_title="Traffic AI System", layout="wide")
st.title("🚦 Traffic AI Analysis System (College Demo)")

# Lazy instantiate detectors inside try/except to avoid app crash on import
@st.cache_resource
def get_models():
    try:
        helmet = HelmetDetector()
    except Exception as e:
        st.error("HelmetDetector initialization error: " + str(e))
        helmet = None
    try:
        vehicle = VehicleDetector()
    except Exception as e:
        st.error("VehicleDetector initialization error: " + str(e))
        vehicle = None
    try:
        plate_reader = PlateReader()
    except Exception as e:
        st.error("PlateReader initialization error: " + str(e))
        plate_reader = None

    emission_calc = EmissionCalculator()
    traffic_counter = TrafficCounter()
    return helmet, vehicle, plate_reader, emission_calc, traffic_counter

helmet_model, vehicle_model, plate_reader, emission_calc, traffic_counter = get_models()

st.sidebar.header("Settings")
conf = st.sidebar.slider("Detection confidence", 0.1, 0.9, 0.25)
st.sidebar.markdown("---")
st.sidebar.write("If you see model load errors, run `python download_models.py` locally or add yolov8n.pt in models/")

uploaded = st.file_uploader("Upload an image (jpg, png)", type=["jpg","jpeg","png"])
if uploaded is None:
    st.info("Upload an image to run detection. You can also run download_models.py to pre-download weights.")
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
                vehicles = vehicle_model.detect(frame, conf=conf)
            except Exception as e:
                st.error("Vehicle detection error: " + str(e))
                vehicles = []

        # Run helmet detection (safe)
        helmets = []
        if helmet_model:
            try:
                helmets = helmet_model.detect(frame, conf=conf)
            except Exception as e:
                st.warning("Helmet detection failed: " + str(e))
                helmets = []

        # OCR plates
        plates = []
        if plate_reader:
            try:
                plates = plate_reader.read_plates(frame)
            except Exception as e:
                st.warning("Plate OCR failed: " + str(e))
                plates = []

        # Counting
        breakdown = traffic_counter.count(vehicles)
        # Estimate emissions using breakdown
        emission = emission_calc.calculate(breakdown, idle_seconds=60)

        out_img = draw_boxes(frame, vehicles=vehicles, helmets=helmets, plates=plates)
        st.image(out_img, use_column_width=True, caption="Processed image")

        st.subheader("Results")
        st.write("Vehicle counts:", breakdown)
        st.write("Estimated CO₂ for 1 minute idle (g):", emission["total_gCO2"])
        st.json({"emission_breakdown": emission["breakdown"], "plates_detected": plates})

# helpful footer
st.markdown("---")
st.markdown("**Note:** If the app crashes while loading models (e.g. `UnpicklingError`), it's usually because the weight file under `models/` is corrupted or incomplete on the host. Run `python download_models.py` locally and push the models/ folder or ensure the host allows outgoing downloads.")
