import streamlit as st
from detection.helmet_detector import HelmetDetector
from detection.vehicle_detector import VehicleDetector
from ocr.plate_reader import PlateReader
from analysis.emission_calculator import EmissionCalculator
from analysis.traffic_counter import TrafficCounter
from utils.draw import draw_boxes

import cv2
import numpy as np
from PIL import Image
import tempfile

st.set_page_config(page_title="Traffic AI System", layout="wide")

helmet_model = HelmetDetector()
vehicle_model = VehicleDetector()
plate_reader = PlateReader()
emission_calc = EmissionCalculator()
traffic_counter = TrafficCounter()

st.title("🚦 Traffic AI Analysis System")

uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded:
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(uploaded.read())

    img = cv2.imread(temp.name)

    # VEHICLE DETECTION
    vehicles = vehicle_model.detect(img)

    # HELMET DETECTION
    helmet_results = helmet_model.detect(img)

    # OCR
    plates = plate_reader.read_plates(img)

    # TRAFFIC COUNT ANALYSIS
    count = traffic_counter.count(vehicles)

    # EMISSION
    emission = emission_calc.calculate(count)

    # DRAW
    output_img = draw_boxes(img, vehicles, helmet_results, plates)

    st.image(output_img, caption="Processed Image")

    st.write(f"**Vehicles:** {count}")
    st.write(f"**Estimated CO₂ (g/min):** {emission}")
    st.write("**Detected License Plates:**")
    st.json(plates)
