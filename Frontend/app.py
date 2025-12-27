import sys
import os
import tempfile
import cv2
import numpy as np
import streamlit as st
from PIL import Image

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from Backend.detector  import detect_vehicles

st.set_page_config(
    page_title="Vehicle Detection & Counting",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0f172a;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🚗 Vehicle Detection & Counting System")
st.markdown(
    "Upload an **image or video** to detect vehicles and see their counts."
)

st.sidebar.header("⚙️ Settings")

input_type = st.sidebar.radio(
    "Select Input Type",
    ["Image", "Video"]
)

show_boxes = st.sidebar.checkbox(
    "Show Bounding Boxes",
    value=True
)

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05
)



