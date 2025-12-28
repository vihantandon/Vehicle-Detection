import sys
import os
import tempfile
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# sys.path.append(
#     os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# )
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
from Backend.detector import detect_vehicles

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
    .upload-section {
        padding: 2rem;
        border: 2px dashed #4a5568;
        border-radius: 10px;
        text-align: center;
        margin: 2rem 0;
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

# Main content area
if input_type == "Image":
    st.subheader("Upload an image")
    uploaded_file = st.file_uploader(
        "Drag and drop file here",
        type=["jpg", "jpeg", "png"],
        key="image_uploader",
        help="Limit 200MB per file • JPG, JPEG, PNG"
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        
        with st.spinner("Detecting Vehicles..."):
            annotated_image, counts = detect_vehicles(
                image_np,
                conf_threshold=confidence,
                show_boxes=show_boxes
            )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(
                annotated_image,
                caption="Detected Vehicles",
                width=700
            )
        
        with col2:
            st.subheader("Vehicle Count")
            total = 0
            for k, v in counts.items():
                st.metric(k.capitalize(), v)
                total += v
            st.markdown(f"### Total Vehicles: **{total}**")

elif input_type == "Video":
    st.subheader("Upload a video")
    uploaded_file = st.file_uploader(
        "Drag and drop file here",
        type=["mp4", "avi", "mov"],
        key="video_uploader",
        help="Limit 200MB per file • MP4, AVI, MOV"
    )
    
    if uploaded_file:
        # Save uploaded video to temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        tfile.close()
        
        cap = cv2.VideoCapture(video_path)
        
        st.subheader("Video Preview (Live Detection)")
        
        # Create placeholders
        stframe = st.empty()
        stats_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        # Get total frames for progress tracking
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame
            annotated_frame, counts = detect_vehicles(
                frame,
                conf_threshold=confidence,
                show_boxes=show_boxes
            )
            
            # Display frame
            stframe.image(
                annotated_frame,
                channels="BGR",
                width=700
            )
            
            # Update stats
            with stats_placeholder.container():
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("🚗 Cars", counts.get("car", 0))
                col2.metric("🏍️ Bikes", counts.get("motorcycle", 0))
                col3.metric("🚌 Buses", counts.get("bus", 0))
                col4.metric("🚚 Trucks", counts.get("truck", 0))
            
            # Update progress
            progress_bar.progress(frame_count / total_frames)
        
        cap.release()
        progress_bar.empty()
        
        # Clean up temporary file
        try:
            os.unlink(video_path)
        except:
            pass
        
        st.success("✅ Video processing completed.")

st.markdown("---")
st.markdown(
    "<center>Built using Streamlit + YOLO</center>",
    unsafe_allow_html=True
)