import PIL
import streamlit as st
from pathlib import Path
from ImageDetector import ImageDetector
from VideoDetector import VideoDetector
from YouTubeDetector import YouTubeDetector
from WebcamDetector import WebcamDetector
import settings
import helper
import os

# Set environment variable to avoid issues with duplicated libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set up the page layout
st.set_page_config(
    page_title="Object Detection And Tracking using YOLOv11",  # Set the page title
    page_icon="🚀",  # Set the page icon
    layout="wide",  # Set the layout to wide mode
    initial_sidebar_state="expanded",  # Expand the sidebar by default
)

# Main page heading
st.title("Object Detection And Tracking using YOLOv11")

# Sidebar - Model Configuration
st.sidebar.header("ML Model Config")
model_type = st.sidebar.radio("Select Task", ['BEST', 'TBM_SAFETY'])  # Choose the task type
confidence = float(st.sidebar.slider("Select Model Confidence", 25, 100, 40)) / 100  # Set the model confidence level

# Select the model path based on the task type
if model_type == "BEST":
    model_path = Path(settings.BEST_MODEL)
elif model_type == "TBM_SAFETY":
    model_path = Path(settings.TBM_SAFETY_MODEL)

# Load the pre-trained machine learning model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Error loading model. Check the specified path: {model_path}")
    st.error(ex)
    model = None  # Ensure model is set to None in case of failure

# Sidebar - Data Configuration
st.sidebar.header("Data Config")
source_radio = st.sidebar.radio("Select Source", ["Image", "Video", "Youtube"])

# Perform detection based on the selected data source
if model is not None:
    if source_radio == "Image":
        image_detector = ImageDetector(model, confidence)
        image_detector.detect()
    elif source_radio == "Youtube":
        youtube_detector = YouTubeDetector(model, confidence)
        youtube_detector.detect()
    elif source_radio == "Video":
        video_detector = VideoDetector(model, confidence)
        video_detector.detect()
    elif source_radio == "Webcam":
        webcam_detector = WebcamDetector(model, confidence)
        webcam_detector.detect()
    elif source_radio in [settings.RTSP]:
        helper.play_rtsp_stream(confidence, model)
else:
    st.error("Model could not be loaded. Please check the model path and configuration.")
