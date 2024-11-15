import PIL
import settings
import streamlit as st
import os
import cv2
from pathlib import Path
import helper

class VideoDetector:
    def __init__(self, model, accuracy):
        self.model = model
        self.accuracy = accuracy

    def detect(self):
        video_path = None
        source_vid = None
        with st.sidebar:
            source_vid = st.file_uploader("Upload a video", type=["mp4"])
        is_display_tracker, tracker = helper.display_tracker_options()
        try:
            if source_vid is not None:
                video_path = Path("videos") / source_vid.name
                with open(video_path, "wb") as video_file:
                    video_file.write(source_vid.read())
            else:
                video_path = Path(settings.DEFAULT_VIDEO)  

            st.video(str(video_path))  
        except Exception as ex:
            st.error(f"Error loading video")
            st.error(ex)

        detected_objects_summary_list = []

        if st.sidebar.button("Detect Objects"):
            vid_cap = cv2.VideoCapture(str(video_path))  
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    res = helper.display_frames(
                        self.model,
                        self.accuracy,
                        st_frame,
                        image,
                        is_display_tracker,
                        tracker,
                    )
                    detected_objects_summary_list.extend(res[0].boxes.cls)
                else:
                    vid_cap.release()
                    if video_path.name != Path(settings.DEFAULT_VIDEO).name:
                        os.remove(str(video_path))  
                    helper.sum_detections(detected_objects_summary_list, self.model)
                    break
