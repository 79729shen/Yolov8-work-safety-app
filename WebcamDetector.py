import cv2
import streamlit as st
import helper

# Initialize a global variable to store the state
if 'detected_objects_summary_list' not in st.session_state:
    st.session_state.detected_objects_summary_list = []

class WebcamDetector:
    def __init__(self, model, accuracy):
        self.model = model
        self.accuracy = accuracy
        self.quit_flag = False 
        
    def detect(self):
        is_display_tracker, tracker = helper.display_tracker_options()
        if st.sidebar.button("Turn On Webcam"):
            try:
                vid_cap = cv2.VideoCapture(0)
                st_frame = st.empty()
                while vid_cap.isOpened() and not self.quit_flag:
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
                        st.session_state.detected_objects_summary_list.extend((res[0].boxes.cls).tolist())
                    else:
                        vid_cap.release()
            except Exception as e:
                st.sidebar.error("Error loading video: " + str(e))
        if st.sidebar.button('Quit Webcam'):
            self.quit_flag = True
            helper.sum_detections(st.session_state.detected_objects_summary_list, self.model)
            st.session_state.detected_objects_summary_list = []
