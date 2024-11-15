import cv2
import streamlit as st
import yt_dlp as youtube_dl
import settings
import helper
import re
from pathlib import Path

class YouTubeDetector:
    def __init__(self, model, accuracy):
        self.model = model
        self.accuracy = accuracy
        self.download_path = Path("C:/Users/lab612/Desktop/workman/Object-Detection-WebApp-main/videos")
        self.download_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    def clean_youtube_url(self, url):
        """Remove any time parameter (&t=...) from the YouTube URL."""
        return re.sub(r'&t=\d+s', '', url)

    def download_video(self, url):
        ydl_opts = {
            'format': 'best',  # Use the best compatible format
            'quiet': True,
            'outtmpl': str(self.download_path / '%(id)s.%(ext)s'),  # Set path to save the downloaded video
            'ignoreerrors': True,
            'retries': 3  # Retry if there are temporary issues
        }
        try:
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=True)  # Download video
                video_path = Path(ydl.prepare_filename(info_dict))  # Get path of the downloaded file
                if not video_path.exists():
                    raise ValueError("Failed to download video.")
            return video_path
        except Exception as e:
            st.sidebar.error(f"Error downloading video: {str(e)}")
            return None

    def detect(self):
        # Clean the URL to remove any additional parameters like &t=274s
        source_youtube = self.clean_youtube_url(st.sidebar.text_input("YouTube Video URL", settings.DEFAULT_URL))
        is_display_tracker, tracker = helper.display_tracker_options()
        detected_objects_summary_list = []

        if st.sidebar.button("Detect Objects"):
            try:
                # Download the video locally
                video_path = self.download_video(source_youtube)
                if not video_path:
                    return  # Exit if the video download fails

                # Load the downloaded video file using cv2.VideoCapture
                vid_cap = cv2.VideoCapture(str(video_path))
                if not vid_cap.isOpened():
                    st.sidebar.error("Unable to open the downloaded video. Please check if the video format is supported.")
                    return

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
                        helper.sum_detections(detected_objects_summary_list, self.model)
                        break
            except Exception as e:
                st.sidebar.error("Error processing video: " + str(e))
