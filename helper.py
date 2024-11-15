from ultralytics import YOLO
import streamlit as st
import cv2
from pytube import YouTube
import settings

def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    try:
        model = YOLO(model_path)
        print("Model loaded successfully!")  # Console output for debugging
        return model
    except Exception as e:
        st.error(f"Error loading model. Check the specified path: {model_path}")
        st.error(f"Error details: {str(e)}")  # Display detailed error message in Streamlit
        raise e  # Re-raise the exception to ensure it's caught in app.py

def display_tracker_options():
    """
    Displays options for enabling object tracking in the Streamlit app.

    Returns:
        Tuple (bool, str): A tuple containing a boolean flag for displaying the tracker and the selected tracker type.
    """
    display_tracker = st.radio("Display Tracker", ("Yes", "No"))
    is_display_tracker = True if display_tracker == "Yes" else False
    if display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None

def display_frames(model, acc, st_frame, image, is_display_tracker=None, tracker_type=None):
    """
    Displays detected objects from a video stream.

    Parameters:
        model (YOLO): A YOLO object detection model.
        acc (float): The model's confidence threshold.
        st_frame (streamlit.Streamlit): A Streamlit frame object.
        image (PIL.Image.Image): A frame from a video stream.
        is_display_tracker (bool): Whether or not to display a tracker.
        tracker_type (str): The type of tracker to display.

    Returns:
        None
    """
    image = cv2.resize(image, (720, int(720 * (9 / 16))))
    if is_display_tracker:
        res = model.track(image, conf=acc, persist=True, tracker=tracker_type)
    else:
        res = model.predict(image, conf=acc)

    res_plot = res[0].plot()
    st_frame.image(
        res_plot,
        caption="Detected Video",
        channels="BGR",
        use_column_width=True,
    )
    return res

def sum_detections(detected_objects_summary_list, model):
    """
    Summarizes detected objects from a list and displays the summary in a Streamlit success message.

    Parameters:
        detected_objects_summary_list (list): List of detected object indices.

    Returns:
        None
    """
    detected_objects_summary = set()
    for obj in detected_objects_summary_list:
        detected_objects_summary.add(model.names[int(obj)])
    name_summary = ", ".join(detected_objects_summary)
    st.success(f"Detected Objects: {name_summary}")

def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """
    image = cv2.resize(image, (720, int(720*(9/16))))
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        res = model.predict(image, conf=conf)
    res_plotted = res[0].plot()
    st_frame.image(res_plotted, caption='Detected Video', channels="BGR", use_column_width=True)

def play_stored_video(conf, model):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None
    """
    source_vid = st.sidebar.selectbox("Choose a video...", settings.VIDEOS_DICT.keys())
    is_display_tracker, tracker = display_tracker_options()
    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)
    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf, model, st_frame, image, is_display_tracker, tracker)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

def play_rtsp_stream(conf, model):
    """
    Plays an RTSP stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None
    """
    source_rtsp = st.sidebar.text_input("RTSP stream url:")
    st.sidebar.caption('Example URL: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf, model, st_frame, image, is_display_tracker, tracker)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            vid_cap.release()
            st.sidebar.error("Error loading RTSP stream: " + str(e))
