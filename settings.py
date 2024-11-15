from pathlib import Path
import sys

# Get the absolute path of the current script file
file_path = Path(__file__).resolve()

# Get the parent directory of the script file
root_path = file_path.parent

# Add the parent directory to the system path if not already present
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Create a relative path from the current working directory to the parent directory
ROOT = root_path.relative_to(Path.cwd())

# Source
IMAGE = 'Image'
VIDEO = 'Video'
YOUTUBE = 'Youtube'
WEBCAM  = 'Webcam'
RTSP = 'RTSP'
SOURCE_LIST = [IMAGE, VIDEO, YOUTUBE, WEBCAM]

# Images configuration
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'default.jpg'

# Videos configuration
VIDEO_DIR = ROOT / 'videos'
DEFAULT_VIDEO = VIDEO_DIR / 'video_8.mp4'

# Youtube configuration
DEFAULT_URL = "https://www.youtube.com/watch?v=41ID7HECvJI"

# ML Model configuration
MODEL_DIR = ROOT / 'weights'
BEST_MODEL = MODEL_DIR / 'BEST.pt'
TBM_SAFETY_MODEL = MODEL_DIR / 'TBMSafety.pt'
