#這段代碼的主要功能是設定和組織文件系統路徑以及配置機器學習模型路徑，
#用於一個可能與影像處理、物體檢測或視頻分析相關的Python應用程序。具體功能包括：

#設定文件系統路徑：

#使用pathlib庫的Path類來獲取和處理文件和目錄的路徑。
#獲取當前腳本文件（即代碼所在的Python文件）的絕對路徑。
#獲取該腳本文件的父目錄，並將其添加到Python的系統路徑中，這樣可以使得這個目錄下的其他模組或文件可以被輕鬆導入和使用。
#從當前工作目錄（CWD）創建到父目錄的相對路徑。

#定義媒體源和模型配置常量：

#定義了多種媒體源類型（如圖像、視頻、YouTube視頻、網絡攝像頭、RTSP）的常量，這些可能用於媒體文件的選擇或處理。
#設定了存放圖像和視頻文件的目錄路徑，以及默認圖像和視頻文件的路徑。
#為YouTube視頻設定了一個默認的URL。
#為機器學習模型設定了存放位置的目錄路徑，並定義了幾個模型的路徑，包括一個被稱為"Best.pt"的模型和一個"TMB安全"模型。
#總結來說，這段代碼是為了支援一個較大的應用程序，這個應用程序可能涉及到從不同來源讀取媒體文件，並使用機器學習模型進行影像分析或物體檢測。
#通過這種方式組織代碼，開發者可以更方便地管理和訪問文件和模型，從而提高開發效率和代碼的可維護性。


from pathlib import Path  # 從pathlib模塊導入Path類，用於處理文件路徑
import sys  # 導入sys模組，用於訪問與Python解釋器緊密相關的變量和函數

# 獲取當前腳本文件的絕對路徑
file_path = Path(__file__).resolve()

# 獲取腳本文件的父目錄
root_path = file_path.parent

# 如果父目錄尚未在系統路徑中，則添加至系統路徑
if root_path not in sys.path:
    sys.path.append(str(root_path))

# 從當前工作目錄創建到父目錄的相對路徑
ROOT = root_path.relative_to(Path.cwd())

# 定義來源
IMAGE = 'Image'  # 定義圖像來源
VIDEO = 'Video'  # 定義視頻來源
YOUTUBE = 'Youtube'  # 定義YouTube視頻來源
WEBCAM  = 'Webcam'  # 定義網絡攝像頭來源
RTSP = 'RTSP'  # 定義RTSP流媒體來源
SOURCE_LIST = [IMAGE, VIDEO, YOUTUBE, WEBCAM]  # 將所有來源類型保存在列表中

# 圖像配置
IMAGES_DIR = ROOT / 'images'  # 定義存放圖像的目錄路徑
DEFAULT_IMAGE = IMAGES_DIR / 'Detection.jpg'  # 定義默認圖像文件的路徑

# 視頻配置
VIDEO_DIR = ROOT / 'videos'  # 定義存放視頻的目錄路徑
DEFAULT_VIDEO = VIDEO_DIR / 'video_7.mp4'  # 定義默認視頻文件的路徑

# Youtube配置
DEFAULT_URL = "https://www.youtube.com/watch?v=41ID7HECvJI&t=7s"  # 定義默認YouTube視頻的URL


# 機器學習模型配置
MODEL_DIR = ROOT / 'weights'  # 定義存放模型權重的目錄路徑
#DETECTION_MODEL = MODEL_DIR / 'yolov8n.pt'  # 定義物體檢測模型的路徑（目前被註釋掉）
#SEGMENTATION_MODEL = MODEL_DIR / 'yolov8n-seg.pt'  # 定義分割模型的路徑（目前被註釋掉）
BEST_MODEL = MODEL_DIR / 'Best.pt'  # 定義最佳模型的路徑
TMB_SAFETY_MODEL = MODEL_DIR / 'TBMSafety.pt'  # 定義TMB安全模型的路徑
