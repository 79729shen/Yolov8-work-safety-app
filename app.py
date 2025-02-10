#代碼的主要功能是在 Streamlit 應用中實現基於 YOLOv8 模型的物件偵測和追蹤，並支持多種數據源（圖像、視頻、YouTube、RTSP流、網絡攝像頭）。
#這段代碼包括以下幾個主要部分：

#導入所需的模組和庫：

#包括圖像處理的 PIL 庫，Streamlit 用於建立 Web 應用，Pathlib 用於處理系統路徑，以及自定義的物件偵測和追蹤模組。
#設置 Streamlit 頁面：

#配置頁面的標題、圖標、佈局以及側邊欄的初始狀態。
#模型配置選項：

#在側邊欄中添加模型配置，允許用戶選擇不同的模型任務（例如“BEST”或“TBM_SAFETY”）和設置模型的置信度閾值。
#加載機器學習模型：

#根據選擇的模型類型，加載相應的 YOLOv8 預訓練模型。
#選擇數據源：

#用戶可在側邊欄中選擇數據源，包括圖像、視頻、YouTube、RTSP 流和網絡攝像頭。
#執行物件偵測：

#根據選擇的數據源，代碼會創建對應的檢測器（例如 ImageDetector、VideoDetector 等）並執行物件偵測功能。
#處理 RTSP：

#如果選擇了 RTSP 數據源，則調用 helper.play_rtsp_stream 函數來處理和展示 RTSP 流的物件偵測結果。
#總之，這段代碼透過整合 YOLOv8 模型和 Streamlit Web 應用，提供了一個交互式平台，用於從不同的數據源進行實時物件偵測和追蹤。



import PIL  # 導入PIL庫，用於圖像處理
import streamlit as st  # 導入streamlit庫，用於建立Web應用
from pathlib import Path  # 從pathlib導入Path類，用於處理系統路徑
from ImageDetector import ImageDetector  # 從自定義模組導入ImageDetector
from VideoDetector import VideoDetector  # 從自定義模組導入VideoDetector
from YouTubeDetector import YouTubeDetector  # 從自定義模組導入YouTubeDetector
from WebcamDetector import WebcamDetector  # 從自定義模組導入WebcamDetector
import settings  # 導入settings模組，通常包含配置信息
import helper  # 導入helper模組，通常包含輔助功能
import os  # 導入os模組，用於與操作系統進行交握
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 設置環境變量以避免庫衝突

# 設置頁面布局
st.set_page_config(
    page_title="Object Detection And Tracking using YOLOv8",  # 設置頁面標題
    page_icon="🚀",  # 設置頁面圖標
    layout="wide",  # 設置頁面佈局為寬屏模式
    initial_sidebar_state="expanded",  # 初始側邊欄狀態設置為展開
)

# 頁面主標題
st.title("Object Detection And Tracking using YOLOv8")  # 顯示主標題

# 側邊欄
st.sidebar.header("ML Model Config")  # 側邊欄添加標題
model_type = st.sidebar.radio("Select Task", ['BEST', 'TBM_SAFETY'])  # 側邊欄單選按鈕選擇模型類型
confidence = float(st.sidebar.slider("Select Model Confidence", 25, 100, 40)) / 100  # 側邊欄滑塊選擇模型置信度閾值

# 選擇檢測或分割模式
if model_type == "BEST":
    model_path = Path(settings.BEST_MODEL)  # 設置BEST模型的路徑
elif model_type == "TBM_SAFETY":
    model_path = Path(settings.TMB_SAFETY_MODEL)  # 設置TBM_SAFETY模型的路徑

# 加載預訓練的機器學習模型
try:
    model = helper.load_model(model_path)  # 加載模型
except Exception as ex:
    st.error(f"Error loading model. Check the specified path: {model_path}")  # 加載失敗時顯示錯誤信息
    st.error(ex)

# 側邊欄
st.sidebar.header("Data Config")  # 側邊欄添加數據配置標題
source_radio = st.sidebar.radio("Select Source", ["Image", "Video", "Youtube", "RTSP", "Webcam"])  # 側邊欄單選按鈕選擇數據源

# 根據選擇的數據源執行不同的檢測功能
if source_radio == "Image":
    image_detector = ImageDetector(model, confidence)  # 創建圖像檢測器
    image_detector.detect()  # 執行圖像檢測
elif source_radio == "Youtube":
    youtube_detector = YouTubeDetector(model, confidence)  # 創建YouTube檢測器
    youtube_detector.detect()  # 執行YouTube檢測
elif source_radio == "Video":
    video_detector = VideoDetector(model, confidence)  # 創建視頻檢測器
    video_detector.detect()  # 執行視頻檢測
elif source_radio == "Webcam":
    webcam_detector = WebcamDetector(model, confidence)  # 創建網絡攝像頭檢測器
    webcam_detector.detect()  # 執行網絡攝像頭檢測
elif source_radio in [settings.RTSP]:
    helper.play_rtsp_stream(confidence, model)  # 使用RTSP進行檢測

