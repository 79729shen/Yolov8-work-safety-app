# YOLOv11 目標檢測與跟蹤

## 簡介
歡迎使用 YOLOv11 目標檢測與追蹤應用！該工具允許您在各種來源上執行目標檢測與追蹤，包括圖像、影片、YouTube 、RTSP 和Webcam。

## 入門指南

### 1. 模型配置
- 在側邊欄的 "ML Model Config" 下，您可以選擇兩個預配置的任務："BEST" 和 "TBM_SAFETY"。
- 使用滑塊調整模型置信度，設置目標檢測的置信閾值。

### 2. 數據配置
- 在側邊欄的 "Data Config" 下，從以下選項中選擇數據來源：
  - **Image：** 選擇此選項在靜態圖像上執行目標檢測。
  - **video：** 選擇此選項以分析影片文件中的對象。
  - **YouTube：** 選擇此選項以分析 YouTube 中的對象。

## 運行應用程序

### 1. 圖像
- 如果選擇 "Image" 作為數據來源，應用程序將使用選定的模型在圖像中檢測和跟蹤對象。

### 2. 影片
- 選擇 "Video" 選項以分析影片文件。應用程序將處理每一幀以檢測和跟蹤對象。

### 3. YouTube 
- 如果選擇 "YouTube"，請輸入 YouTube 影片的 URL。應用程序將獲取影片並執行目標檢測和跟蹤。

### 4. RTSP 
- 對於 RTSP 流，請確保在設置中配置正確的 RTSP URL。應用程序將播放流並實時應用目標檢測。

### 5. Webcam 
- 使用Webam請確保網路攝影機或相機鏡頭正常及線路正常。

## 注意
- 如果加載選定模型時出現問題，將顯示錯誤消息。請仔細檢查指定的模型路徑。

### Running Locally
#1. 安裝 Conda 環境
確保已經安裝 Conda。在命令行中，使用以下命令創建並激活 Conda 環境：
```bash
conda create --name your_environment_name python=3.8
conda activate your_environment_name
# run app.py
streamlit run app.py
```

## 盡情探索並使用 YOLOv11 進行檢測與追蹤！🚀

