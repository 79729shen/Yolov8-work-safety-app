#這段代碼是一個基於Streamlit框架的Web應用程序，用於實現視頻中物體的檢測功能。代碼的主要功能如下：

#初始化類 VideoDetector：

#被設計來處理視頻中的物體檢測。
#它接收一個模型（model）和一個準確度（accuracy）作為參數，這些參數用於後續的視頻分析過程。
#上傳和處理視頻：

#使用Streamlit的側邊欄功能，提供一個用戶界面讓用戶上傳MP4格式的視頻文件。
#上傳的視頻被保存到指定路徑（這裡是videos目錄）。
#如果沒有上傳視頻，則使用一個預設的視頻（video_11.mp4）。
#顯示視頻：

#在Streamlit的主界面上顯示上傳的或預設的視頻。
#物體檢測功能：

#提供一個按鈕讓用戶啟動物體檢測過程。
#使用OpenCV庫讀取視頻文件，並逐幀進行處理。
#對於每一幀圖像，利用先前提供的模型和準確度進行物體檢測。
#可選擇是否使用物體跟蹤器（由is_display_tracker和tracker決定）。
#將檢測結果顯示在Streamlit界面上。
#檢測結果的處理和匯總：

#每次檢測物體後，將檢測到的物體類型添加到一個列表中。
#當視頻讀取完成或讀取失敗時，釋放視頻資源並刪除非預設的臨時視頻文件。
#使用輔助函數對檢測到的物體進行匯總並展示結果。
#總體來說，這段代碼為用戶提供了一個界面來上傳視頻，並利用機器學習模型來進行物體檢測，然後將結果顯示給用戶。



import PIL  # 導入PIL庫，用於圖像處理
import settings  # 導入設置模組，可能包含配置和常量定義
import streamlit as st  # 導入streamlit庫，用於構建web應用
import os  # 導入os模組，用於操作系統功能，如文件路徑操作
import cv2  # 導入OpenCV庫，用於視頻處理
from pathlib import Path  # 從pathlib模塊導入Path類，用於處理文件路徑
import shutil  # 導入shutil模組，用於執行高級文件操作
import helper  # 導入輔助功能模組，可能包含額外的功能或工具

class VideoDetector:  # 定義一個視頻檢測類
    def __init__(self, model, accuracy):  # 初始化方法，接受模型和準確度作為參數
        self.model = model  # 將傳入的模型賦值給實例變量
        self.accuracy = accuracy  # 將傳入的準確度賦值給實例變量

    def detect(self):  # 定義檢測方法
        video_path = None  # 初始化視頻路徑變量
        source_vid = None  # 初始化來源視頻變量
        with st.sidebar:  # 在Streamlit的側邊欄中
            source_vid = st.file_uploader("Upload a video", type=["mp4"])  # 創建一個文件上傳器，只接受MP4格式的視頻
        is_display_tracker, tracker = helper.display_tracker_options()  # 從輔助模組獲取跟蹤器選項
        try:  # 錯誤處理
            if source_vid is not None:  # 如果上傳了視頻
                video_path = os.path.join("videos", source_vid.name)  # 創建視頻文件的路徑
                with open(video_path, "wb") as video_file:  # 打開文件進行寫入
                    video_file.write(source_vid.read())  # 將上傳的視頻寫入文件
            else:  # 如果沒有上傳視頻
                video_path = os.path.join("videos", "video_7.mp4")  # 使用默認視頻路徑
            st.video(video_path)  # 在Streamlit中顯示視頻
        except Exception as ex:  # 處理加載視頻時可能發生的異常
            st.error(f"Error loading video")  # 顯示錯誤信息
            st.error(ex)  # 顯示異常詳細信息

        detected_objects_summary_list = []  # 初始化檢測到的對象列表

        if st.sidebar.button("Detect Objects"):  # 如果側邊欄的檢測按鈕被點擊
            vid_cap = cv2.VideoCapture(video_path)  # 使用OpenCV打開視頻
            st_frame = st.empty()  # 在Streamlit中創建一個空白的框架
            while vid_cap.isOpened():  # 當視頻開啟時
                success, image = vid_cap.read()  # 讀取視頻的一幀
                if success:  # 如果讀取成功
                    # 調用helper模組中的display_frames函數進行幀處理和顯示
                    res = helper.display_frames(
                        self.model,  # 使用初始化時指定的機器學習模型
                        self.accuracy,  # 使用初始化時設定的檢測準確度
                        st_frame,  # Streamlit的空白框架，用於顯示處理後的圖像
                        image,  # 從視頻中讀取的當前幀
                        is_display_tracker,  # 布爾值，決定是否顯示物體跟蹤器的結果
                        tracker,  # 物體跟蹤器的實例，如果is_display_tracker為True則使用
                    )
                    detected_objects_summary_list.extend(res[0].boxes.cls)  # 添加檢測到的對象類別到列表中
                else:  # 如果讀取失敗
                    vid_cap.release()  # 釋放視頻資源
                    if Path(video_path).name != "video_7.mp4":  # 如果不是默認視頻
                        os.remove(video_path)  # 刪除臨時保存的視頻文件
                    helper.sum_detections(detected_objects_summary_list, self.model)  # 使用輔助函數匯總檢測結果
                    break  # 跳出循環
