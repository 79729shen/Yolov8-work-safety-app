#這段code是為了創建一個基於Streamlit框架的Web應用程序，用於從攝像頭實時檢測物體。代碼的主要功能如下：

#初始化和管理全局狀態：

#使用st.session_state來存儲和管理全局狀態（在這裡是檢測到的物體列表）。這種方法確保了應用程序在多個請求之間可以保持狀態。
#定義 WebcamDetector 類：

#這個類用於處理攝像頭的物體檢測。
#它接收一個機器學習模型和一個準確度參數來初始化。
#包含一個detect方法，用於實施物體檢測的邏輯。
#打開和讀取攝像頭數據：

#使用OpenCV打開預設的攝像頭。
#在Streamlit的Web界面上創建一個空白框架用於顯示視頻。
#實時物體檢測：

#在一個循環中讀取攝像頭的每一幀圖像。
#對每一幀圖像進行物體檢測，並選擇是否使用物體跟蹤器。
#將檢測結果添加到st.session_state.detected_objects_summary_list中。

#用戶交互：
#提供兩個按鈕，一個用於打開攝像頭並開始檢測，另一個用於關閉攝像頭並退出檢測。
#使用self.quit_flag作為一個標記來控制是否退出攝像頭讀取循環。

#檢測結果的處理和匯總：

#當用戶選擇退出攝像頭時，使用輔助函數匯總並顯示檢測到的物體。
#清空全局列表以準備下一次檢測。
#總體來說，這段code提供了一個實用的界面，讓用戶可以通過Web應用來實時檢測攝像頭中的物體，並且可以隨時開始和終止檢測過程。這對於需要實時視頻監控或分析的應用程序非常有用。


import cv2  # 導入OpenCV庫，用於處理視頻和圖像
import streamlit as st  # 導入streamlit庫，用於構建Web應用
import helper  # 導入輔助功能模塊，可能包含額外的功能或工具

# 初始化一個全局變量以存儲狀態
if 'detected_objects_summary_list' not in st.session_state:
    st.session_state.detected_objects_summary_list = []  # 如果變量不存在於session_state中，則創建一個空列表

class WebcamDetector:  # 定義一個Webcam檢測類
    def __init__(self, model, accuracy):  # 初始化方法，接受模型和準確度作為參數
        self.model = model  # 將傳入的模型賦值給實例變量
        self.accuracy = accuracy  # 將傳入的準確度賦值給實例變量
        self.quit_flag = False  # 初始化一個標記，用於控制退出循環的條件

    def detect(self):  # 定義檢測方法
        is_display_tracker, tracker = helper.display_tracker_options()  # 從輔助模塊獲取跟蹤器選項
        if st.sidebar.button("Turn On Webcam"):  # 如果側邊欄中的“打開攝像頭”按鈕被點擊
            try:  # 錯誤處理
                vid_cap = cv2.VideoCapture(0)  # 使用OpenCV打開預設的攝像頭
                st_frame = st.empty()  # 在Streamlit中創建一個空白的框架
                while vid_cap.isOpened() and not self.quit_flag:  # 當攝像頭開啟且沒有退出標記時
                    success, image = vid_cap.read()  # 讀取攝像頭的一幀
                    if success:  # 如果從攝像頭成功讀取了一幀圖像
                        # 調用helper模塊中的display_frames函數進行幀的處理和顯示
                        res = helper.display_frames(
                            self.model,  # 使用此類別初始化時提供的機器學習模型
                            self.accuracy,  # 使用此類別初始化時設定的物體檢測準確度
                            st_frame,  # 使用Streamlit的空白框架，用於在Web應用界面上顯示處理後的圖像
                            image,  # 從攝像頭讀取到的當前幀圖像
                            is_display_tracker,  # 布爾值，用於決定是否顯示物體跟蹤器的結果
                            tracker,  # 物體跟蹤器的實例，如果is_display_tracker為True，則在物體檢測中使用這個跟蹤器
                        )
                        st.session_state.detected_objects_summary_list.extend((res[0].boxes.cls).tolist())  # 添加檢測到的對象類別到session_state列表中
                    else:  # 如果讀取失敗
                        vid_cap.release()  # 釋放攝像頭資源
            except Exception as e:  # 處理攝像頭加載時可能發生的異常
                st.sidebar.error("Error loading video: " + str(e))  # 在側邊欄顯示錯誤信息
        if st.sidebar.button('Quit Webcam'):  # 如果側邊欄中的“退出攝像頭”按鈕被點擊
            self.quit_flag = True  # 設置退出標記為True
            helper.sum_detections(st.session_state.detected_objects_summary_list, self.model)  # 使用輔助函數匯總檢測結果
            st.session_state.detected_objects_summary_list = []  # 清空session_state中的列表
