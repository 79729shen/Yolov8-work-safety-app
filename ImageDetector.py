#Code的功能是在 Streamlit 應用中實現圖像的物件偵測。用戶可以上傳圖像，系統會使用預先設定的模型來偵測圖像中的物件，並將偵測結果顯示在用戶界面上。
#具體功能如下：

#上傳圖像：用戶可以通過 Streamlit 的側邊欄上傳圖像，支持多種圖像格式（jpg、jpeg、png、bmp、webp）。

#顯示上傳的圖像：上傳的圖像會在 Streamlit 應用的一個列中顯示。如果沒有上傳圖像，則會顯示一個默認的圖像。

#物件偵測：當用戶點擊“Detect Objects”按鈕後，系統會使用指定的模型對上傳或默認圖像進行物件偵測。

#顯示偵測結果：偵測到的物件會在 Streamlit 應用的另一個列中以繪製了邊界框的圖像形式顯示。此外，還會有一個區域顯示具體的偵測結果，如物件的位置和大小。

#錯誤處理：如果在圖像加載或偵測過程中發生錯誤，會在應用中顯示錯誤信息。

#這段code展示了 Streamlit 在機器學習和數據科學應用中的互動性和可視化能力，使用者可以直觀地上傳圖像，並即時看到模型的偵測結果。


import PIL  # 導入PIL庫，用於處理圖片
import settings  # 導入設置模組，通常包含配置信息
import streamlit as st  # 導入streamlit庫，用於構建web應用
import helper  # 導入輔助功能模組，可能包含額外的功能或工具

class ImageDetector:  # 定義一個圖像檢測類
    def __init__(self, model, accuracy):  # 初始化方法，接受一個模型和準確度作為參數
        self.model = model  # 將傳入的模型賦值給實例變量
        self.accuracy = accuracy  # 將傳入的準確度賦值給實例變量

    def detect(self):  # 定義檢測方法
        image_process = None  # 初始化一個變量，用於後續處理圖片
        source_image = st.sidebar.file_uploader(
            "Upload an image", type=("jpg", "jpeg", "png", "bmp", "webp")
        )  # 在側邊欄創建一個文件上傳器，接受圖片文件
        col1, col2 = st.columns(2)  # 創建兩列佈局
        with col1:  # 第一列的內容
            try:  # 錯誤處理
                if source_image is not None:  # 如果上傳了圖片
                    image_process = PIL.Image.open(source_image)  # 打開並處理上傳的圖片
                    st.image(image_process, caption="Uploaded Image", use_column_width=True)  # 顯示上傳的圖片
                else:  # 如果沒有上傳圖片
                    default_image = PIL.Image.open(settings.DEFAULT_IMAGE)  # 打開默認圖片
                    st.image(default_image, caption="Default Image", use_column_width=True)  # 顯示默認圖片
                    image_process = default_image  # 將默認圖片用於後續處理
            except Exception as ex:  # 處理打開圖片時可能發生的異常
                st.error(f"Error loading image")  # 顯示錯誤信息
                st.error(ex)  # 顯示異常詳細信息
        if st.sidebar.button("Detect Objects"):  # 如果側邊欄的檢測按鈕被點擊
            detected_objects_summary_list = []  # 初始化一個列表，用於存儲檢測結果
            res = self.model.predict(image_process, conf=self.accuracy)  # 使用模型對圖片進行預測
            boxes = res[0].boxes  # 獲取預測結果中的邊界框
            res_plotted = res[0].plot()[:,:,::-1]  # 獲取繪製了邊界框的圖片
            detected_objects_summary_list.extend(res[0].boxes.cls)  # 將檢測到的對象類別添加到列表中
            with col2:  # 第二列的內容
                st.image(res_plotted, caption='Detected Image', use_column_width=True)  # 顯示檢測後的圖片
                try:  # 錯誤處理
                    with st.expander("Detection Results"):  # 創建一個展開器，顯示檢測結果
                        if not boxes:  # 如果沒有檢測到對象
                            st.write("No objects detected")  # 顯示沒有檢測到對象的信息
                        else:  # 如果檢測到了對象
                            for box in boxes:  # 遍歷每個邊界框
                                st.write(box.xywh)  # 顯示邊界框的位置和大小
                except Exception as ex:  # 處理展示結果時可能發生的異常
                    st.write("An error occurred while processing the detection results")  # 顯示錯誤信息
            if boxes:  # 如果有檢測到對象
                helper.sum_detections(detected_objects_summary_list, self.model)  # 使用輔助函數處理並匯總檢測結果
