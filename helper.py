#這是一段使用ultralytics的YOLO模型進行物件偵測和追蹤的Python代碼，結合了streamlit進行視覺化呈現。
#code的主要功能是在 Streamlit 應用中實現 RTSP（Real Time Streaming Protocol）視頻流的實時物件偵測和追蹤。
#這個功能是通過整合 ultralytics 的 YOLO 模型來完成的。以下是代碼的主要功能細節：

#RTSP 的 URL 輸入：

#使用 Streamlit 的側邊欄來讓用戶輸入 RTSP 的 URL。這個 URL 是 RTSP 視频的地址。
#選擇是否顯示追蹤器：

#讓用戶選擇是否啟用物件追蹤功能。如果選擇啟用，用戶還可以選擇特定的追蹤器類型。
#開始物件偵測：

#當用戶點擊“Detect Objects”按鈕時，代碼會嘗試打開用戶提供的 RTSP，並進行物件偵測和追蹤。
#處理 RTSP 視頻：

#代碼使用 OpenCV 的 cv2.VideoCapture 方法來捕獲 RTSP 流。然後，它會進入一個循環，從流中讀取視頻幀，並使用 YOLO 模型進行物件偵測。
#顯示偵測結果：

#對於每個成功讀取的視頻幀，代碼使用 YOLO 模型來偵測幀中的物件。如果啟用了追蹤器，則會顯示物件追蹤的結果。偵測結果會在 Streamlit 應用中以圖形的形式顯示。
#錯誤處理：

#如果在處理 RTSP 的過程中發生錯誤（如無法加載或讀取幀失敗），代碼會捕獲異常，釋放視頻捕獲對象，並在 Streamlit 的側邊欄中顯示錯誤訊息。
#這段代碼結合了 ultralytics 的 YOLO 物件偵測模型和 Streamlit 的視覺化功能，提供了一個用戶友好的界面來實時偵測和追蹤 RTSP 視頻流中的物件。




#以下是對代碼中每一行的繁體中文詳解：
from ultralytics import YOLO  # 從 ultralytics 庫導入 YOLO 模組
import time  # 導入 time 模組，用於處理時間相關的功能
import streamlit as st  # 導入 streamlit 模組並命名為 st，用於建立 Web 應用
import cv2  # 導入 OpenCV 模組，用於處理影像
from pytube import YouTube  # 從 pytube 庫導入 YouTube 模組，用於下載 YouTube 影片
import settings  # 導入 settings 模組，通常包含配置或常數


def load_model(model_path): 
    """
  
    Loads a YOLO object detection model from the specified model_path.
    #>從指定的路徑加載 YOLO 物件偵測模型。

    Parameters:
        model_path (str): The path to the YOLO model file.
    #>參數:
        model_path (str): YOLO 模型文件的路徑。

    Returns:
        A YOLO object detection model.
    #>返回:
        一個 YOLO 物件偵測模型。
    """
    model = YOLO(model_path)  # 使用 YOLO 構造函數創建模型實例
    return model  # 返回模型實例



def display_tracker_options():
    """
    Displays options for enabling object tracking in the Streamlit app.
    #在 Streamlit 應用中顯示啟用物件追蹤的選項。

    Returns:
        Tuple (bool, str): A tuple containing a boolean flag for displaying the tracker and the selected tracker type.
    #返回:
        Tuple (bool, str): 包含是否顯示追蹤器的布林值和選擇的追蹤器類型的元組。
    """

    display_tracker = st.radio("Display Tracker", ("Yes", "No")) # 在 Streamlit 應用中創建一個單選按鈕，用於選擇是否顯示追蹤器
    is_display_tracker = True if display_tracker == "Yes" else False# 根據選擇設置是否顯示追蹤器的布林值
    if display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))# 如果選擇顯示追蹤器，則提供選擇追蹤器類型的單選按鈕
        return is_display_tracker, tracker_type# 返回是否顯示追蹤器和追蹤器類型
    return is_display_tracker, None# 如果不顯示追蹤器，則返回布林值和 None


def display_frames(
    model, acc, st_frame, image, is_display_tracker=None, tracker_type=None
):  #從視頻流中顯示檢測到的物件。
    """
    Displays detectes objects from a video stream.

    Parameters:
        model (YOLO): A YOLO object detection model.  #model (YOLO): YOLO 物件偵測模型。
        acc (float): The model's confidence threshold. #模型的信心閾值。
        st_frame (streamlit.Streamlit): A Streamlit frame object. #框架物件。
        image (PIL.Image.Image): A frame from a video stream.#視頻流中的一幀。
        is_display_tracker (bool): Whether or not to display a tracker. #是否顯示追蹤器。
        tracker_type (str): The type of tracker to display. #要顯示的追蹤器類型。
    
    Returns: #返回
        None
    """

    image = cv2.resize(image, (720, int(720 * (9 / 16)))) # 調整影像大小
    if is_display_tracker:
        res = model.track(image, conf=acc, persist=True, tracker=tracker_type)# 如果啟用追蹤器，則使用追蹤器進行物件追蹤
    else:
        res = model.predict(image, conf=acc)# 如果未啟用追蹤器，則直接進行物件偵測
    
    
    res_plot = res[0].plot()# 繪製偵測結果
    st_frame.image(
        res_plot,
        caption="Detected Video",
        channels="BGR",
        use_column_width=True,
    )# 在 Streamlit 應用中顯示偵測結果
    return res

#res 是之前某個步驟產生的結果，包含多個元素的列表或數據結構。
#res[0] 取出 res 列表中的第一個元素。
#res[0].plot() 調用這個元素的 plot 方法來繪製圖形，並將繪製的圖形賦值給 res_plot 變數。
#st_frame.image(...)

#st_frame 是 Streamlit 的容器或框架元素，用於在 Streamlit 應用中放置和展示內容。
#st_frame.image(...) 是一個 Streamlit 的方法，用於在該框架內顯示圖像。
#res_plot：這是要顯示的圖像，之前生成的偵測結果圖。
#caption="Detected Video"：在圖像下方顯示的文字說明，這裡設置為 "Detected Video"。
#channels="BGR"：指定圖像的顏色通道格式，這裡使用的是 BGR 格式（藍、綠、紅）。
#use_column_width=True：讓圖像自動調整大小以適應 Streamlit 應用中的列寬。
#return res 這行代碼表示函數將 res 作為返回值。這意味著函數執行結束後，會將 res 的值返回給調用它的代碼。


#這段代碼是在一個 Streamlit 應用中用來展示偵測結果的視覺化圖像。它先繪製圖像，然後在 Streamlit 應用的一個框架內展示這個圖像，並且返回偵測結果以供進一步處理


def sum_detections(detected_objects_summary_list, model):
    """
    Summarizes detected objects from a list and displays the summary in a Streamlit success message.
    #從列表中匯總檢測到的物件並在 Streamlit 成功訊息中顯示摘要。
    
    Parameters:
        detected_objects_summary_list (list): List of detected object indices.
    #參數:
        detected_objects_summary_list (list): 檢測到的物件索引列表。
    Returns:
    #返回:
        None
    """
    detected_objects_summary = set() #這行代碼創建了一個空的集合（set），命名為 detected_objects_summary。集合是一種不包含重複元素的數據結構。
    for obj in detected_objects_summary_list: #這行代碼開始一個循環，遍歷 detected_objects_summary_list 列表中的每個元素。每次循環中，obj 變數將會被賦予列表中的下一個元素的值。
        detected_objects_summary.add(model.names[int(obj)]) # 將檢測到的物件名稱添加到集合中
    name_summary = ", ".join(detected_objects_summary)# 將物件名稱連接成字符串
    st.success(f"Detected Objects: {name_summary}")# 在 Streamlit 應用中顯示檢測到的物件名稱

#detected_objects_summary = set()

#這行代碼創建了一個空的集合（set），命名為 detected_objects_summary。集合是一種不包含重複元素的數據結構。
#for obj in detected_objects_summary_list:
#這行代碼開始一個循環，遍歷 detected_objects_summary_list 列表中的每個元素。每次循環中，obj 變數將會被賦予列表中的下一個元素的值。
#detected_objects_summary.add(model.names[int(obj)])
#model.names 可能是一個列表或數組，包含了模型可以識別的對象名稱。
#int(obj) 將 obj 轉換成整數。這假設 obj 是一個可以被轉換為整數的值，例如一個表示類別編號的字符串。
#model.names[int(obj)] 獲取對應於 obj 指定編號的對象名稱。
#.add(...) 方法將這個名稱添加到 detected_objects_summary 集合中。由於集合不允許重複的元素，如果這個名稱已經存在於集合中，則不會重複添加。
#總結來說，這段代碼的目的是從一個包含識別到的對象編號的列表（detected_objects_summary_list）中，提取出每個對象的名稱，並將這些名稱存儲在一個不包含重複元素的集合（detected_objects_summary）中。這樣可以得到所有被識別出的獨特對象的名稱。



def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    #使用 YOLOv8 模型在視頻幀上顯示檢測到的物件。

    Args:
    - conf (float): Confidence threshold for object detection. #物件檢測的信心閾值。
    - model (Yolov8): A YOLOv8 object detection model. #YOLOv8 物件檢測模型。
    - st_frame (Streamlit object): A Streamlit object to display the detected video. #用於顯示檢測視頻的 Streamlit 物件。
    - image (numpy array): A numpy array representing the video frame. #表示視頻幀的 numpy 陣列。
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).
    #指示是否顯示物件追蹤的標誌。
    Returns: #返回
    None
    """
  
    # Resize the image to a standard size 將影像調整為標準大小
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified 如果指定，顯示物件追蹤
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model 使用 YOLOv8 模型預測影像中的物件
        res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame 在視頻幀上繪製檢測到的物件
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )

#註解:res_plotted = res[0].plot()

#res 是一個先前生成的結果列表或其他數據結構。
#res[0] 取出這個結果列表中的第一個元素。
#res[0].plot() 調用這個元素的 plot 方法，該方法負責繪製並返回一幅圖像，這幅圖像顯示了在視頻幀上檢測到的物件。
#繪製的圖像被賦值給 res_plotted 變數。
#st_frame.image(res_plotted, caption='Detected Video', channels="BGR", use_column_width=True)

#st_frame 是 Streamlit 框架中的一個容器或組件。
#st_frame.image(...) 是 Streamlit 方法，用於在指定的框架或容器中顯示圖像。
#res_plotted 是要顯示的圖像，即之前繪製的含有檢測物件的視頻幀。
#caption='Detected Video' 為顯示的圖像設置標題，這裡設為“Detected Video”（檢測到的視頻）。
#channels="BGR" 指定圖像的顏色通道格式，這裡使用 BGR 格式（藍、綠、紅）。
#use_column_width=True 表示圖像將自動調整大小以適應 Streamlit 應用中的列寬。
#總之 ，這段代碼負責在視頻幀上繪製檢測到的物件，並在 Streamlit 應用中的指定容器或框架內展示這個繪製的圖像。






    


def play_stored_video(conf, model):
    """
    #播放存儲的視頻文件。使用 YOLOv8 物件檢測模型實時追蹤和檢測物件。
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.
    
    Parameters: #參數
        conf: Confidence of YOLOv8 model. #YOLOv8 模型的置信度。
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.#包含 YOLOv8 模型的 `YOLOv8` 類的實例。

    Returns:#返回
        None

    Raises:#引發
        None
    """
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())# 在 Streamlit 側邊欄中創建一個下拉選單，用於選擇視頻

    is_display_tracker, tracker = display_tracker_options()# 顯示追蹤器選項

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read() #讀取視頻文件
    if video_bytes:
        st.video(video_bytes) #在Streamlit 應用中播放視頻

    if st.sidebar.button('Detect Video Objects'): # 創建一個按鈕，用於開始檢測視頻中的物件
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))  # 打開視頻文件
            st_frame = st.empty()   # 創建一個空的 Streamlit 框架
            while (vid_cap.isOpened()):   # 當視頻文件打開時
                success, image = vid_cap.read()   # 讀取視頻幀
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             ) # 顯示檢測到的幀
#if success:這是一個條件判斷語句。它檢查變量 success 的值是否為真（True）。如果是，則執行縮進的代碼塊；如果不是，則跳過這個代碼塊。
#_display_detected_frames(conf, model, st_frame, image, is_display_tracker, tracker)

#這是一個函數調用，函數名為 _display_detected_frames。這個函數名前的下劃線（_）通常表示這是一個內部使用的或私有的函數。
#函數接收幾個參數：
#conf：可能是一個配置對象或相關設置。
#model：這可能是一個用於物件檢測的模型。
#st_frame：這是一個 Streamlit 框架或容器，用於在 Streamlit 應用中顯示內容。
#image：這是要進行物件檢測的圖像。
#is_display_tracker：這是一個布爾值（True 或 False），指示是否顯示追蹤器的信息。
#tracker：這可能是一個追蹤器對象，用於追蹤圖像中的物件。
# 顯示檢測到的幀：這是一個註釋，解釋函數調用的目的，即顯示檢測到的視頻幀。
#總之，這段代碼在檢測到成功的情況下，會呼叫 _display_detected_frames 函數來顯示處理後的圖像幀。這個函數可能涉及到在圖像上繪製檢測到的物件、應用追蹤算法，以及將結果顯示在 Streamlit 應用的界面上。
                
                
                
                else:
                    vid_cap.release()  # 釋放視頻捕獲對象
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e)) # 如果出現錯誤，顯示錯誤訊息

#else:

#這是一個與前面的 if 語句相關的 else 語句。如果 if 語句中的條件不成立（即 success 變量的值不是真），則執行 else 塊中的代碼。
#vid_cap.release()

#vid_cap 於視頻捕獲的對象，比如使用 OpenCV 捕獲視頻。
#.release() 是一個方法，用於釋放與 vid_cap 對象相關的資源。這通常在不再需要捕獲視頻時進行，以確保釋放系統資源。
#break

#break 語句用於立即結束最近的循環。在這個上下文中，它可能是結束一個 while 或 for 循環，這個循環可能是用於處理視頻幀。
#except Exception as e:

#這是一個異常捕獲語句。如果在 try 塊中的代碼執行過程中出現任何異常，則執行這個 except 塊。
#Exception 是 Python 中所有內置異常的基類。這裡捕獲的是任何類型的異常。
#as e 表示將捕獲到的異常對象賦值給變量 e。
#st.sidebar.error("Error loading video: " + str(e))

#st.sidebar.error(...) 是 Streamlit 的一個方法，用於在應用的側邊欄顯示錯誤訊息。
#"Error loading video: " + str(e) 是要顯示的錯誤訊息，它將 "Error loading video: "（載入視頻錯誤）和 e 變量的字符串表示連接起來。e 變量包含了關於異常的詳細信息。
#總結來說，這段代碼在處理視頻時，如果沒有成功（即 if 條件不成立），則釋放視頻捕獲對象並結束循環。如果在執行過程中發生任何異常，則捕獲該異常並在 Streamlit 應用的側邊欄中顯示錯誤訊息。










#這段code是用於在 Streamlit 應用中處理 RTSP（Real Time Streaming Protocol）視頻流的。
def play_rtsp_stream(conf, model): #播放 rtsp 流。使用 YOLOv8 物件檢測模型實時檢測物件。
    """
    Plays an rtsp stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters: #參數: 
        conf: Confidence of YOLOv8 model. #YOLOv8 模型的置信度
        model: An instance of the `YOLOv8` class containing the YOLOv8 model. #model: 包含 YOLOv8 模型的 `YOLOv8` 類的實例。


    Returns:#返回
        None

    Raises:#引發
        None
    """
    source_rtsp = st.sidebar.text_input("rtsp stream url:") # 在 Streamlit 側邊欄中創建一個文本輸入框，用於輸入 rtsp 流的 URL
    st.sidebar.caption('Example URL: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101') # 提供一個 rtsp 流 URL 的範例
    is_display_tracker, tracker = display_tracker_options()  # 顯示追蹤器選項
    if st.sidebar.button('Detect Objects'): # 創建一個按鈕，用於開始檢測 rtsp 流中的物件
        try:
            vid_cap = cv2.VideoCapture(source_rtsp) # 打開 rtsp 
            st_frame = st.empty()  # 創建一個空的 Streamlit 框架
            while (vid_cap.isOpened()): # 當 rtsp 打開時
                success, image = vid_cap.read() # 讀取 rtsp 中的幀
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )  # 顯示檢測到的幀
##                    

                else:
                    vid_cap.release()    # 釋放視頻捕獲對象
                    # vid_cap = cv2.VideoCapture(source_rtsp)
                    # time.sleep(0.1)
                    # continue
                    break
        except Exception as e:
            vid_cap.release()  # 釋放視頻捕獲對象
            st.sidebar.error("Error loading RTSP stream: " + str(e)) # 如果出現錯誤，顯示錯誤訊息

#以下事詳細解釋
#source_rtsp = st.sidebar.text_input("rtsp stream url:")

#在 Streamlit 應用的側邊欄中創建一個文本輸入框，用於用戶輸入 RTSP 流的 URL。輸入的值會被賦予給變量 source_rtsp。
#st.sidebar.caption('Example URL: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')

#在側邊欄中顯示一個範例 URL，指導用戶如何輸入 RTSP 流的 URL。
#is_display_tracker, tracker = display_tracker_options()

#呼叫 display_tracker_options 函數，這個函數可能用於顯示並選擇不同的物件追蹤器選項。函數返回的結果被賦值給 is_display_tracker（布爾值，指示是否顯示追蹤器）和 tracker（選擇的追蹤器）。
#if st.sidebar.button('Detect Objects'):

#在側邊欄中創建一個按鈕，當按鈕被點擊時，if 語句內的代碼將被執行。按鈕的文字為 “Detect Objects”。
#try:

#開始一個 try 塊，用於捕獲並處理可能發生的異常。
#vid_cap = cv2.VideoCapture(source_rtsp)

#使用 OpenCV 的 VideoCapture 方法打開 RTSP 流，並將其賦值給變量 vid_cap。
#st_frame = st.empty()

#在 Streamlit 應用中創建一個空的框架，用於後續顯示處理過的視頻幀。
#while (vid_cap.isOpened()):

#使用 while 循環來不斷讀取 RTSP 流，只要該流被成功打開。
#success, image = vid_cap.read()

#從 RTSP 流中讀取下一個視頻幀。如果成功讀取，success 為真，並且 image 包含該幀的圖像數據。
#if success:

#如果成功讀取到視頻幀，則執行接下來的代碼。
#_display_detected_frames(conf, model, st_frame, image, is_display_tracker, tracker)

#呼叫 _display_detected_frames 函數來處理並顯示檢測到的幀。該函數可能會在幀上繪製物件檢測的結果。
#else:

#如果無法成功讀取視頻幀，則執行 else 塊的代碼。
#vid_cap.release()

#釋放與 vid_cap 對象相關的資源。
#break

#結束 while 循環。
#except Exception as e:

#如果在 try 塊中的代碼執行過程中出現異常，則執行 except 塊的代碼。
#vid_cap.release()

#再次釋放與 vid_cap 對象相關的資源。
#st.sidebar.error("Error loading RTSP stream: " + str(e))

#在 Streamlit 應用的側邊欄顯示錯誤訊息，告知用戶載入 RTSP 流時發生錯誤，並附上具體的錯誤信息。            
