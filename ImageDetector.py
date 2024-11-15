import PIL
import settings
import streamlit as st
import helper

class ImageDetector:
    def __init__(self, model, accuracy):
        self.model = model
        self.accuracy = accuracy

    def detect(self):
        image_process = None
        source_image = st.sidebar.file_uploader("Upload an image", type=("jpg", "jpeg", "png", "bmp", "webp"))
        col1, col2 = st.columns(2)
        with col1:
            try:
                if source_image is not None:
                    image_process = PIL.Image.open(source_image)
                    st.image(image_process, caption="Uploaded Image", use_column_width=True)
                else:
                    default_image = PIL.Image.open(settings.DEFAULT_IMAGE)
                    st.image(default_image, caption="Default Image", use_column_width=True)
                    image_process = default_image
            except Exception as ex:
                st.error(f"Error loading image")
                st.error(ex)
        if st.sidebar.button("Detect Objects"):
            detected_objects_summary_list = []
            res = self.model.predict(image_process, conf=self.accuracy)
            boxes = res[0].boxes
            res_plotted = res[0].plot()[:,:,::-1]
            detected_objects_summary_list.extend(res[0].boxes.cls)
            with col2:
                st.image(res_plotted, caption='Detected Image', use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        if not boxes:
                            st.write("No objects detected")
                        else:
                            for box in boxes:
                                st.write(box.xywh)
                except Exception as ex:
                    st.write("An error occurred while processing the detection results")
            if boxes:
                helper.sum_detections(detected_objects_summary_list, self.model)
