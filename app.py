import streamlit as st
import cv2
import numpy as np

from roboflow import Roboflow
rf = Roboflow(api_key="nECPBxrHa2AFNSOxLPla")
project = rf.workspace().project("engine-detector")
model = project.version(3).model

st.title ("Car Engine Detecting")
st.caption("developed by Black Dasher")

uploaded_file = st.file_uploader("Choose a image file", type=['jpg'])

col1, col2 = st.columns(2)

with col1:

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        st.image(opencv_image, channels="BGR")
        cv2.imwrite('uploaded.jpg',opencv_image)

with col2:

    if uploaded_file is not None:
        print(model.predict("output.jpg", confidence=40, overlap=30).json())
        model.predict("uploaded.jpg", confidence=40, overlap=30).save("result.jpg")
        result_image=cv2.imread("result.jpg")
        st.image(result_image, channels="BGR")