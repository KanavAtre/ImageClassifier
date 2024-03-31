import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile
import os

st.title("Edge Detection Application")
st.write("Upload an image to begin.")


def apply_edge_detection(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edge_image = cv2.Canny(gray_image, 100, 200)

    return edge_image


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True, width=300)


    st.write("Image Details:")
    st.write(f"Format: {image.format}")
    st.write(f"Size: {image.size}")
    st.write(f"Mode: {image.mode}")

    if st.button("Perform Edge Detection"):
      
        opencv_image = np.array(image)


        edge_detected_image = apply_edge_detection(opencv_image)


        pil_edge_detected_image = Image.fromarray(edge_detected_image)

        # Display the edge-detected image
        st.image(pil_edge_detected_image, caption='Edge Detected Image', use_column_width=True)

        # Download button for the edge-detected image
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            pil_edge_detected_image.save(temp_file.name, format="PNG")
            st.download_button(
                label="Download Edge Detected Image",
                data=open(temp_file.name, "rb").read(),
                file_name="edge_detected_image.png",
                mime="image/png"
            )

