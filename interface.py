import streamlit as st
from PIL import Image
import numpy as np

st.title("Image Upload App")
st.write("Upload an image to see its details.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the image data
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Get image details
    st.write("Image Details:")
    st.write(f"Format: {image.format}")
    st.write(f"Size: {image.size}")
    st.write(f"Mode: {image.mode}")

    # Display the image data as a NumPy array
    st.write("Image Data (NumPy Array):")
    st.write(np.array(image))
