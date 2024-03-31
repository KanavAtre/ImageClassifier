import streamlit as st
from PIL import Image
import numpy as np
import cv2


def apply_edge_detection(image, sigma=2, kernel_size=(3, 3), threshold=75):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.GaussianBlur(gray_image, kernel_size, sigma)

    # Perform edge detection using non-maximum suppression
    def non_max_suppression(image):
        gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)
        gradient_direction = cv2.phase(gradient_x, gradient_y, angleInDegrees=True)

        suppressed_image = gradient_magnitude.copy()

        rows, cols = gradient_magnitude.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                direction = gradient_direction[i, j]

                if (0 <= direction < 22.5) or (157.5 <= direction <= 180):
                    nh1 = (i, j + 1)
                    nh2 = (i, j - 1)
                elif (22.5 <= direction < 67.5):
                    nh1 = (i - 1, j + 1)
                    nh2 = (i + 1, j - 1)
                elif (67.5 <= direction < 112.5):
                    nh1 = (i - 1, j)
                    nh2 = (i + 1, j)
                else:
                    nh1 = (i - 1, j - 1)
                    nh2 = (i + 1, j + 1)

                if gradient_magnitude[i, j] < gradient_magnitude[nh1] or gradient_magnitude[i, j] < gradient_magnitude[
                    nh2]:
                    suppressed_image[i, j] = 0

        return suppressed_image

    edge_image = non_max_suppression(blurred_image)

    edge_image[edge_image < threshold] = 0

    return edge_image

st.title("Image Upload App")
st.write("Upload an image to see its details.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the image data
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=False)

    # Get image details
    st.write("Image Details:")
    st.write(f"Format: {image.format}")
    st.write(f"Size: {image.size}")
    st.write(f"Mode: {image.mode}")

    if st.button("Perform Edge Detection"):
        # Convert the Pillow image to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Apply edge detection
        edge_detected_image = apply_edge_detection(opencv_image)

        # Convert the edge-detected image back to Pillow format
        pil_edge_detected_image = Image.fromarray(cv2.cvtColor(edge_detected_image, cv2.COLOR_BGR2RGB))

        # Display the edge-detected image
        st.image(pil_edge_detected_image, caption='Edge Detected Image', use_column_width=True)
