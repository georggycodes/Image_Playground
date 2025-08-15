import streamlit as st
import numpy as np
from PIL import Image
import cv2
st.set_page_config(page_title="Image Processing Playground", layout="wide")
st.title("🖼️ Image Processing Playground")


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Read file bytes into a numpy array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    # Decode into an OpenCV image (BGR format)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Error: Could not decode the image.")
    else:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        st.image(gray, channels="GRAY", caption="Grayscale Image")
else:
    st.warning("Please upload an image to proceed.")
    st.stop()  # This stops execution so 'image' won't be undefined


#processing Technique Selection
option = st.sidebar.selectbox(
        "Select Technique",
        ["Thresholding", "Blurring", "Edge Detection", "Contour Detection",
         "Template Matching", "Watershed Segmentation", "Color Space Conversion"]
    )

#Technique Implementation
if option == "Thresholding":
    method = st.sidebar.selectbox(
        "Method",
        ["Simple Binary", "Adaptive Mean", "Adaptive Gaussian", "Otsu"]
    )
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if method == "Simple Binary":
        thresh_val = st.sidebar.slider("Threshold", 0, 255, 128)
        _, processed = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    elif method == "Adaptive Mean":
        processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, 11, 2)

    elif method == "Adaptive Gaussian":
        processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)

    elif method == "Otsu":
        _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    st.image([gray, processed], caption=["Grayscale", "Thresholded"], width=300)

#Blurring
if option == "Blurring":
    blur_type = st.sidebar.selectbox("Blur Type", ["Averaging", "Gaussian", "Median"])
    ksize = st.sidebar.slider("Kernel Size", 1, 15, 3, step=2)

    if blur_type == "Averaging":
        processed = cv2.blur(image, (ksize, ksize))
    elif blur_type == "Gaussian":
        processed = cv2.GaussianBlur(image, (ksize, ksize), 0)
    elif blur_type == "Median":
        processed = cv2.medianBlur(image, ksize)

    st.image([image
              , processed], caption=["Original", "Blurred"], width=300)

#Edge Detection
if option == "Edge Detection":
    method = st.sidebar.selectbox("Method", ["Sobel", "Laplacian", "Canny"])
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if method == "Sobel":
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
            processed = cv2.magnitude(sobelx, sobely)
    elif method == "Laplacian":
        processed = cv2.Laplacian(gray, cv2.CV_64F)
    elif method == "Canny":
        t1 = st.sidebar.slider("Threshold 1", 0, 255, 100)
        t2 = st.sidebar.slider("Threshold 2", 0, 255, 200)
        processed = cv2.Canny(gray, t1, t2)

    st.image([gray, processed], caption=["Grayscale", "Edges"], width=300, clamp=True)

#Contour Detection
if option == "Contour Detection":
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    processed = image.copy()
    cv2.drawContours(processed, contours, -1, (0,255,0), 2)
    st.image([image, processed], caption=["Original", "With Contours"], width=300)

#Template Matching
if option == "Template Matching":
    st.info("Upload a template image:")
    template_file = st.file_uploader("Template", type=["jpg", "jpeg", "png"], key="template")
    if template_file:
        template = np.array(Image.open(template_file))
        method = cv2.TM_CCOEFF_NORMED
        res = cv2.matchTemplate(image, template, method)
        _, _, _, max_loc = cv2.minMaxLoc(res)
        h, w = template.shape[:2]
        processed = image.copy()
        cv2.rectangle(processed, max_loc, (max_loc[0]+w, max_loc[1]+h), (0,255,0), 2)
        st.image([image, processed], caption=["Original", "Match Found"], width=300)

#Water segmentation
if option == "Watershed Segmentation":
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers+1
    markers[unknown==255] = 0
    markers = cv2.watershed(image, markers)
    processed = image.copy()
    processed[markers == -1] = [255,0,0]
    st.image([image, processed], caption=["Original", "Segmented"], width=300)

#Color Space Conversion
if option == "Color Space Conversion":
    method = st.sidebar.selectbox("Conversion", ["BGR2GRAY", "BGR2HSV", "BGR2RGB"])
    if method == "BGR2GRAY":
        processed = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif method == "BGR2HSV":
        processed = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif method == "BGR2RGB":
        processed = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image([image, processed], caption=["Original", f"{method}"], width=300)


