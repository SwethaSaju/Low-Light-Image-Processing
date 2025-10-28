import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Low-Light Image Enhancement", layout="wide")

st.title("🌙 Low-Light Image Enhancement (OpenCV Demo)")
st.write("Upload a low-light image and apply enhancement using classical image processing methods.")

def enhance_image_opencv(img):
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE for brightness and contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)

    lab = cv2.merge((l2,a,b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Optional: gamma correction
    gamma = 1.2
    enhanced = np.power(enhanced / 255.0, gamma) * 255
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    return enhanced

uploaded_file = st.file_uploader("📸 Upload a Low-Light Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("✨ Enhance Image"):
        with st.spinner("Enhancing..."):
            enhanced = enhance_image_opencv(img_np)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(image, use_container_width=True)
        with col2:
            st.subheader("Enhanced")
            st.image(enhanced, use_container_width=True)
else:
    st.info("👆 Upload a low-light image to begin.")

st.markdown("""
---
**Project:** Low-Light Image Enhancement Using U-Net (Demo version without model)  
**Developed by:** Swetha S  
**Tech Stack:** Python, OpenCV, Streamlit  
""")
