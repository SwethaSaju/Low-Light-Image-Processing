import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance

st.set_page_config(page_title="Low-Light Image Enhancement", layout="wide")

st.title("🌙 Low-Light Image Enhancement (Pure-Python Demo)")
st.write("Upload a low-light image and enhance it using brightness, contrast and gamma correction — all in Pillow + NumPy (no OpenCV).")

def enhance_image(img, brightness=1.3, contrast=1.4, gamma=1.2):
    # Brightness & contrast using Pillow
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)

    # Gamma correction using NumPy
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.power(arr, 1.0 / gamma)
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

uploaded_file = st.file_uploader("📸 Upload a Low-Light Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("✨ Enhance Image"):
        with st.spinner("Enhancing..."):
            enhanced = enhance_image(image)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(image, use_container_width=True)
        with col2:
            st.subheader("Enhanced")
            st.image(enhanced, use_container_width=True)

        # Optional download
        st.download_button(
            label="⬇️ Download Enhanced Image",
            data=enhanced.tobytes(),
            file_name="enhanced.png",
            mime="image/png",
        )
else:
    st.info("👆 Upload a low-light image to begin.")

st.markdown("""
---
**Project:** Low-Light Image Enhancement Using U-Net (Demo version without heavy libs)  
**Developed by:** Swetha S  
**Tech Stack:** Python 3.13 · Streamlit · Pillow · NumPy
""")
