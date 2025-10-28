import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from PIL import Image
import cv2

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(page_title="Low-Light Image Enhancement", layout="wide")

st.title("🌙 Low-Light Image Enhancement Using U-Net")
st.write("Upload a low-light image and let the trained U-Net model enhance it!")

# ------------------------------
# Load U-Net Model
# ------------------------------
@st.cache_resource
def load_unet_model():
    # Replace 'unet_model.h5' with your trained model path
    model = load_model("unet_model.h5", compile=False)
    return model

model = load_unet_model()

# ------------------------------
# Preprocessing Function
# ------------------------------
def preprocess_image(image: Image.Image):
    img = np.array(image.convert("RGB"))
    img = cv2.resize(img, (256, 256))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# ------------------------------
# Postprocessing Function
# ------------------------------
def postprocess_image(pred):
    pred = np.clip(pred[0], 0, 1)
    pred = (pred * 255).astype(np.uint8)
    return Image.fromarray(pred)

# ------------------------------
# Image Upload
# ------------------------------
uploaded_file = st.file_uploader("📸 Upload a Low-Light Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption="Uploaded Image", use_container_width=True)

    if st.button("✨ Enhance Image"):
        with st.spinner("Enhancing image... please wait ⏳"):
            input_data = preprocess_image(input_image)
            enhanced = model.predict(input_data)
            enhanced_img = postprocess_image(enhanced)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(input_image, use_container_width=True)
        with col2:
            st.subheader("Enhanced Image")
            st.image(enhanced_img, use_container_width=True)

        # ------------------------------
        # Metrics Calculation (optional)
        # ------------------------------
        original = np.array(input_image.resize((256, 256))).astype("float32") / 255.0
        enhanced_np = np.array(enhanced_img).astype("float32") / 255.0
        psnr_val = psnr(original, enhanced_np, data_range=1.0)
        ssim_val = ssim(original, enhanced_np, channel_axis=2, data_range=1.0)

        st.metric("🔹 PSNR", f"{psnr_val:.2f} dB")
        st.metric("🔹 SSIM", f"{ssim_val:.3f}")

else:
    st.info("👆 Upload a low-light image to begin.")

# ------------------------------
# Footer
# ------------------------------
st.markdown("""
---
**Project:** Low-Light Image Enhancement Using U-Net  
**Developed by:** Swetha S  
**Tech Stack:** Python, TensorFlow, Streamlit  
""")
