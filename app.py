import streamlit as st
import numpy as np
import onnxruntime_lite as ort

from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from PIL import Image
import cv2

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(page_title="Low-Light Image Enhancement Using U-Net", layout="wide")

st.title("🌙 Low-Light Image Enhancement Using U-Net")
st.write("Upload a low-light image to enhance it using a pre-trained U-Net model (ONNX Runtime).")

# ------------------------------
# Load U-Net ONNX Model
# ------------------------------
@st.cache_resource
def load_unet_model():
    try:
        model = ort.InferenceSession("unet_model.onnx")
        return model
    except Exception as e:
        st.error(f"Model could not be loaded: {e}")
        return None

model = load_unet_model()

# ------------------------------
# Preprocessing Function
# ------------------------------
def preprocess_image(image: Image.Image):
    img = np.array(image.convert("RGB"))
    img = cv2.resize(img, (256, 256))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ------------------------------
# Postprocessing Function
# ------------------------------
def postprocess_image(pred):
    pred = np.clip(pred[0], 0, 1)
    pred = (pred * 255).astype(np.uint8)
    return Image.fromarray(pred)

# ------------------------------
# Prediction Function
# ------------------------------
def predict_image(model, img):
    inputs = {model.get_inputs()[0].name: img.astype(np.float32)}
    pred = model.run(None, inputs)[0]
    return pred

# ------------------------------
# Image Upload Section
# ------------------------------
uploaded_file = st.file_uploader("📸 Upload a Low-Light Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption="Uploaded Low-Light Image", use_container_width=True)

    if st.button("✨ Enhance Image"):
        if model is not None:
            with st.spinner("Enhancing image... please wait ⏳"):
                input_data = preprocess_image(input_image)
                enhanced = predict_image(model, input_data)
                enhanced_img = postprocess_image(enhanced)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(input_image, use_container_width=True)
            with col2:
                st.subheader("Enhanced Image")
                st.image(enhanced_img, use_container_width=True)

            # ------------------------------
            # Metrics Calculation
            # ------------------------------
            original = np.array(input_image.resize((256, 256))).astype("float32") / 255.0
            enhanced_np = np.array(enhanced_img).astype("float32") / 255.0
            psnr_val = psnr(original, enhanced_np, data_range=1.0)
            ssim_val = ssim(original, enhanced_np, channel_axis=2, data_range=1.0)

            st.metric("🔹 PSNR", f"{psnr_val:.2f} dB")
            st.metric("🔹 SSIM", f"{ssim_val:.3f}")
        else:
            st.error("Model not loaded. Please check your model file (unet_model.onnx).")

else:
    st.info("👆 Please upload a low-light image to begin.")

# ------------------------------
# Footer
# ------------------------------
st.markdown("""
---
**Project:** Low-Light Image Enhancement Using U-Net  
**Developed by:** Swetha S  
**Tech Stack:** Python, ONNX Runtime, Streamlit  
""")

