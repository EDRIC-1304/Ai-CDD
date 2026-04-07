import streamlit as st
from PIL import Image
import numpy as np

from src.utils.scan_type_classifier import predict_scan_type
from src.utils.ct_classifier import segment_ct_lung
from src.utils.xray_classifier import xray_process

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="AI-CDD", layout="wide")

prediction_dict = {1: "CT scan", 2: "X-ray"}

# -----------------------------
# UI HEADER
# -----------------------------
st.title("🧠 AI-CDD: TB Detection System")
st.markdown("Upload a scan to analyze lung regions and detect Tuberculosis.")

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg", "jfif"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.session_state["image"] = image

    st.subheader("📌 Uploaded Image")
    st.image(image, use_container_width=True)

# -----------------------------
# ANALYZE BUTTON
# -----------------------------
if st.button("Analyze Image"):

    if "image" not in st.session_state:
        st.warning("Upload an image first.")
        st.stop()

    image = st.session_state["image"]

    # -----------------------------
    # STEP 1: SCAN TYPE
    # -----------------------------
    scan_type = prediction_dict[predict_scan_type(image)]
    st.success(f"Scan Type Detected → {scan_type}")

    st.markdown("---")

    # =============================
    # CT PIPELINE
    # =============================
    if scan_type == "CT scan":

        orig, binary_mask, morph_mask = segment_ct_lung(image)

        st.subheader("🧪 CT Segmentation Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Original**")
            st.image(orig, clamp=True)

        with col2:
            st.markdown("**Binary Mask**")
            st.image(binary_mask * 255, clamp=True)

        with col3:
            st.markdown("**Cleaned Mask**")
            st.image(morph_mask * 255, clamp=True)

        st.info("CT classification not implemented yet.")

    # =============================
    # X-RAY PIPELINE
    # =============================
    elif scan_type == "X-ray":

        # IMPORTANT: this function MUST return 4 outputs
        # original, mask, roi, prediction
        original, mask, roi, prediction = xray_process(image)

        st.subheader("🩻 X-ray Processing Pipeline")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Original**")
            st.image(original, clamp=True)

        with col2:
            st.markdown("**Stage 3: Lung Mask**")
            st.image(mask * 255, clamp=True)

        with col3:
            st.markdown("**Stage 4: ROI (Model Input)**")
            st.image(roi, clamp=True)

        st.markdown("---")

        # -----------------------------
        # FINAL PREDICTION
        # -----------------------------
        st.subheader("🧾 Final Diagnosis")

        if prediction == 1:
            st.error("⚠️ Tuberculosis Detected")
        else:
            st.success("✅ Normal")

        # Optional: debug info
        st.caption("ROI is the actual input used by the model for prediction.")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("AI-CDD • Medical Imaging Pipeline Demo")