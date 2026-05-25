import streamlit as st
from PIL import Image
import numpy as np

from src.utils.xray_classifier import xray_full_pipeline

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="AI-CDD X-ray TB", layout="wide")

# -----------------------------
# HEADER
# -----------------------------
st.title("🩻 AI-CDD: Tuberculosis Detection (X-ray)")
st.markdown("Upload an X-ray image. The system will process it through all stages and predict TB.")

# -----------------------------
# UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload X-ray Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("📌 Uploaded Image")
    st.image(image, use_container_width=True)

    st.markdown("---")

    # -----------------------------
    # RUN FULL PIPELINE
    # -----------------------------
    with st.spinner("Processing through pipeline..."):

        original, stage1, stage2, stage3, stage4, prediction, confidence = xray_full_pipeline(image)

    st.success("Processing Complete")

    # -----------------------------
    # SHOW PIPELINE (DEBUG VIEW)
    # -----------------------------
    st.subheader("🔬 Pipeline Stages")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Stage 1: Preprocessing (CLAHE)**")
        st.image(stage1, clamp=True)

    with col2:
        st.markdown("**Stage 2: UNet Output**")
        st.image(stage2, clamp=True)

    with col3:
        st.markdown("**Stage 3: Clean Mask**")
        st.image(stage3 * 255, clamp=True)

    st.markdown("---")

    col4, col5 = st.columns(2)

    with col4:
        st.markdown("**Original**")
        st.image(original, clamp=True)

    with col5:
        st.markdown("**Stage 4: Lung ROI (Model Input)**")
        st.image(stage4, clamp=True)

    st.markdown("---")

    # -----------------------------
    # FINAL RESULT
    # -----------------------------
    st.subheader("🧾 Diagnosis")

    if prediction == 1:
        st.error(f"⚠️ Tuberculosis Detected (Confidence: {confidence:.2f})")
    else:
        st.success(f"✅ Normal (Confidence: {confidence:.2f})")

    # -----------------------------
    # IMPORTANT NOTE
    # -----------------------------
    st.caption("Note: Confidence is model probability, not accuracy.")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("AI-CDD • End-to-End TB Detection Pipeline")