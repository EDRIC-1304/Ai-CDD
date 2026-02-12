import streamlit as st
from PIL import Image
from src.utils.scan_type_classifier import predict_scan_type
from src.utils.ct_classifier import segment_ct_lung
from src.utils.xray_classifier import xray_process

prediction_dict = {1: "CT scan", 2: "X-ray"}

st.set_page_config(page_title="AI-CDD", layout="centered")
st.title("Image Upload Interface")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg", "jfif"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.session_state["image"] = image
    st.image(image, caption="Original Image", use_container_width=True)

if st.button("Analyze Image"):
    if "image" not in st.session_state:
        st.warning("Upload an image first.")
    else:
        image = st.session_state["image"]

        scan_type = prediction_dict[predict_scan_type(image)]
        st.success(f"Classification complete → {scan_type}")

        if scan_type == "CT scan":
            orig, binary_mask, morph_mask = segment_ct_lung(image)
            st.success("Segmentation complete")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original")
                st.image(orig, clamp=True)

            with col2:
                st.subheader("After Morphology")
                st.image(morph_mask * 255, clamp=True)
        
        elif scan_type == "X-ray":
            original, morph_close = xray_process(image)
            st.success("X-ray processing complete")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original")
                st.image(original, clamp=True)

            with col2:
                st.subheader("After Morphological Closing")
                st.image(morph_close, clamp=True)
