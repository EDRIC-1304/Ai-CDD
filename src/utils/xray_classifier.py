import cv2
import numpy as np
import os

from streamlit import image
import tensorflow as tf
import matplotlib.pyplot as plt

HOME_DIR = "D:/FY_Project/Ai-CDD"
OUTPUT_DIR = HOME_DIR + "/data/processed/XRAY"
os.makedirs(OUTPUT_DIR, exist_ok=True)
IMG_SIZE = 256
print(tf.__version__)
# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------
model = tf.keras.models.load_model(HOME_DIR+"/models_saved/xray_unet_lung.h5")


def xray_process(image):
    image = np.array(image)
    original = image
    

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    denoised = cv2.medianBlur(equalized, 5)
    contrast = cv2.convertScaleAbs(denoised, alpha=1.5, beta=0)
    h, w = contrast.shape
    unet_input = cv2.resize(contrast, (IMG_SIZE, IMG_SIZE)) / 255.0
    unet_input = np.expand_dims(unet_input, axis=0)
    unet_input = np.expand_dims(unet_input, axis=-1)

    pred = model.predict(unet_input, verbose=0)[0, :, :, 0]
    pred_resized = cv2.resize(pred, (w, h))
    unet_output = (pred_resized * 255).astype(np.uint8)

    _, binary_mask = cv2.threshold(unet_output, 127, 255, cv2.THRESH_BINARY)
    if binary_mask[5, 5] == 255:
        binary_mask = cv2.bitwise_not(binary_mask)

    coords = np.where(binary_mask > 0)
    if coords[0].size == 0:
        print("⚠️ No ROI detected")
        return

    top, bottom = np.min(coords[0]), np.max(coords[0])
    left, right = np.min(coords[1]), np.max(coords[1])

    cropped_mask = binary_mask[top:bottom, left:right]
    cropped_original = original[top:bottom, left:right]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph_open = cv2.morphologyEx(cropped_mask, cv2.MORPH_OPEN, kernel)
    morph_close = cv2.morphologyEx(morph_open, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite(OUTPUT_DIR+"/xray_output.png", morph_close)
    print("Workflow completed")

    return original, morph_close

