import cv2
import numpy as np
import os
import tensorflow as tf

HOME_DIR = "D:/FY_Project/Ai-CDD"
OUTPUT_DIR = HOME_DIR + "/data/processed/XRAY"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 256

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------
model = tf.keras.models.load_model(
    HOME_DIR + "/models_saved/xray_unet_lung.h5",
    compile=False
)

# ------------------------------------------------
# HELPERS
# ------------------------------------------------
def keep_largest_components(mask, num_components=2):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = np.argsort(areas)[-num_components:] + 1

    clean = np.zeros_like(mask)
    for idx in largest:
        clean[labels == idx] = 255

    return clean


def fill_holes(mask):
    h, w = mask.shape
    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    return mask | flood_inv


# ------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------
def xray_process(image):
    image = np.array(image)
    original = image.copy()

    # --- Preprocessing (unchanged) ---
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    denoised = cv2.medianBlur(equalized, 5)
    contrast = cv2.convertScaleAbs(denoised, alpha=1.5, beta=0)

    h, w = contrast.shape

    # --- Model inference ---
    unet_input = cv2.resize(contrast, (IMG_SIZE, IMG_SIZE)) / 255.0
    unet_input = unet_input[np.newaxis, ..., np.newaxis]

    pred = model.predict(unet_input, verbose=0)[0, :, :, 0]
    pred_resized = cv2.resize(pred, (w, h))

    # 🔥 LOWER THRESHOLD (CRITICAL)
    binary_mask = (pred_resized > 0.35).astype(np.uint8) * 255

    # 🔥 CLOSE FIRST (reconnect right lung)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # 🔥 FILL BEFORE FILTERING
    mask = fill_holes(mask)

    # 🔥 KEEP ONLY TWO LUNGS
    mask = keep_largest_components(mask, num_components=2)

    # 🔥 VERY LIGHT OPENING (edge cleanup only)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # --- Save ---
    output_path = OUTPUT_DIR + "/xray_output.png"
    cv2.imwrite(output_path, mask)

    print("✅ XRAY lung segmentation completed")
    print("Saved to:", output_path)

    return original, mask
