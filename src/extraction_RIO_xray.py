import os
import cv2
import numpy as np
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
ORIGINAL_DIR = "data/raw/train/preprocessed_xray_images"
MASK_DIR = "data/processed/masks_clean"
OUTPUT_DIR = "data/processed/lung_roi"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# ROI EXTRACTION
# -----------------------------
def extract_lung_roi(image_path, mask_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        return None

    # Ensure same size
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # Apply mask
    lung = cv2.bitwise_and(image, image, mask=mask)

    return lung

# -----------------------------
# RUN ON DATASET
# -----------------------------
image_files = os.listdir(ORIGINAL_DIR)

for file in tqdm(image_files):
    img_path = os.path.join(ORIGINAL_DIR, file)
    mask_path = os.path.join(MASK_DIR, file)

    if not os.path.exists(mask_path):
        continue

    lung = extract_lung_roi(img_path, mask_path)

    if lung is None:
        continue

    save_path = os.path.join(OUTPUT_DIR, file)
    cv2.imwrite(save_path, lung)

print("✅ Stage 5 completed: Lung ROI images saved")