import os
import cv2
import numpy as np
from tqdm import tqdm

# -----------------------------
# BASE PATHS
# -----------------------------
BASE_IMG = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/processed/xray_image_preprocessing"
BASE_MASK = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/processed/xray_masks_clean"
BASE_OUT = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/processed/xray_lung_roi"

DATASETS = [
    ("test/Normal", "test/Normal"),
    ("test/TB", "test/TB"),
    ("train/normal", "train/normal"),
    ("train/TB", "train/TB"),
    ("validation/Normal", "validation/Normal"),
    ("validation/TB", "validation/TB"),
]

# -----------------------------
# ROI EXTRACTION FUNCTION
# -----------------------------
def extract_lung_roi(image, mask):

    # Ensure same size (safety check)
    if image.shape != mask.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # Ensure mask is binary (important)
    mask = (mask > 127).astype(np.uint8) * 255

    # Apply mask
    lung = cv2.bitwise_and(image, image, mask=mask)

    return lung

# -----------------------------
# PROCESS DATASET
# -----------------------------
def process_dataset(img_dir, mask_dir, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    img_files = os.listdir(img_dir)

    processed = 0
    skipped = 0

    for file in tqdm(img_files):

        img_path = os.path.join(img_dir, file)
        mask_path = os.path.join(mask_dir, file)

        if not os.path.exists(mask_path):
            print("MASK PATH", mask_path, os.path.exists(mask_path))
            skipped += 1
            continue

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            skipped += 1
            continue

        lung = extract_lung_roi(image, mask)

        save_path = os.path.join(out_dir, file)
        success = cv2.imwrite(save_path, lung)

        if not success:
            skipped += 1
            continue

        processed += 1

    print(f"\n✅ Completed: {img_dir}")
    print(f"✔ ROI Saved: {processed}")
    print(f"❌ Skipped: {skipped}")
    print("-" * 50)

# -----------------------------
# RUN ALL DATASETS
# -----------------------------
for sub_in, sub_out in DATASETS:

    img_path = os.path.join(BASE_IMG, sub_in)
    mask_path = os.path.join(BASE_MASK, sub_in)
    out_path = os.path.join(BASE_OUT, sub_out)

    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        print(f"❌ Missing path: {sub_in}")
        continue

    process_dataset(img_path, mask_path, out_path)

print("✅ Stage 5 completed for all datasets")