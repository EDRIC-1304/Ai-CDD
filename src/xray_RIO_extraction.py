import os
import cv2
import numpy as np
from tqdm import tqdm

# -----------------------------
# PATHS
# -----------------------------
BASE_IMG = r"G:\Ai-CDD\data\preprocessed\stage1"
BASE_MASK = r"G:\Ai-CDD\data\preprocessed\stage3"
BASE_OUT = r"G:\Ai-CDD\data\preprocessed\stage4"

# -----------------------------
# ROI FUNCTION
# -----------------------------
def extract_lung_roi(image, mask):

    if image.shape != mask.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    mask = (mask > 127).astype(np.uint8) * 255

    lung = cv2.bitwise_and(image, image, mask=mask)

    return lung

# -----------------------------
# PROCESS
# -----------------------------
def process_split(split):
    for class_name in ["NORMAL", "TUBERCULOSIS"]:

        img_dir = os.path.join(BASE_IMG, split, class_name)
        mask_dir = os.path.join(BASE_MASK, split, class_name)
        out_dir = os.path.join(BASE_OUT, split, class_name)

        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            print(f"❌ Missing: {split}/{class_name}")
            continue

        os.makedirs(out_dir, exist_ok=True)

        files = os.listdir(img_dir)

        processed, skipped = 0, 0

        for file in tqdm(files, desc=f"{split}/{class_name}"):

            img_path = os.path.join(img_dir, file)
            mask_path = os.path.join(mask_dir, file)

            if not os.path.exists(mask_path):
                skipped += 1
                continue

            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if image is None or mask is None:
                skipped += 1
                continue

            lung = extract_lung_roi(image, mask)

            out_path = os.path.join(out_dir, file)

            if not cv2.imwrite(out_path, lung):
                skipped += 1
                continue

            processed += 1

        print(f"\n✅ {split}/{class_name}")
        print(f"✔ ROI Saved: {processed}")
        print(f"❌ Skipped: {skipped}")
        print("-" * 40)

# -----------------------------
# RUN
# -----------------------------
for split in ["train", "val", "test"]:
    process_split(split)

print("\n🎯 Stage 4 ROI extraction complete")