import os
import cv2
import numpy as np
from tqdm import tqdm

# -----------------------------
# PATHS
# -----------------------------
BASE_IN = r"G:\Ai-CDD\data\preprocessed\stage2"
BASE_OUT = r"G:\Ai-CDD\data\preprocessed\stage3"

# -----------------------------
# HELPERS
# -----------------------------
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


# -----------------------------
# PROCESS FUNCTION
# -----------------------------
def process_split(split):
    for class_name in ["NORMAL", "TUBERCULOSIS"]:
        input_dir = os.path.join(BASE_IN, split, class_name)
        output_dir = os.path.join(BASE_OUT, split, class_name)

        if not os.path.exists(input_dir):
            print(f"(xray_mask_processing.py)❌ Missing: {input_dir}")
            continue

        os.makedirs(output_dir, exist_ok=True)

        files = os.listdir(input_dir)

        processed, failed = 0, 0

        for file in tqdm(files, desc=f"{split}/{class_name}"):

            input_path = os.path.join(input_dir, file)

            pred = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if pred is None:
                failed += 1
                continue

            pred = pred.astype(np.float32) / 255.0

            # -------------------------
            # CLEAN MASK PIPELINE
            # -------------------------

            # 1. Threshold
            mask = (pred > 0.5).astype(np.uint8) * 255

            # 2. Remove noise
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=2)

            # 3. Close gaps
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

            # 4. Keep lungs
            mask = keep_largest_components(mask, 2)

            # 5. Fill holes
            mask = fill_holes(mask)

            output_path = os.path.join(output_dir, file)

            if not cv2.imwrite(output_path, mask):
                failed += 1
                continue

            processed += 1

        print(f"\n✅ {split}/{class_name}")
        print(f"✔ Processed: {processed}")
        print(f"❌ Failed: {failed}")
        print("-" * 40)


# -----------------------------
# RUN
# -----------------------------
for split in ["train", "val", "test"]:
    process_split(split)

print("\n🎯 Stage 3 mask cleaning complete")