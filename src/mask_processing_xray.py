import os
import cv2
import numpy as np
from tqdm import tqdm

# -----------------------------
# BASE PATHS
# -----------------------------
BASE_IN = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/processed/xray_unet_output"
BASE_OUT = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/processed/xray_masks_clean"

DATASETS = [
    ("test/Normal", "test/Normal"),
    ("test/TB", "test/TB"),
    ("train/normal", "train/normal"),
    ("train/TB", "train/TB"),
    ("validation/Normal", "validation/Normal"),
    ("validation/TB", "validation/TB"),
]

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
# MAIN PROCESS FUNCTION
# -----------------------------
def process_dataset(input_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    files = os.listdir(input_dir)

    processed = 0
    failed = 0

    for file in tqdm(files):

        input_path = os.path.join(input_dir, file)

        # Load probability mask (grayscale)
        pred = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        if pred is None:
            failed += 1
            continue

        # Convert to 0–1
        pred = pred.astype(np.float32) / 255.0

        # -------------------------
        # STAGE 4
        # -------------------------

        # 1. THRESHOLD
        mask = (pred > 0.5).astype(np.uint8) * 255

        # 2. MORPHOLOGICAL OPENING (remove noise)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=2)

        # 3. MORPHOLOGICAL CLOSING (fix broken lungs)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        # 4. KEEP ONLY 2 LARGEST COMPONENTS
        mask = keep_largest_components(mask, 2)

        # 5. FILL HOLES
        mask = fill_holes(mask)

        # Save
        output_path = os.path.join(output_dir, file)
        success = cv2.imwrite(output_path, mask)

        if not success:
            failed += 1
            continue

        processed += 1

    print(f"\n✅ Completed: {input_dir}")
    print(f"✔ Processed: {processed}")
    print(f"❌ Failed: {failed}")
    print("-" * 50)


# -----------------------------
# RUN ALL DATASETS
# -----------------------------
for in_sub, out_sub in DATASETS:

    input_path = os.path.join(BASE_IN, in_sub)
    output_path = os.path.join(BASE_OUT, out_sub)

    if not os.path.exists(input_path):
        print(f"❌ Missing path: {input_path}")
        continue

    process_dataset(input_path, output_path)

print("✅ Stage 4 completed for all datasets")