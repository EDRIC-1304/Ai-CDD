import os
import cv2
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================================================
# PATHS
# ============================================================

CSV_PATH = (
    r"G:\Ai-CDD\data\segmentation"
    r"\TBX11K_segmentation dataset"
    r"\tbx11k-simplified\data.csv"
)

IMAGE_DIR = (
    r"G:\Ai-CDD\data\preprocessed"
    r"\segmentation\images_stage1"
)

MASK_DIR = (
    r"G:\Ai-CDD\data\preprocessed"
    r"\segmentation\masks"
)

os.makedirs(MASK_DIR, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

ORIGINAL_SIZE = 512
TARGET_SIZE = 224

SCALE = TARGET_SIZE / ORIGINAL_SIZE

# ============================================================
# LOAD CSV
# ============================================================

df = pd.read_csv(CSV_PATH)

print(f"\nTotal CSV Rows: {len(df)}")

# ============================================================
# GROUP ALL ROWS
# ============================================================

grouped = df.groupby("fname")

print(f"Total Images In CSV: {len(grouped)}")

# ============================================================
# GENERATE MASKS
# ============================================================

valid_masks = 0
empty_masks = 0

for image_name, rows in tqdm(grouped):

    image_path = os.path.join(
        IMAGE_DIR,
        image_name
    )

    # ========================================================
    # CHECK IMAGE EXISTS
    # ========================================================

    if not os.path.exists(image_path):
        continue

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        continue

    h, w = image.shape

    # ========================================================
    # CREATE EMPTY MASK
    # ========================================================

    mask = np.zeros((h, w), dtype=np.uint8)

    # ========================================================
    # HEALTHY IMAGE
    # ========================================================

    first_bbox = rows.iloc[0]["bbox"]

    if first_bbox == "none":

        save_path = os.path.join(
            MASK_DIR,
            image_name
        )

        cv2.imwrite(save_path, mask)

        empty_masks += 1

        continue

    # ========================================================
    # TB IMAGE WITH LESIONS
    # ========================================================

    for _, row in rows.iterrows():

        try:

            bbox = ast.literal_eval(row["bbox"])

        except:
            continue

        # ====================================================
        # EXTRACT COORDINATES
        # ====================================================

        xmin = int(
            round(bbox["xmin"] * SCALE)
        )

        ymin = int(
            round(bbox["ymin"] * SCALE)
        )

        box_width = max(
            1,
            int(round(bbox["width"] * SCALE))
        )

        box_height = max(
            1,
            int(round(bbox["height"] * SCALE))
        )

        xmax = xmin + box_width
        ymax = ymin + box_height

        # ====================================================
        # SAFETY CLIPPING
        # ====================================================

        xmin = max(0, xmin)
        ymin = max(0, ymin)

        xmax = min(w - 1, xmax)
        ymax = min(h - 1, ymax)

        # ====================================================
        # SKIP INVALID BOXES
        # ====================================================

        if xmax <= xmin or ymax <= ymin:
            continue

        # ====================================================
        # DRAW LESION REGION
        # ====================================================

        cv2.rectangle(
            mask,
            (xmin, ymin),
            (xmax, ymax),
            255,
            -1
        )

    # ========================================================
    # CHECK MASK CONTENT
    # ========================================================

    mask_sum = np.sum(mask)

    if mask_sum > 0:
        valid_masks += 1
    else:
        empty_masks += 1

    print(
        f"{image_name} | "
        f"Unique: {np.unique(mask)} | "
        f"Pixel Sum: {mask_sum}"
    )

    # ========================================================
    # SAVE MASK
    # ========================================================

    save_path = os.path.join(
        MASK_DIR,
        image_name
    )

    cv2.imwrite(save_path, mask)

# ============================================================
# FINAL STATS
# ============================================================

print("\n===================================")
print(f"Valid Masks : {valid_masks}")
print(f"Empty Masks : {empty_masks}")
print("===================================")

print("\nMask generation completed successfully.")