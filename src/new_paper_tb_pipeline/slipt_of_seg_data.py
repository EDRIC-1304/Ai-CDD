import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm

# ============================================================
# INPUT PATHS
# ============================================================

IMAGE_DIR = (
    r"G:\Ai-CDD\data\preprocessed"
    r"\segmentation\images_stage1"
)

MASK_DIR = (
    r"G:\Ai-CDD\data\preprocessed"
    r"\segmentation\masks"
)

# ============================================================
# OUTPUT PATHS
# ============================================================

OUTPUT_IMAGE_DIR = (
    r"G:\Ai-CDD\data\preprocessed"
    r"\segmentation\images_stage1_split"
)

OUTPUT_MASK_DIR = (
    r"G:\Ai-CDD\data\preprocessed"
    r"\segmentation\masks_split"
)

# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

healthy_image_dir = os.path.join(
    OUTPUT_IMAGE_DIR,
    "healthy"
)

tb_image_dir = os.path.join(
    OUTPUT_IMAGE_DIR,
    "tb"
)

healthy_mask_dir = os.path.join(
    OUTPUT_MASK_DIR,
    "healthy"
)

tb_mask_dir = os.path.join(
    OUTPUT_MASK_DIR,
    "tb"
)

os.makedirs(healthy_image_dir, exist_ok=True)
os.makedirs(tb_image_dir, exist_ok=True)

os.makedirs(healthy_mask_dir, exist_ok=True)
os.makedirs(tb_mask_dir, exist_ok=True)

# ============================================================
# GET MASK FILES
# ============================================================

mask_files = sorted(os.listdir(MASK_DIR))

print(f"\nTotal Masks Found: {len(mask_files)}")

# ============================================================
# COUNTERS
# ============================================================

healthy_count = 0
tb_count = 0

# ============================================================
# PROCESS ALL MASKS
# ============================================================

for mask_name in tqdm(mask_files):

    mask_path = os.path.join(
        MASK_DIR,
        mask_name
    )

    image_path = os.path.join(
        IMAGE_DIR,
        mask_name
    )

    # ========================================================
    # CHECK IMAGE EXISTS
    # ========================================================

    if not os.path.exists(image_path):
        print(f"Missing image: {mask_name}")
        continue

    # ========================================================
    # LOAD MASK
    # ========================================================

    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        print(f"Invalid mask: {mask_name}")
        continue

    # ========================================================
    # CHECK IF MASK IS EMPTY
    # ========================================================

    mask_sum = np.sum(mask)

    # ========================================================
    # HEALTHY CASE
    # ========================================================

    if mask_sum == 0:

        shutil.copy2(
            image_path,
            os.path.join(
                healthy_image_dir,
                mask_name
            )
        )

        shutil.copy2(
            mask_path,
            os.path.join(
                healthy_mask_dir,
                mask_name
            )
        )

        healthy_count += 1

    # ========================================================
    # TB CASE
    # ========================================================

    else:

        shutil.copy2(
            image_path,
            os.path.join(
                tb_image_dir,
                mask_name
            )
        )

        shutil.copy2(
            mask_path,
            os.path.join(
                tb_mask_dir,
                mask_name
            )
        )

        tb_count += 1

# ============================================================
# FINAL STATS
# ============================================================

print("\n====================================")
print(f"Healthy Samples : {healthy_count}")
print(f"TB Samples      : {tb_count}")
print("====================================")

print("\nDataset splitting completed.")