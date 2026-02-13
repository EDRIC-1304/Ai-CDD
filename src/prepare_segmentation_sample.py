import os
import random
import shutil
# 🔴 CHANGE THESE PATHS
SOURCE_IMAGES = r"D:\project_dataset\archive\Lung Segmentation\CXR_png"
SOURCE_MASKS  = r"D:\project_dataset\archive\Lung Segmentation\masks"

# Project destination
DEST_IMAGES = r"D:\FY_Project\Ai-CDD\data\raw\segmentation_sample\images"
DEST_MASKS  = r"D:\FY_Project\Ai-CDD\data\raw\segmentation_sample\masks"

SAMPLE_SIZE = 50

os.makedirs(DEST_IMAGES, exist_ok=True)
os.makedirs(DEST_MASKS, exist_ok=True)

valid_pairs = []

for img_name in os.listdir(SOURCE_IMAGES):
    if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    base = os.path.splitext(img_name)[0]
    mask_name = base + "_mask.png"

    if os.path.exists(os.path.join(SOURCE_MASKS, mask_name)):
        valid_pairs.append((img_name, mask_name))

print("Total valid pairs found:", len(valid_pairs))

random.seed(42)  # reproducible
selected_pairs = random.sample(valid_pairs, SAMPLE_SIZE)

print("Selected pairs:", len(selected_pairs))

for idx, (img_name, mask_name) in enumerate(selected_pairs, start=1):
    new_img_name  = f"{idx:04d}.png"
    new_mask_name = f"{idx:04d}_mask.png"

    shutil.copy(
        os.path.join(SOURCE_IMAGES, img_name),
        os.path.join(DEST_IMAGES, new_img_name)
    )

    shutil.copy(
        os.path.join(SOURCE_MASKS, mask_name),
        os.path.join(DEST_MASKS, new_mask_name)
    )

print("✅ 50 image-mask pairs copied and renamed successfully")
