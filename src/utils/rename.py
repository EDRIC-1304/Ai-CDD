import os

IMAGE_DIR = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/raw/train/preprocessed_xray_images"
MASK_DIR  = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/raw/train/xray-mask"

mask_files = sorted(os.listdir(MASK_DIR))

for i, file in enumerate(mask_files):
    old_path = os.path.join(MASK_DIR, file)
    new_name = f"mask_{i}.png"
    new_path = os.path.join(MASK_DIR, new_name)

    os.rename(old_path, new_path)

print("All masks renamed successfully")
print(sorted(os.listdir(IMAGE_DIR))[:10])
print(sorted(os.listdir(MASK_DIR))[:10])