import os
import cv2
from tqdm import tqdm

# -----------------------------
# PATHS (CHANGE THESE)
# -----------------------------
INPUT_DIR = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/raw/train/xray-image"
OUTPUT_DIR = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/raw/train/clean_xray_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# VALID IMAGE EXTENSIONS
# -----------------------------
VALID_EXTENSIONS = [".png", ".jpg", ".jpeg"]

# -----------------------------
# CHECK IF IMAGE IS VALID
# -----------------------------
def is_valid_image(path):
    try:
        img = cv2.imread(path)
        if img is None:
            return False
        if img.size == 0:
            return False
        return True
    except:
        return False

# -----------------------------
# MAIN CLEANING PIPELINE
# -----------------------------
def clean_dataset(input_dir, output_dir):
    count = 0
    skipped = 0

    for root, _, files in os.walk(input_dir):
        for file in tqdm(files):
            ext = os.path.splitext(file)[1].lower()

            if ext not in VALID_EXTENSIONS:
                continue

            input_path = os.path.join(root, file)

            # Validate image
            if not is_valid_image(input_path):
                print(f"❌ Corrupt image removed: {input_path}")
                skipped += 1
                continue

            # Read image
            img = cv2.imread(input_path)

            # Convert to PNG format
            output_name = f"img_{count}.png"
            output_path = os.path.join(output_dir, output_name)

            cv2.imwrite(output_path, img)

            count += 1

    print("\n✅ Cleaning Completed")
    print(f"✔ Valid images saved: {count}")
    print(f"❌ Corrupt/Skipped: {skipped}")

# -----------------------------
# RUN
# -----------------------------
clean_dataset(INPUT_DIR, OUTPUT_DIR)