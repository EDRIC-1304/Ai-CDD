import os
import cv2
import numpy as np
from tqdm import tqdm

# -----------------------------
# PATHS (CHANGE THESE)
# -----------------------------
INPUT_DIR = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/raw/train/clean_xray_images"
OUTPUT_DIR = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/raw/train/preprocessed_xray_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# PARAMETERS
# -----------------------------
IMG_SIZE = 256  # or 224

# CLAHE setup
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# -----------------------------
# PREPROCESS FUNCTION
# -----------------------------
def preprocess_image(img):
    
    # Convert to grayscale if RGB
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # CLAHE (contrast enhancement)
    img = clahe.apply(img)

    # Normalize to 0–1
    img = img.astype(np.float32) / 255.0

    return img

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def process_dataset(input_dir, output_dir):
    count = 0

    for file in tqdm(os.listdir(input_dir)):
        input_path = os.path.join(input_dir, file)

        # Read image
        img = cv2.imread(input_path)

        if img is None:
            continue

        # Preprocess
        processed = preprocess_image(img)

        # Convert back to 0–255 for saving
        save_img = (processed * 255).astype(np.uint8)

        # Save
        output_path = os.path.join(output_dir, f"img_{count}.png")
        cv2.imwrite(output_path, save_img)

        count += 1

    print(f"\n✅ Preprocessing Completed: {count} images processed")

# -----------------------------
# RUN
# -----------------------------
process_dataset(INPUT_DIR, OUTPUT_DIR)