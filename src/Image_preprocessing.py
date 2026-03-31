import os
import cv2
import numpy as np
from tqdm import tqdm

# -----------------------------
# PATHS (CHANGE THESE)
# -----------------------------
INPUT_DIR_TEST_NORMAL = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/processed/xray classification dataset input handled/test/Normal"
OUTPUT_DIR_TEST_NORMAL = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/processed/xray image preprcoessing/test/Normal"

INPUT_DIR_TEST_TB = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/processed/xray classification dataset input handled/test/TB"
OUTPUT_DIR_TEST_TB = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/processed/xray image preprocessing/test/TB"

INPUT_DIR_TRAIN_NORMAL = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/processed/xray classification dataset input handled/train/normal"
OUTPUT_DIR_TRAIN_NORMAL = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/processed/xray image preprocessing/train/normal"

# ✅ FIXED PATH (was wrong before)
INPUT_DIR_TRAIN_TB = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/processed/xray classification dataset input handled/train/TB"
OUTPUT_DIR_TRAIN_TB = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/processed/xray image preprocessing/train/TB"

INPUT_DIR_VAL_NORMAL = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/processed/xray classification dataset input handled/validation/Normal"
OUTPUT_DIR_VAL_NORMAL = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/processed/xray image preprocessing/validation/Normal"

INPUT_DIR_VAL_TB = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/processed/xray classification dataset input handled/validation/TB"
OUTPUT_DIR_VAL_TB = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/processed/xray image preprocessing/validation/TB"


def clean_dataset(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True) 
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
clean_dataset(INPUT_DIR_TEST_NORMAL, OUTPUT_DIR_TEST_NORMAL)
clean_dataset(INPUT_DIR_TEST_TB, OUTPUT_DIR_TEST_TB)
clean_dataset(INPUT_DIR_TRAIN_NORMAL, OUTPUT_DIR_TRAIN_NORMAL)
clean_dataset(INPUT_DIR_TRAIN_TB, OUTPUT_DIR_TRAIN_TB)
clean_dataset(INPUT_DIR_VAL_NORMAL, OUTPUT_DIR_VAL_NORMAL)
clean_dataset(INPUT_DIR_VAL_TB, OUTPUT_DIR_VAL_TB)