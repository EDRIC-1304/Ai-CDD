import os
import cv2
import numpy as np
from tqdm import tqdm

# -----------------------------
# PATHS (FIXED + CONSISTENT)
# -----------------------------
BASE_IN = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/processed/xray classification dataset input handled"
BASE_OUT = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/processed/xray_image_preprocessing"

DATASETS = [
    ("test/Normal", "test/Normal"),
    ("test/TB", "test/TB"),
    ("train/normal", "train/normal"),
    ("train/TB", "train/TB"),
    ("validation/Normal", "validation/Normal"),
    ("validation/TB", "validation/TB"),
]

# -----------------------------
# PARAMETERS
# -----------------------------
IMG_SIZE = 256
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# -----------------------------
# PREPROCESS FUNCTION
# -----------------------------
def preprocess_image(img):

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = clahe.apply(img)
    img = img.astype(np.float32) / 255.0

    return img

# -----------------------------
# MAIN PROCESS FUNCTION
# -----------------------------
def process_dataset(input_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    count = 0
    failed = 0

    files = os.listdir(input_dir)

    for file in tqdm(files):
        input_path = os.path.join(input_dir, file)

        img = cv2.imread(input_path)
        if img is None:
            failed += 1
            continue

        processed = preprocess_image(img)

        save_img = (processed * 255).astype(np.uint8)

        # ✅ UNIQUE NAME (NO OVERWRITE)
        filename = f"{count}_{os.path.splitext(file)[0]}.png"
        output_path = os.path.join(output_dir, filename)

        success = cv2.imwrite(output_path, save_img)

        # VERIFY WRITE
        check = cv2.imread(output_path)

        if not success or check is None:
            failed += 1
            continue

        count += 1

    print(f"\n✅ Completed: {input_dir}")
    print(f"✔ Processed: {count}")
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