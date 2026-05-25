import os
import cv2
import numpy as np
from tqdm import tqdm

# -----------------------------
# PATH
# -----------------------------
BASE_IN = r"G:\Ai-CDD\data\raw\dataset"
SAVE_DIR = r"G:\Ai-CDD\src\debug_stage1_compare"

os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------
# PARAMS
# -----------------------------
IMG_SIZE = 256
MAX_PER_CLASS = 100

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# -----------------------------
# PREPROCESS FUNCTIONS
# -----------------------------
def preprocess_no_clahe(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    return img


def preprocess_with_clahe(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = clahe.apply(img)
    img = img.astype(np.float32) / 255.0
    return img


# -----------------------------
# PROCESS ONE CLASS
# -----------------------------
def process_class(class_name):

    count = 0

    for split in ["train", "val", "test"]:
        input_dir = os.path.join(BASE_IN, split, class_name)

        if not os.path.exists(input_dir):
            continue

        files = os.listdir(input_dir)

        for file in tqdm(files, desc=f"{class_name}"):

            if count >= MAX_PER_CLASS:
                print(f"✅ Done {MAX_PER_CLASS} images for {class_name}")
                return

            path = os.path.join(input_dir, file)
            img = cv2.imread(path)

            if img is None:
                continue

            # -------------------------
            # PROCESS
            # -------------------------
            no_clahe = preprocess_no_clahe(img)
            with_clahe = preprocess_with_clahe(img)

            # convert to uint8
            no_clahe_u8 = (no_clahe * 255).astype(np.uint8)
            with_clahe_u8 = (with_clahe * 255).astype(np.uint8)

            # -------------------------
            # COMBINE SIDE BY SIDE
            # -------------------------
            combined = np.hstack([no_clahe_u8, with_clahe_u8])

            # labels
            cv2.putText(combined, "NO CLAHE", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)

            cv2.putText(combined, "CLAHE", (IMG_SIZE + 10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)

            # -------------------------
            # SAVE
            # -------------------------
            save_name = f"{class_name}_{count}.png"
            save_path = os.path.join(SAVE_DIR, save_name)

            cv2.imwrite(save_path, combined)

            count += 1


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    process_class("NORMAL")
    process_class("TUBERCULOSIS")

    print("\n✅ DONE: 100 NORMAL + 100 TB")