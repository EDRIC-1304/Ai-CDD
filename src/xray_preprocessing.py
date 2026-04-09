import os
import cv2
import numpy as np
from tqdm import tqdm

# -----------------------------
# PATHS
# -----------------------------
# BASE_IN = r"G:\Ai-CDD\data\raw\dataset"
# BASE_OUT = r"G:\Ai-CDD\data\preprocessed\stage1"

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
# PROCESS ENTIRE DATASET
# -----------------------------
# def process_all():
#     for split in ["train", "val", "test"]:
#         split_in = os.path.join(BASE_IN, split)
#         split_out = os.path.join(BASE_OUT, split)

#         for class_name in ["NORMAL", "TUBERCULOSIS"]:
#             input_dir = os.path.join(split_in, class_name)
#             output_dir = os.path.join(split_out, class_name)

#             if not os.path.exists(input_dir):
#                 print(f"(xray_preprocessing.py)❌ Missing: {input_dir}")
#                 continue

#             os.makedirs(output_dir, exist_ok=True)

#             files = os.listdir(input_dir)

#             count, failed = 0, 0

#             for file in tqdm(files, desc=f"{split}/{class_name}"):
#                 input_path = os.path.join(input_dir, file)

#                 img = cv2.imread(input_path)
#                 if img is None:
#                     failed += 1
#                     continue

#                 processed = preprocess_image(img)
#                 save_img = (processed * 255).astype(np.uint8)

#                 # keep same filename (no need to rename again)
#                 output_path = os.path.join(output_dir, file)

#                 if not cv2.imwrite(output_path, save_img):
#                     failed += 1
#                     continue

#                 count += 1

#             print(f"\n✅ {split}/{class_name}")
#             print(f"✔ Processed: {count}")
#             print(f"❌ Failed: {failed}")
#             print("-" * 40)

# # -----------------------------
# # RUN
# # -----------------------------
# process_all()