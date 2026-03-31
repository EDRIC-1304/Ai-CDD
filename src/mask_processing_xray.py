import os
import cv2
import numpy as np
import torch
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 256

MODEL_PATH = "checkpoints/best.pth"
INPUT_DIR = "data/raw/train/preprocessed_xray_images"
OUTPUT_MASK_DIR = "data/processed/masks_clean"

os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD MODEL
# -----------------------------
from Unet_xray import UNet  # import your model class

model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -----------------------------
# HELPERS
# -----------------------------
def keep_largest_components(mask, num_components=2):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = np.argsort(areas)[-num_components:] + 1

    clean = np.zeros_like(mask)

    for idx in largest:
        clean[labels == idx] = 255

    return clean


def fill_holes(mask):
    h, w = mask.shape
    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)

    return mask | flood_inv


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def process_image(image_path):
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        return None

    h, w = original.shape

    # Resize for model
    img = cv2.resize(original, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)

    img_tensor = torch.tensor(img, dtype=torch.float32).to(DEVICE)

    # -------------------------
    # STAGE 3 OUTPUT (probability mask)
    # -------------------------
    with torch.no_grad():
        pred = model(img_tensor)[0, 0].cpu().numpy()

    pred = cv2.resize(pred, (w, h))

    # -------------------------
    # STAGE 4 STARTS HERE
    # -------------------------

    # 1. THRESHOLD (CRITICAL)
    mask = (pred > 0.5).astype(np.uint8) * 255

    # 2. MORPHOLOGICAL OPENING (remove noise)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=2)

    # 3. MORPHOLOGICAL CLOSING (fix broken lungs)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # 4. KEEP ONLY 2 LARGEST COMPONENTS
    mask = keep_largest_components(mask, 2)

    # 5. FILL HOLES
    mask = fill_holes(mask)

    return mask


# -----------------------------
# RUN ON DATASET
# -----------------------------
image_files = os.listdir(INPUT_DIR)

for file in tqdm(image_files):
    path = os.path.join(INPUT_DIR, file)

    mask = process_image(path)

    if mask is None:
        continue

    save_path = os.path.join(OUTPUT_MASK_DIR, file)
    cv2.imwrite(save_path, mask)

print("✅ Stage 4 completed: Clean masks saved")