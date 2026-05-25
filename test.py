import os
import cv2
import torch
import torch.nn as nn
import numpy as np

from train_convnext import ConvNextTB
from src.xray_unet_preprocessing import UNet

# =========================================
# CONFIG
# =========================================

IMAGE_PATH = r"G:\Ai-CDD\image4.png"

SEG_MODEL_PATH = (
    r"G:\Ai-CDD\segmentation_checkpoints\best_unet.pth"
)

CLS_MODEL_PATH = (
    r"G:\Ai-CDD\convnext_checkpoints\best_convnext.pth"
)

OUTPUT_DIR = r"G:\Ai-CDD\debug_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMG_SIZE = 224

# =========================================
# LOAD SEGMENTATION MODEL
# =========================================

seg_model = UNet().to(DEVICE)

checkpoint = torch.load(
    SEG_MODEL_PATH,
    map_location=DEVICE
)

seg_model.load_state_dict(
    checkpoint["model_state_dict"]
)

seg_model.eval()

print("✅ Segmentation model loaded")

# =========================================
# LOAD CLASSIFICATION MODEL
# =========================================

cls_model = ConvNextTB().to(DEVICE)

cls_model.load_state_dict(
    torch.load(
        CLS_MODEL_PATH,
        map_location=DEVICE
    )
)

cls_model.eval()

print("✅ ConvNeXt model loaded")

# =========================================
# CLAHE
# =========================================

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

# =========================================
# PREPROCESS
# =========================================

def preprocess_image(img):

    if len(img.shape) == 3:

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

    img = cv2.resize(
        img,
        (256, 256),
        interpolation=cv2.INTER_AREA
    )

    img = clahe.apply(img)

    return img

# =========================================
# SEGMENTATION
# =========================================

def segment_lungs(img):

    x = img.astype(np.float32) / 255.0

    x = torch.tensor(
        x,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        pred = seg_model(x)

        pred = torch.sigmoid(pred)

        pred = pred.squeeze().cpu().numpy()

    mask = (pred > 0.5).astype(np.uint8)

    mask = mask * 255

    return mask

# =========================================
# MASK CLEANING
# =========================================

def clean_mask(mask):

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask

# 

# =========================================
# ROI EXTRACTION
# =========================================

def extract_roi(image, mask):

    coords = cv2.findNonZero(mask)

    if coords is None:

        return cv2.resize(
            image,
            (IMG_SIZE, IMG_SIZE)
        )

    x, y, w, h = cv2.boundingRect(coords)

    pad = 10

    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)

    x2 = min(
        x + w + pad,
        image.shape[1]
    )

    y2 = min(
        y + h + pad,
        image.shape[0]
    )

    roi = image[y1:y2, x1:x2]

    roi = cv2.resize(
        roi,
        (IMG_SIZE, IMG_SIZE)
    )

    return roi

# =========================================
# LOAD IMAGE
# =========================================

img = cv2.imread(
    IMAGE_PATH,
    cv2.IMREAD_GRAYSCALE
)

if img is None:
    raise ValueError("❌ Failed to load image")

# =========================================
# STAGE 1
# =========================================

processed = preprocess_image(img)

# =========================================
# STAGE 2
# =========================================

mask = segment_lungs(processed)

# =========================================
# STAGE 3
# =========================================

mask = clean_mask(mask)

# =========================================
# STAGE 4
# =========================================

roi = extract_roi(
    processed,
    mask
)

# =========================================
# SAVE DEBUG OUTPUTS
# =========================================

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "1_processed.png"),
    processed
)

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "2_mask.png"),
    mask
)

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "3_roi.png"),
    roi
)

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "4_roi.png"),
    roi
)

# =========================================
# NORMALIZATION
# =========================================

roi = roi.astype(np.float32) / 255.0

roi = np.stack(
    [roi, roi, roi],
    axis=-1
)

roi = np.transpose(
    roi,
    (2, 0, 1)
)

tensor = torch.tensor(
    roi,
    dtype=torch.float32
).unsqueeze(0).to(DEVICE)

# =========================================
# CLASSIFICATION
# =========================================

with torch.no_grad():

    output = cls_model(tensor)

    probs = torch.softmax(
        output,
        dim=1
    )

    confidence, pred = torch.max(
        probs,
        dim=1
    )

pred_class = pred.item()

confidence = confidence.item()

# =========================================
# LABELS
# =========================================

classes = {
    0: "NORMAL",
    1: "TUBERCULOSIS"
}

print("\n========== RESULT ==========")

print(f"Prediction : {classes[pred_class]}")

print(f"Confidence : {confidence:.4f}")

print("============================")

print("\n✅ Debug outputs saved")