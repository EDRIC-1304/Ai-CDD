import os
import cv2
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 224

# -----------------------------
# INPUT / OUTPUT PATHS
# -----------------------------
INPUT_IMAGE = (
    r"G:\Ai-CDD\convnext testing\outputs\stage1_preprocessed.png"
)

INPUT_MASK = (
    r"G:\Ai-CDD\convnext testing\outputs\stage3_clean_mask.png"
)

OUTPUT_ROI = (
    r"G:\Ai-CDD\convnext testing\outputs\4_roi.png"
)

DEBUG_OUTPUT = (
    r"G:\Ai-CDD\convnext testing\outputs\4_debug_roi.png"
)

# -----------------------------
# ROI FUNCTION
# -----------------------------
def extract_lung_roi(image, mask):

    # -------------------------
    # MATCH SIZE
    # -------------------------
    if image.shape != mask.shape:

        mask = cv2.resize(
            mask,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    # -------------------------
    # BINARIZE MASK
    # -------------------------
    mask = (
        mask > 127
    ).astype(np.uint8)

    # -------------------------
    # APPLY MASK
    # -------------------------
    lung = image * mask

    # -------------------------
    # FIND BOUNDING BOX
    # -------------------------
    coords = cv2.findNonZero(mask)

    # -------------------------
    # EMPTY MASK FALLBACK
    # -------------------------
    if coords is None:

        resized = cv2.resize(
            image,
            (IMG_SIZE, IMG_SIZE)
        )

        return resized, False

    x, y, w, h = cv2.boundingRect(coords)

    # -------------------------
    # SAFETY CHECK
    # -------------------------
    if w < 20 or h < 20:

        resized = cv2.resize(
            image,
            (IMG_SIZE, IMG_SIZE)
        )

        return resized, False

    # -------------------------
    # PADDING
    # -------------------------
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

    # -------------------------
    # CROP ROI
    # -------------------------
    cropped = lung[y1:y2, x1:x2]

    # -------------------------
    # FINAL RESIZE
    # -------------------------
    cropped = cv2.resize(
        cropped,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA
    )

    return cropped, True

# -----------------------------
# LOAD IMAGE + MASK
# -----------------------------
image = cv2.imread(
    INPUT_IMAGE,
    cv2.IMREAD_GRAYSCALE
)

mask = cv2.imread(
    INPUT_MASK,
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    raise ValueError(
        "❌ Failed to load input image"
    )

if mask is None:
    raise ValueError(
        "❌ Failed to load clean mask"
    )

# -----------------------------
# EXTRACT ROI
# -----------------------------
roi, valid = extract_lung_roi(
    image,
    mask
)

if valid:
    print("✅ Valid ROI extracted")
else:
    print("⚠️ Weak/invalid mask fallback used")

# -----------------------------
# SAVE ROI
# -----------------------------
success = cv2.imwrite(
    OUTPUT_ROI,
    roi
)

if not success:
    raise ValueError(
        "❌ Failed to save ROI"
    )

# -----------------------------
# DEBUG VISUALIZATION
# -----------------------------
debug_mask = cv2.resize(
    mask,
    (image.shape[1], image.shape[0])
)

debug_mask = (
    (debug_mask > 127)
    .astype(np.uint8) * 255
)

masked = image * (
    debug_mask // 255
)

debug_roi = cv2.resize(
    roi,
    (image.shape[1], image.shape[0])
)

combined = np.hstack([
    image,
    debug_mask,
    masked,
    debug_roi
])

cv2.imwrite(
    DEBUG_OUTPUT,
    combined
)

print("\n✅ Stage 4 ROI extraction complete")

print(f"Saved ROI: {OUTPUT_ROI}")

print(f"Saved Debug: {DEBUG_OUTPUT}")