import os
import cv2
import numpy as np

# -----------------------------
# PATHS
# -----------------------------
INPUT_MASK = (
    r"G:\Ai-CDD\convnext testing\outputs\stage2_mask.png"
)

OUTPUT_MASK = (
    r"G:\Ai-CDD\convnext testing\outputs\stage3_clean_mask.png"
)

FAILED_DIR = (
    r"G:\Ai-CDD\convnext testing\failed_masks"
)

os.makedirs(FAILED_DIR, exist_ok=True)

# -----------------------------
# CONFIG
# -----------------------------
THRESHOLD = 0.3

MIN_WHITE_PIXELS = 500

# -----------------------------
# HELPERS
# -----------------------------
def keep_largest_components(mask, num_components=2):

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    # no foreground
    if num_labels <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]

    # fewer components than requested
    if len(areas) < num_components:
        num_components = len(areas)

    largest = (
        np.argsort(areas)[-num_components:] + 1
    )

    clean = np.zeros_like(mask)

    for idx in largest:

        clean[labels == idx] = 255

    return clean


def fill_holes(mask):

    h, w = mask.shape

    flood = mask.copy()

    flood_mask = np.zeros(
        (h + 2, w + 2),
        np.uint8
    )

    cv2.floodFill(
        flood,
        flood_mask,
        (0, 0),
        255
    )

    flood_inv = cv2.bitwise_not(flood)

    return mask | flood_inv


def is_weak_mask(mask):

    white_pixels = np.sum(mask > 0)

    return white_pixels < MIN_WHITE_PIXELS

# -----------------------------
# LOAD MASK
# -----------------------------
pred = cv2.imread(
    INPUT_MASK,
    cv2.IMREAD_GRAYSCALE
)

if pred is None:
    raise ValueError(
        "❌ Failed to load segmentation mask"
    )

# -----------------------------
# NORMALIZE
# -----------------------------
pred = pred.astype(np.float32) / 255.0

# -----------------------------
# THRESHOLD
# -----------------------------
mask = (
    pred > THRESHOLD
).astype(np.uint8) * 255

# -----------------------------
# REMOVE SMALL NOISE
# -----------------------------
kernel_open = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (3, 3)
)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_OPEN,
    kernel_open,
    iterations=1
)

# -----------------------------
# CLOSE GAPS
# -----------------------------
kernel_close = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (7, 7)
)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_CLOSE,
    kernel_close,
    iterations=1
)

# -----------------------------
# KEEP LARGEST COMPONENTS
# -----------------------------
mask = keep_largest_components(
    mask,
    num_components=2
)

# -----------------------------
# FILL HOLES
# -----------------------------
mask = fill_holes(mask)

# -----------------------------
# FINAL CLEAN
# -----------------------------
mask = cv2.GaussianBlur(
    mask,
    (3, 3),
    0
)

mask = (
    mask > 127
).astype(np.uint8) * 255

# -----------------------------
# WEAK MASK CHECK
# -----------------------------
if is_weak_mask(mask):

    failed_path = os.path.join(
        FAILED_DIR,
        "weak_mask.png"
    )

    cv2.imwrite(
        failed_path,
        mask
    )

    print("⚠️ Weak mask detected")

# -----------------------------
# SAVE CLEAN MASK
# -----------------------------
success = cv2.imwrite(
    OUTPUT_MASK,
    mask
)

if not success:
    raise ValueError(
        "❌ Failed to save clean mask"
    )

print("\n✅ Stage 3 mask cleaning complete")

print(f"Saved: {OUTPUT_MASK}")