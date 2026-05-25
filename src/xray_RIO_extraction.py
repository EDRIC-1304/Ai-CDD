import os
import cv2
import numpy as np
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 224

# -----------------------------
# PATHS
# -----------------------------
BASE_IMG = r"G:\Ai-CDD\data\preprocessed\stage1"
BASE_MASK = r"G:\Ai-CDD\data\preprocessed\stage3"
BASE_OUT = r"G:\Ai-CDD\data\preprocessed\stage4"

# -----------------------------
# DEBUG
# -----------------------------
DEBUG_DIR = r"G:\Ai-CDD\src\debug_stage4"
os.makedirs(DEBUG_DIR, exist_ok=True)

# -----------------------------
# VALID EXTENSIONS
# -----------------------------
VALID_EXT = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp"
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
    mask = (mask > 127).astype(np.uint8)

    # -------------------------
    # APPLY MASK
    # -------------------------
    lung = image * mask

    # -------------------------
    # FIND BOUNDING BOX
    # -------------------------
    coords = cv2.findNonZero(mask)

    # fallback if mask empty
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
    # ADD SMALL PADDING
    # -------------------------
    pad = 10

    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)

    x2 = min(x + w + pad, image.shape[1])
    y2 = min(y + h + pad, image.shape[0])

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
# PROCESS
# -----------------------------
def process_split(split):

    for class_name in ["NORMAL", "TUBERCULOSIS"]:

        img_dir = os.path.join(
            BASE_IMG,
            split,
            class_name
        )

        mask_dir = os.path.join(
            BASE_MASK,
            split,
            class_name
        )

        out_dir = os.path.join(
            BASE_OUT,
            split,
            class_name
        )

        if not os.path.exists(img_dir):
            print(f"❌ Missing image dir: {img_dir}")
            continue

        if not os.path.exists(mask_dir):
            print(f"❌ Missing mask dir: {mask_dir}")
            continue

        os.makedirs(out_dir, exist_ok=True)

        files = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith(VALID_EXT)
        ]

        processed = 0
        skipped = 0
        weak_masks = 0

        for i, file in enumerate(
            tqdm(files, desc=f"{split}/{class_name}")
        ):

            img_path = os.path.join(img_dir, file)
            mask_path = os.path.join(mask_dir, file)

            # -------------------------
            # CHECK MASK EXISTS
            # -------------------------
            if not os.path.exists(mask_path):
                skipped += 1
                continue

            # -------------------------
            # READ IMAGE
            # -------------------------
            image = cv2.imread(
                img_path,
                cv2.IMREAD_GRAYSCALE
            )

            mask = cv2.imread(
                mask_path,
                cv2.IMREAD_GRAYSCALE
            )

            if image is None or mask is None:
                skipped += 1
                continue

            # -------------------------
            # EXTRACT ROI
            # -------------------------
            roi, valid = extract_lung_roi(
                image,
                mask
            )

            if not valid:
                weak_masks += 1

            # -------------------------
            # SAVE ROI
            # -------------------------
            out_path = os.path.join(
                out_dir,
                file
            )

            success = cv2.imwrite(
                out_path,
                roi
            )

            if not success:
                skipped += 1
                continue

            processed += 1

            # -------------------------
            # DEBUG SAVE
            # -------------------------
            if i < 20:

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

                debug_path = os.path.join(
                    DEBUG_DIR,
                    f"{split}_{class_name}_{i}.png"
                )

                cv2.imwrite(
                    debug_path,
                    combined
                )

        # -------------------------
        # SUMMARY
        # -------------------------
        print(f"\n✅ {split}/{class_name}")
        print(f"✔ ROI Saved: {processed}")
        print(f"⚠️ Weak Masks: {weak_masks}")
        print(f"❌ Skipped: {skipped}")
        print("-" * 50)


# -----------------------------
# RUN
# -----------------------------
for split in ["train", "val", "test"]:
    process_split(split)

print("\n🎯 Stage 4 ROI extraction complete")












