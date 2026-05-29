import os
import cv2
import numpy as np
from tqdm import tqdm

# -----------------------------
# UPDATED PATHS
# -----------------------------

# INPUT = OUTPUT OF U-NET SEGMENTATION
BASE_IN = r"G:\Ai-CDD\data\preprocessed\stage2"

# OUTPUT = CLEANED MASKS
BASE_OUT = r"G:\Ai-CDD\data\preprocessed\stage3"

# FAILED / WEAK MASKS
FAILED_DIR = r"G:\Ai-CDD\data\preprocessed\failed_masks"

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

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    # no foreground
    if num_labels <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]

    # fewer components than requested
    if len(areas) < num_components:
        num_components = len(areas)

    largest = np.argsort(areas)[-num_components:] + 1

    clean = np.zeros_like(mask)

    for idx in largest:
        clean[labels == idx] = 255

    return clean


def fill_holes(mask):

    h, w = mask.shape

    flood = mask.copy()

    flood_mask = np.zeros((h + 2, w + 2), np.uint8)

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
# PROCESS FUNCTION
# -----------------------------
def process_split(split):

    # -----------------------------
    # UPDATED CLASSES
    # -----------------------------
    for class_name in [
        "health",
        "sick",
        "tb"
    ]:

        input_dir = os.path.join(
            BASE_IN,
            split,
            class_name
        )

        output_dir = os.path.join(
            BASE_OUT,
            split,
            class_name
        )

        failed_output_dir = os.path.join(
            FAILED_DIR,
            split,
            class_name
        )

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(failed_output_dir, exist_ok=True)

        if not os.path.exists(input_dir):

            print(f"❌ Missing: {input_dir}")
            continue

        valid_ext = (
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp"
        )

        files = [
            f for f in os.listdir(input_dir)
            if f.lower().endswith(valid_ext)
        ]

        processed = 0
        failed = 0
        weak_masks = 0

        for file in tqdm(
            files,
            desc=f"{split}/{class_name}"
        ):

            input_path = os.path.join(
                input_dir,
                file
            )

            pred = cv2.imread(
                input_path,
                cv2.IMREAD_GRAYSCALE
            )

            if pred is None:
                failed += 1
                continue

            # -------------------------
            # NORMALIZE
            # -------------------------
            pred = pred.astype(np.float32) / 255.0

            # -------------------------
            # THRESHOLD
            # -------------------------
            mask = (
                pred > THRESHOLD
            ).astype(np.uint8) * 255

            # -------------------------
            # REMOVE SMALL NOISE
            # -------------------------
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

            # -------------------------
            # CLOSE GAPS
            # -------------------------
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

            # -------------------------
            # KEEP LARGEST COMPONENTS
            # -------------------------
            mask = keep_largest_components(
                mask,
                num_components=2
            )

            # -------------------------
            # FILL HOLES
            # -------------------------
            mask = fill_holes(mask)

            # -------------------------
            # FINAL CLEAN
            # -------------------------
            mask = cv2.GaussianBlur(
                mask,
                (3, 3),
                0
            )

            mask = (
                mask > 127
            ).astype(np.uint8) * 255

            # -------------------------
            # WEAK MASK CHECK
            # -------------------------
            if is_weak_mask(mask):

                weak_masks += 1

                # save failed case separately
                failed_path = os.path.join(
                    failed_output_dir,
                    file
                )

                cv2.imwrite(
                    failed_path,
                    mask
                )

            # -------------------------
            # SAVE CLEAN MASK
            # -------------------------
            output_path = os.path.join(
                output_dir,
                file
            )

            success = cv2.imwrite(
                output_path,
                mask
            )

            if not success:
                failed += 1
                continue

            processed += 1

        # -------------------------
        # SUMMARY
        # -------------------------
        print(f"\n✅ {split}/{class_name}")
        print(f"✔ Processed: {processed}")
        print(f"⚠️ Weak Masks: {weak_masks}")
        print(f"❌ Failed Saves: {failed}")
        print("-" * 50)


# -----------------------------
# RUN
# -----------------------------
for split in [
    "train",
    "val",
    "internal_test",
    "external_test"
]:

    process_split(split)

print("\n🎯 Stage 3 mask cleaning complete")