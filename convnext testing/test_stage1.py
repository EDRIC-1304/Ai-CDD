import os
import warnings

# =========================================
# SUPPRESS WARNINGS
# =========================================

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import cv2

# =========================================
# PATHS
# =========================================

INPUT_IMAGE = (
    r"G:\Ai-CDD\convnext testing\Normal_sample1.png"
)

OUTPUT_DIR = (
    r"G:\Ai-CDD\convnext testing\outputs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

OUTPUT_IMAGE = os.path.join(
    OUTPUT_DIR,
    "stage1_preprocessed.png"
)

# =========================================
# PARAMETERS
# =========================================

IMG_SIZE = 256

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

# =========================================
# PREPROCESS FUNCTION
# =========================================

def preprocess_image(img):

    # ---------------------------------
    # CONVERT TO GRAYSCALE
    # ---------------------------------
    if len(img.shape) == 3:

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

    # ---------------------------------
    # RESIZE
    # ---------------------------------
    img = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # ---------------------------------
    # CLAHE
    # ---------------------------------
    img = clahe.apply(img)

    return img

# =========================================
# MAIN
# =========================================

def main():

    print("\nLoading image...")

    img = cv2.imread(
        INPUT_IMAGE,
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:

        raise ValueError(
            f"❌ Failed to load image:\n{INPUT_IMAGE}"
        )

    print("✅ Image loaded")

    # =====================================
    # PREPROCESS
    # =====================================

    processed = preprocess_image(img)

    # =====================================
    # SAVE OUTPUT
    # =====================================

    cv2.imwrite(
        OUTPUT_IMAGE,
        processed
    )

    print("\n✅ Stage 1 complete")

    print(f"\nSaved output:")
    print(OUTPUT_IMAGE)

# =========================================
# RUN
# =========================================

if __name__ == "__main__":
    main()