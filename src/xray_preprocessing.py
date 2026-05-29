import os
import warnings

# ---------------------------------
# SUPPRESS WARNINGS
# ---------------------------------
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------
# PATHS
# ---------------------------------

# INPUT DATASET
BASE_IN = r"G:\Ai-CDD\data\raw\updated dataset"

# OUTPUT PREPROCESSED DATASET
BASE_OUT = r"G:\Ai-CDD\data\preprocessed\stage1"

# ---------------------------------
# PARAMETERS
# ---------------------------------
IMG_SIZE = 256

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

# ---------------------------------
# IMAGE PREPROCESS FUNCTION
# ---------------------------------
def preprocess_image(img):

    # Convert to grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

    # Resize
    img = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # Apply CLAHE
    img = clahe.apply(img)

    return img


# ---------------------------------
# PROCESS ENTIRE DATASET
# ---------------------------------
def process_all():

    valid_ext = (
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp"
    )

    # ---------------------------------
    # UPDATED SPLITS
    # ---------------------------------
    splits = [
        "train",
        "val",
        "internal_test",
        "external_test"
    ]

    # ---------------------------------
    # UPDATED CLASSES
    # ---------------------------------
    classes = [
        "health",
        "sick",
        "tb"
    ]

    for split in splits:

        split_in = os.path.join(
            BASE_IN,
            split
        )

        split_out = os.path.join(
            BASE_OUT,
            split
        )

        for class_name in classes:

            # ---------------------------------
            # INPUT / OUTPUT PATHS
            # ---------------------------------
            input_dir = os.path.join(
                split_in,
                class_name
            )

            output_dir = os.path.join(
                split_out,
                class_name
            )

            # ---------------------------------
            # CHECK INPUT EXISTS
            # ---------------------------------
            if not os.path.exists(input_dir):
                print(f"❌ Missing: {input_dir}")
                continue

            # ---------------------------------
            # CREATE OUTPUT DIR
            # ---------------------------------
            os.makedirs(
                output_dir,
                exist_ok=True
            )

            # ---------------------------------
            # GET VALID FILES
            # ---------------------------------
            files = [
                f for f in os.listdir(input_dir)
                if f.lower().endswith(valid_ext)
            ]

            count = 0
            failed = 0

            # ---------------------------------
            # PROCESS IMAGES
            # ---------------------------------
            progress_bar = tqdm(
                files,
                desc=f"{split}/{class_name}",
                ncols=100
            )

            for file in progress_bar:

                input_path = os.path.join(
                    input_dir,
                    file
                )

                try:

                    # ---------------------------------
                    # READ IMAGE
                    # ---------------------------------
                    img = cv2.imread(
                        input_path,
                        cv2.IMREAD_GRAYSCALE
                    )

                    if img is None:
                        failed += 1
                        continue

                    # ---------------------------------
                    # PREPROCESS
                    # ---------------------------------
                    processed_img = preprocess_image(img)

                    # ---------------------------------
                    # SAVE
                    # ---------------------------------
                    output_path = os.path.join(
                        output_dir,
                        file
                    )

                    cv2.imwrite(
                        output_path,
                        processed_img
                    )

                    count += 1

                    # ---------------------------------
                    # UPDATE PROGRESS TEXT
                    # ---------------------------------
                    progress_bar.set_postfix({
                        "Processed": count,
                        "Failed": failed
                    })

                except Exception:
                    failed += 1
                    continue

            # ---------------------------------
            # SUMMARY
            # ---------------------------------
            print(f"\n✅ {split}/{class_name}")
            print(f"✔ Processed: {count}")
            print(f"❌ Failed: {failed}")
            print("-" * 40)


# ---------------------------------
# RUN
# ---------------------------------
if __name__ == "__main__":
    process_all()