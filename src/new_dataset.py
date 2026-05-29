# =========================================================
# MULTI-CLASS CHEST X-RAY DATASET PREPARATION PIPELINE
# =========================================================
#
# CORRECTED VERSION
#
# ✔ 1493 IMAGES PER CLASS
# ✔ PERFECTLY BALANCED DATASET
# ✔ FINAL TOTAL = 4479 IMAGES
#
# CLASSES:
# 1. health
# 2. sick
# 3. tb
#
# FEATURES:
# ✔ Automatic balancing
# ✔ Oversampling if needed
# ✔ Train / Val / Internal Test / External Test
# ✔ Confusion-matrix-ready split
# ✔ Manual testing split
#
# OUTPUT:
# G:\Ai-CDD\data\raw
#
# =========================================================

import os
import random
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

# =========================================================
# CONFIG
# =========================================================

RANDOM_SEED = 42

# TARGET IMAGES PER CLASS
TARGET_PER_CLASS = 1493

# SPLIT RATIOS
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.10
EXTERNAL_RATIO = 0.05

# =========================================================
# INPUT PATHS
# =========================================================

DATASET_PATHS = {
    "health": r"G:\Ai-CDD\cleaned_dataset\health",
    "sick": r"G:\Ai-CDD\cleaned_dataset\sick",
    "tb": r"G:\Ai-CDD\cleaned_dataset\tb"
}

# =========================================================
# OUTPUT PATH
# =========================================================

OUTPUT_ROOT = Path(r"G:\Ai-CDD\data\raw")

# =========================================================
# VALID IMAGE EXTENSIONS
# =========================================================

VALID_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
]

# =========================================================
# RANDOM SEED
# =========================================================

random.seed(RANDOM_SEED)

# =========================================================
# CREATE OUTPUT DIRECTORIES
# =========================================================

SPLITS = [
    "train",
    "val",
    "internal_test",
    "external_test"
]

for split in SPLITS:

    for class_name in DATASET_PATHS.keys():

        os.makedirs(
            OUTPUT_ROOT / split / class_name,
            exist_ok=True
        )

# =========================================================
# LOAD IMAGE PATHS
# =========================================================

all_class_images = {}

print("\n=================================================")
print("LOADING DATASET")
print("=================================================")

for class_name, folder_path in DATASET_PATHS.items():

    image_paths = []

    for ext in VALID_EXTENSIONS:

        image_paths.extend(
            Path(folder_path).glob(f"*{ext}")
        )

        image_paths.extend(
            Path(folder_path).glob(f"*{ext.upper()}")
        )

    image_paths = list(set(image_paths))

    random.shuffle(image_paths)

    if len(image_paths) == 0:
        raise ValueError(
            f"No images found in {folder_path}"
        )

    print(f"{class_name}: {len(image_paths)} images found")

    all_class_images[class_name] = image_paths

# =========================================================
# BALANCE DATASET
# =========================================================

balanced_dataset = {}

print("\n=================================================")
print("BALANCING DATASET")
print("=================================================")

for class_name, images in all_class_images.items():

    print(f"\nCLASS: {class_name}")
    print(f"Available Images : {len(images)}")
    print(f"Target Images    : {TARGET_PER_CLASS}")

    # -----------------------------------------------------
    # CASE 1: ENOUGH IMAGES
    # -----------------------------------------------------

    if len(images) >= TARGET_PER_CLASS:

        selected_images = random.sample(
            images,
            TARGET_PER_CLASS
        )

    # -----------------------------------------------------
    # CASE 2: NOT ENOUGH IMAGES
    # OVERSAMPLING
    # -----------------------------------------------------

    else:

        needed = TARGET_PER_CLASS - len(images)

        print(f"Oversampling Needed: {needed}")

        extra_images = random.choices(
            images,
            k=needed
        )

        selected_images = images + extra_images

    balanced_dataset[class_name] = selected_images

    print(f"Final Count: {len(selected_images)}")

# =========================================================
# SPLIT FUNCTION
# =========================================================

def split_dataset(images):

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    train_imgs, temp_imgs = train_test_split(
        images,
        test_size=(1 - TRAIN_RATIO),
        random_state=RANDOM_SEED,
        shuffle=True
    )

    # -----------------------------------------------------
    # REMAINING
    # -----------------------------------------------------

    remaining_ratio = (
        VAL_RATIO +
        TEST_RATIO +
        EXTERNAL_RATIO
    )

    test_external_relative = (
        TEST_RATIO + EXTERNAL_RATIO
    ) / remaining_ratio

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    val_imgs, temp2_imgs = train_test_split(
        temp_imgs,
        test_size=test_external_relative,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    # -----------------------------------------------------
    # INTERNAL + EXTERNAL
    # -----------------------------------------------------

    internal_relative = (
        TEST_RATIO /
        (TEST_RATIO + EXTERNAL_RATIO)
    )

    internal_test_imgs, external_test_imgs = train_test_split(
        temp2_imgs,
        test_size=(1 - internal_relative),
        random_state=RANDOM_SEED,
        shuffle=True
    )

    return (
        train_imgs,
        val_imgs,
        internal_test_imgs,
        external_test_imgs
    )

# =========================================================
# CREATE SPLITS + COPY FILES
# =========================================================

print("\n=================================================")
print("CREATING SPLITS")
print("=================================================")

split_statistics = {}

for class_name, images in balanced_dataset.items():

    (
        train_imgs,
        val_imgs,
        internal_test_imgs,
        external_test_imgs
    ) = split_dataset(images)

    split_statistics[class_name] = {
        "train": len(train_imgs),
        "val": len(val_imgs),
        "internal_test": len(internal_test_imgs),
        "external_test": len(external_test_imgs)
    }

    split_mapping = {
        "train": train_imgs,
        "val": val_imgs,
        "internal_test": internal_test_imgs,
        "external_test": external_test_imgs
    }

    # -----------------------------------------------------
    # COPY FILES
    # -----------------------------------------------------

    for split_name, split_images in split_mapping.items():

        destination_folder = (
            OUTPUT_ROOT /
            split_name /
            class_name
        )

        for idx, img_path in enumerate(split_images):

            extension = img_path.suffix

            new_filename = (
                f"{class_name}_{idx:05d}{extension}"
            )

            destination_path = (
                destination_folder /
                new_filename
            )

            shutil.copy2(
                img_path,
                destination_path
            )

# =========================================================
# FINAL STATISTICS
# =========================================================

print("\n=================================================")
print("FINAL DATASET STATISTICS")
print("=================================================")

grand_total = 0

for class_name, stats in split_statistics.items():

    print(f"\nCLASS: {class_name}")

    class_total = 0

    for split_name, count in stats.items():

        print(f"{split_name:<15}: {count}")

        class_total += count

    print(f"TOTAL          : {class_total}")

    grand_total += class_total

print("\n=================================================")
print(f"FINAL TOTAL DATASET SIZE: {grand_total}")
print("=================================================")

# =========================================================
# SAVE SUMMARY FILE
# =========================================================

summary_path = OUTPUT_ROOT / "dataset_summary.txt"

with open(summary_path, "w") as f:

    f.write("MULTI-CLASS DATASET SUMMARY\n")
    f.write("====================================\n\n")

    for class_name, stats in split_statistics.items():

        f.write(f"CLASS: {class_name}\n")

        class_total = 0

        for split_name, count in stats.items():

            f.write(f"{split_name:<15}: {count}\n")

            class_total += count

        f.write(f"TOTAL          : {class_total}\n\n")

    f.write(
        f"FINAL TOTAL DATASET SIZE: {grand_total}\n"
    )

print(f"\nSummary saved at:\n{summary_path}")

print("\n=================================================")
print("DATASET PREPARATION COMPLETED")
print("=================================================")