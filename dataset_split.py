import os
import random
import shutil
from sklearn.model_selection import train_test_split

# =========================================================
# 🔹 Paths
# =========================================================
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

IMAGE_DIR = os.path.join(
    BASE_PATH,
    "data",
    "segmentation",
    "images"
)

MASK_DIR = os.path.join(
    BASE_PATH,
    "data",
    "segmentation",
    "masks"
)

OUTPUT_DIR = os.path.join(
    BASE_PATH,
    "data",
    "segmentation"
)

# =========================================================
# 🔹 Split ratios
# =========================================================
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# =========================================================
# 🔹 Random seed
# =========================================================
SEED = 42

# =========================================================
# 🔹 Create output folders
# =========================================================
for split in ["train", "val", "test"]:

    os.makedirs(
        os.path.join(OUTPUT_DIR, split, "images"),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(OUTPUT_DIR, split, "masks"),
        exist_ok=True
    )

# =========================================================
# 🔹 Collect valid image-mask pairs
# =========================================================
all_files = []

image_files = os.listdir(IMAGE_DIR)

for file in image_files:

    image_path = os.path.join(IMAGE_DIR, file)
    mask_path = os.path.join(MASK_DIR, file)

    # -----------------------------------------------------
    # Ensure both image and mask exist
    # -----------------------------------------------------
    if not os.path.isfile(image_path):
        continue

    if not os.path.isfile(mask_path):
        print(f"⚠️ Missing mask for: {file}")
        continue

    all_files.append(file)

# =========================================================
# 🔹 Shuffle
# =========================================================
random.seed(SEED)
random.shuffle(all_files)

# =========================================================
# 🔹 Train split
# =========================================================
train_files, temp_files = train_test_split(
    all_files,
    test_size=(1 - TRAIN_RATIO),
    random_state=SEED
)

# =========================================================
# 🔹 Validation + Test split
# =========================================================
val_size_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

val_files, test_files = train_test_split(
    temp_files,
    test_size=(1 - val_size_adjusted),
    random_state=SEED
)

# =========================================================
# 🔹 Copy helper
# =========================================================
def copy_files(file_list, split):

    for file in file_list:

        src_image = os.path.join(IMAGE_DIR, file)
        src_mask = os.path.join(MASK_DIR, file)

        dst_image = os.path.join(
            OUTPUT_DIR,
            split,
            "images",
            file
        )

        dst_mask = os.path.join(
            OUTPUT_DIR,
            split,
            "masks",
            file
        )

        shutil.copy2(src_image, dst_image)
        shutil.copy2(src_mask, dst_mask)

# =========================================================
# 🔹 Copy splits
# =========================================================
print("\n📂 Copying TRAIN files...")
copy_files(train_files, "train")

print("📂 Copying VAL files...")
copy_files(val_files, "val")

print("📂 Copying TEST files...")
copy_files(test_files, "test")

# =========================================================
# 🔹 Final summary
# =========================================================
print("\n🎯 Dataset split completed successfully")

print(f"\nTRAIN: {len(train_files)}")
print(f"VAL:   {len(val_files)}")
print(f"TEST:  {len(test_files)}")

print(f"\nTOTAL: {len(all_files)}")