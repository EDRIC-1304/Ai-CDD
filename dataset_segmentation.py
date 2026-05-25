import os
from PIL import Image

# =========================================================
# 🔹 Source datasets
# =========================================================
BASE_DIR = r"G:\Ai-CDD"

DATASETS = [
    "dataset1xrayseg",
    "dataset2xrayseg",
    "dataset3xrayseg",
    "dataset4xrayseg"
]

# =========================================================
# 🔹 Output dataset
# =========================================================
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(
    BASE_PATH,
    "data",
    "segmentation"
)

OUT_IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
OUT_MASK_DIR = os.path.join(OUTPUT_DIR, "masks")

os.makedirs(OUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)

# =========================================================
# 🔹 Counters
# =========================================================
counter = 0

# =========================================================
# 🔹 Logs
# =========================================================
missing_masks = []
failed_files = []
missing_folders = []

# =========================================================
# 🔹 Naming function
# =========================================================
def get_new_name():
    global counter
    counter += 1
    return f"sample_{counter:05d}.png"

# =========================================================
# 🔹 Save image
# =========================================================
def save_png(src_path, dst_path, is_mask=False):

    try:
        with Image.open(src_path) as img:

            # -------------------------------------------------
            # Convert formats
            # -------------------------------------------------
            if is_mask:

                # Convert to grayscale
                img = img.convert("L")

            else:
                img = img.convert("RGB")

            img.save(dst_path, "PNG")

    except Exception as e:

        failed_files.append(src_path)

        print(f"❌ Failed: {src_path}")
        print(e)

# =========================================================
# 🔹 Main merge loop
# =========================================================
for dataset in DATASETS:

    dataset_path = os.path.join(BASE_DIR, dataset)

    print(f"\n📂 Processing: {dataset}")

    # -------------------------------------------------
    # Dataset folders
    # -------------------------------------------------
    image_dir = os.path.join(dataset_path, "img")
    mask_dir = os.path.join(dataset_path, "mask")

    if not os.path.exists(image_dir):
        missing_folders.append(image_dir)
        print(f"⚠️ Missing images folder: {image_dir}")
        continue

    if not os.path.exists(mask_dir):
        missing_folders.append(mask_dir)
        print(f"⚠️ Missing masks folder: {mask_dir}")
        continue

    # -------------------------------------------------
    # Process files
    # -------------------------------------------------
    for file in os.listdir(image_dir):

        image_path = os.path.join(image_dir, file)

        if not os.path.isfile(image_path):
            continue

        # -------------------------------------------------
        # Match mask
        # -------------------------------------------------
        filename_no_ext = os.path.splitext(file)[0]

        mask_found = False

        for ext in [".png", ".jpg", ".jpeg", ".bmp"]:

            possible_mask = os.path.join(
                mask_dir,
                filename_no_ext + ext
            )

            if os.path.exists(possible_mask):

                mask_path = possible_mask
                mask_found = True
                break

        if not mask_found:

            missing_masks.append(file)

            print(f"⚠️ Missing mask for: {file}")
            continue

        # -------------------------------------------------
        # Unified naming
        # -------------------------------------------------
        new_name = get_new_name()

        out_image_path = os.path.join(
            OUT_IMAGE_DIR,
            new_name
        )

        out_mask_path = os.path.join(
            OUT_MASK_DIR,
            new_name
        )

        # -------------------------------------------------
        # Save image and mask
        # -------------------------------------------------
        save_png(image_path, out_image_path, is_mask=False)

        save_png(mask_path, out_mask_path, is_mask=True)

    print(f"✅ Finished: {dataset}")

# =========================================================
# 🔹 Final Summary
# =========================================================
print("\n🎯 Segmentation datasets merged successfully")

print(f"\n📊 Total Samples: {counter}")

print("\n⚠️ Missing folders:")
for item in missing_folders:
    print(item)

print("\n⚠️ Missing masks:")
for item in missing_masks:
    print(item)

print(f"\n❌ Failed files: {len(failed_files)}")