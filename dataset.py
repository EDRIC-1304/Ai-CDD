import os
from PIL import Image

# =========================================================
# 🔹 Source datasets
# =========================================================
BASE_DIR = r"G:\Ai-CDD"
DATASETS = ["dataset1", "dataset2", "dataset3"]

# =========================================================
# 🔹 Output dataset
# =========================================================
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_PATH, "data", "raw", "dataset")

# =========================================================
# 🔹 Class normalization
# =========================================================
CLASS_MAP = {
    "normal": "NORMAL",
    "NORMAL": "NORMAL",
    "Normal": "NORMAL",

    "tuberculosis": "TUBERCULOSIS",
    "Tuberculosis": "TUBERCULOSIS",
    "TUBERCULOSIS": "TUBERCULOSIS",
    "TURBERCULOSIS": "TUBERCULOSIS",  # fix typo
}

# =========================================================
# 🔹 Counters per split + class
# =========================================================
counters = {
    "train": {"NORMAL": 0, "TUBERCULOSIS": 0},
    "val": {"NORMAL": 0, "TUBERCULOSIS": 0},
    "test": {"NORMAL": 0, "TUBERCULOSIS": 0},
}

# =========================================================
# 🔹 Logging containers
# =========================================================
missing_folders = []
unknown_classes = []
failed_images = []

# =========================================================
# 🔹 Naming function
# =========================================================
def get_new_name(split, class_name):
    counters[split][class_name] += 1
    prefix = "normal" if class_name == "NORMAL" else "tb"
    return f"{prefix}_{counters[split][class_name]:05d}.png"


# =========================================================
# 🔹 Image processing
# =========================================================
def process_image(src_path, dst_path):
    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            img.save(dst_path, "PNG")
    except Exception as e:
        failed_images.append(src_path)
        print(f"❌ Error: {src_path} -> {e}")


# =========================================================
# 🔹 Main merging loop
# =========================================================
for dataset in DATASETS:
    dataset_path = os.path.join(BASE_DIR, dataset)

    for split in ["train", "val", "test"]:
        split_path = os.path.join(dataset_path, split)

        if not os.path.exists(split_path):
            print(f"⚠️ Missing split skipped: {split_path}")
            continue

        for class_name in os.listdir(split_path):

            normalized_class = CLASS_MAP.get(class_name)

            # 🔹 Unknown class
            if normalized_class is None:
                unknown_classes.append(class_name)
                print(f"⚠️ Unknown class skipped: {class_name}")
                continue

            class_path = os.path.join(split_path, class_name)

            # 🔹 Missing folder fix (IMPORTANT)
            if not os.path.exists(class_path):
                missing_folders.append(class_path)
                print(f"⚠️ Missing folder skipped: {class_path}")
                continue

            output_class_path = os.path.join(
                OUTPUT_DIR, split, normalized_class
            )
            os.makedirs(output_class_path, exist_ok=True)

            for file in os.listdir(class_path):
                src_file = os.path.join(class_path, file)

                if not os.path.isfile(src_file):
                    continue

                new_name = get_new_name(split, normalized_class)
                dst_file = os.path.join(output_class_path, new_name)

                process_image(src_file, dst_file)

    print(f"✅ Finished merging: {dataset}")

# =========================================================
# 🔹 Final Summary (VERY IMPORTANT FOR PAPER)
# =========================================================
print("\n🎯 All datasets merged successfully into unified dataset")

print("\n📊 Dataset Statistics:")
for split in counters:
    print(f"\n{split.upper()}:")
    for cls in counters[split]:
        print(f"  {cls}: {counters[split][cls]} images")

print("\n⚠️ Missing folders:")
for m in missing_folders:
    print(m)

print("\n⚠️ Unknown classes:")
for u in set(unknown_classes):
    print(u)

print(f"\n❌ Failed images: {len(failed_images)}")