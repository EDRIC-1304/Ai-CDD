import os
from PIL import Image

# 🔹 Source datasets
BASE_DIR = r"G:\Ai-CDD"
DATASETS = ["dataset1", "dataset2", "dataset3"]

# 🔹 Output dataset
OUTPUT_DIR = r"G:\Ai-CDD\raw\dataset"

# 🔹 Class normalization
CLASS_MAP = {
    "normal": "NORMAL",
    "NORMAL": "NORMAL",
    "Normal": "NORMAL",

    "tuberculosis": "TUBERCULOSIS",
    "Tuberculosis": "TUBERCULOSIS",
    "TUBERCULOSIS": "TUBERCULOSIS",
    "TURBERCULOSIS": "TUBERCULOSIS",  # fix typo
}

# 🔹 Counters per split + class (FIXED)
counters = {
    "train": {"NORMAL": 0, "TUBERCULOSIS": 0},
    "val": {"NORMAL": 0, "TUBERCULOSIS": 0},
    "test": {"NORMAL": 0, "TUBERCULOSIS": 0},
}

def get_new_name(split, class_name):
    counters[split][class_name] += 1
    prefix = "normal" if class_name == "NORMAL" else "tb"
    return f"{prefix}_{counters[split][class_name]:05d}.png"


def process_image(src_path, dst_path):
    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            img.save(dst_path, "PNG")
    except Exception as e:
        print(f"❌ Error: {src_path} -> {e}")


for dataset in DATASETS:
    dataset_path = os.path.join(BASE_DIR, dataset)

    for split in ["train", "val", "test"]:
        split_path = os.path.join(dataset_path, split)

        if not os.path.exists(split_path):
            continue

        for class_name in os.listdir(split_path):
            normalized_class = CLASS_MAP.get(class_name)

            if normalized_class is None:
                print(f"⚠️ Unknown class: {class_name}")
                continue

            class_path = os.path.join(split_path, class_name)

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

print("\n🎯 All datasets merged successfully into unified dataset")