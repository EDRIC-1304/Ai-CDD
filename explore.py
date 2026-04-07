import os
from collections import defaultdict

BASE_DIR = r"G:\Ai-CDD"  # change this

DATASETS = ["dataset", "dataset1", "dataset2", "dataset3"]

def explore_dataset(base_path):
    summary = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for dataset in DATASETS:
        dataset_path = os.path.join(base_path, dataset)

        if not os.path.exists(dataset_path):
            print(f"⚠️ Missing: {dataset}")
            continue

        for split in os.listdir(dataset_path):
            split_path = os.path.join(dataset_path, split)
            if not os.path.isdir(split_path):
                continue

            for class_name in os.listdir(split_path):
                class_path = os.path.join(split_path, class_name)
                if not os.path.isdir(class_path):
                    continue

                files = [
                    f for f in os.listdir(class_path)
                    if os.path.isfile(os.path.join(class_path, f))
                ]

                summary[dataset][split][class_name] = len(files)

    return summary


def print_summary(summary):
    for dataset, splits in summary.items():
        print(f"\n📁 {dataset}")
        for split, classes in splits.items():
            print(f"  └── {split}")
            total_split = 0
            for cls, count in classes.items():
                print(f"      ├── {cls}: {count}")
                total_split += count
            print(f"      └── Total: {total_split}")


summary = explore_dataset(BASE_DIR)
print_summary(summary)