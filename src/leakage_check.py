import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import imagehash
import hashlib
from collections import defaultdict
import argparse
import pickle
import warnings
import csv

# ============================================================
# SUPPRESS WARNINGS
# ============================================================

warnings.filterwarnings("ignore")

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    type=str,
    required=True,
    choices=[
        "index",
        "merge"
    ]
)

parser.add_argument(
    "--start",
    type=int,
    default=0
)

parser.add_argument(
    "--end",
    type=int,
    default=None
)

args = parser.parse_args()

# ============================================================
# CONFIG
# ============================================================

DATASET_PATHS = {
    "train": r"G:\Ai-CDD\data\raw\dataset\train",
    "test": r"G:\Ai-CDD\data\raw\dataset\test",
    "val": r"G:\Ai-CDD\data\raw\dataset\val"
}

IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff"
)

PHASH_THRESHOLD = 5

COMPARE_SIZE = 256

INDEX_DIR = "batch_indexes"

os.makedirs(INDEX_DIR, exist_ok=True)

# ============================================================
# HELPERS
# ============================================================

def get_file_hash(path):

    sha256 = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(8192)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()

# ============================================================

def get_phash(path):

    try:

        image = Image.open(path).convert("L")

        return str(imagehash.phash(image))

    except:

        return None

# ============================================================

def extract_patient_id(filename):

    filename = os.path.splitext(filename)[0]

    filename = filename.lower()

    separators = [
        "_",
        "-",
        " "
    ]

    for sep in separators:

        filename = filename.replace(sep, " ")

    tokens = filename.split()

    if len(tokens) == 0:
        return None

    return tokens[0]

# ============================================================
# COLLECT ALL IMAGES
# ============================================================

all_images = []

print("\nCollecting dataset images...")

for split_name, split_path in DATASET_PATHS.items():

    for root, dirs, files in os.walk(split_path):

        for file in files:

            if not file.lower().endswith(IMAGE_EXTENSIONS):
                continue

            full_path = os.path.join(root, file)

            all_images.append({
                "split": split_name,
                "path": full_path,
                "name": file
            })

print(f"\nTotal Images Found: {len(all_images)}")

# ============================================================
# INDEX MODE
# ============================================================

if args.mode == "index":

    start = args.start

    end = args.end

    if end is None:
        end = len(all_images)

    batch_images = all_images[start:end]

    print(f"\nProcessing Batch: {start} -> {end}")
    print(f"Batch Size: {len(batch_images)}")

    batch_data = []

    for item in tqdm(batch_images):

        path = item["path"]

        try:

            file_hash = get_file_hash(path)

            phash = get_phash(path)

            patient_id = extract_patient_id(
                item["name"]
            )

            batch_data.append({

                "split": item["split"],

                "path": path,

                "name": item["name"],

                "patient_id": patient_id,

                "file_hash": file_hash,

                "phash": phash
            })

        except Exception as e:

            print(f"\nERROR: {path}")
            print(e)

    output_path = os.path.join(
        INDEX_DIR,
        f"index_{start}_{end}.pkl"
    )

    with open(output_path, "wb") as f:

        pickle.dump(batch_data, f)

    print("\n====================================")
    print("BATCH INDEXING COMPLETED")
    print("====================================")

    print(f"\nSaved: {output_path}")

# ============================================================
# MERGE MODE
# ============================================================

elif args.mode == "merge":

    print("\nLoading batch indexes...")

    merged_data = []

    for file in os.listdir(INDEX_DIR):

        if not file.endswith(".pkl"):
            continue

        full_path = os.path.join(
            INDEX_DIR,
            file
        )

        print(f"Loading: {file}")

        with open(full_path, "rb") as f:

            data = pickle.load(f)

            merged_data.extend(data)

    print(f"\nTotal Indexed Images: {len(merged_data)}")

    # ========================================================
    # STORAGE
    # ========================================================

    exact_duplicates = []

    perceptual_duplicates = []

    same_patient_samples = []

    # ========================================================
    # EXACT DUPLICATES
    # ========================================================

    print("\nChecking exact duplicates...")

    hash_map = defaultdict(list)

    for item in merged_data:

        hash_map[item["file_hash"]].append(item)

    for file_hash, items in hash_map.items():

        if len(items) > 1:

            splits = set([
                x["split"]
                for x in items
            ])

            if len(splits) > 1:

                exact_duplicates.append(items)

    # ========================================================
    # SAME PATIENT
    # ========================================================

    print("\nChecking same-patient overlap...")

    patient_map = defaultdict(list)

    for item in merged_data:

        patient_id = item["patient_id"]

        if patient_id is None:
            continue

        patient_map[patient_id].append(item)

    for patient_id, items in patient_map.items():

        splits = set([
            x["split"]
            for x in items
        ])

        if len(splits) > 1:

            same_patient_samples.append({

                "patient_id": patient_id,

                "samples": items
            })

    # ========================================================
    # PERCEPTUAL DUPLICATES
    # ========================================================

    print("\nChecking perceptual duplicates...")

    for i in tqdm(range(len(merged_data))):

        hash1 = merged_data[i]["phash"]

        if hash1 is None:
            continue

        hash1 = imagehash.hex_to_hash(hash1)

        for j in range(i + 1, len(merged_data)):

            hash2 = merged_data[j]["phash"]

            if hash2 is None:
                continue

            hash2 = imagehash.hex_to_hash(hash2)

            distance = hash1 - hash2

            if distance <= PHASH_THRESHOLD:

                perceptual_duplicates.append({

                    "distance": distance,

                    "img1": merged_data[i],

                    "img2": merged_data[j]
                })

    # ========================================================
    # SAVE REPORT
    # ========================================================

    REPORT_PATH = "dataset_leakage_report.txt"

    with open(REPORT_PATH, "w", encoding="utf-8") as f:

        f.write("\n=================================================\n")
        f.write("DATASET LEAKAGE REPORT\n")
        f.write("=================================================\n")

        # ====================================================
        # EXACT DUPLICATES
        # ====================================================

        f.write("\n\n========== EXACT DUPLICATES ==========\n")

        f.write(
            f"\nTotal Groups: {len(exact_duplicates)}\n"
        )

        for idx, group in enumerate(exact_duplicates):

            f.write(f"\n--- Group {idx+1} ---\n")

            for item in group:

                f.write(
                    f"{item['split']} --> "
                    f"{item['path']}\n"
                )

        # ====================================================
        # PERCEPTUAL DUPLICATES
        # ====================================================

        f.write("\n\n========== PERCEPTUAL DUPLICATES ==========\n")

        f.write(
            f"\nTotal Pairs: "
            f"{len(perceptual_duplicates)}\n"
        )

        for idx, pair in enumerate(perceptual_duplicates):

            f.write(f"\n--- Pair {idx+1} ---\n")

            f.write(
                f"Distance: {pair['distance']}\n"
            )

            f.write(
                f"{pair['img1']['split']} --> "
                f"{pair['img1']['path']}\n"
            )

            f.write(
                f"{pair['img2']['split']} --> "
                f"{pair['img2']['path']}\n"
            )

        # ====================================================
        # SAME PATIENT
        # ====================================================

        f.write(
            "\n\n========== SAME PATIENT ACROSS SPLITS ==========\n"
        )

        f.write(
            f"\nTotal Patients: "
            f"{len(same_patient_samples)}\n"
        )

        for idx, patient in enumerate(same_patient_samples):

            f.write(
                f"\n--- Patient {idx+1}: "
                f"{patient['patient_id']} ---\n"
            )

            for sample in patient["samples"]:

                f.write(
                    f"{sample['split']} --> "
                    f"{sample['path']}\n"
                )

  

        # ========================================================
        # SAVE EXACT DUPLICATES CSV
        # ========================================================

        exact_csv = "exact_duplicates.csv"

        with open(exact_csv, "w", newline="", encoding="utf-8") as f:

            writer = csv.writer(f)

            writer.writerow([
                "group_id",
                "split",
                "file_name",
                "file_path"
            ])

            for idx, group in enumerate(exact_duplicates):

                for item in group:

                    writer.writerow([
                        idx + 1,
                        item["split"],
                        item["name"],
                        item["path"]
                    ])

        print(f"\nSaved: {exact_csv}")

        # ========================================================
        # SAVE PERCEPTUAL DUPLICATES CSV
        # ========================================================

        perceptual_csv = "perceptual_duplicates.csv"

        with open(perceptual_csv, "w", newline="", encoding="utf-8") as f:

            writer = csv.writer(f)

            writer.writerow([
                "distance",
                "split_1",
                "file_name_1",
                "file_path_1",
                "split_2",
                "file_name_2",
                "file_path_2"
            ])

            for pair in perceptual_duplicates:

                writer.writerow([

                    pair["distance"],

                    pair["img1"]["split"],
                    pair["img1"]["name"],
                    pair["img1"]["path"],

                    pair["img2"]["split"],
                    pair["img2"]["name"],
                    pair["img2"]["path"]
                ])

        print(f"Saved: {perceptual_csv}")

        # ========================================================
        # SAVE SAME PATIENT CSV
        # ========================================================

        patient_csv = "same_patient_overlaps.csv"

        with open(patient_csv, "w", newline="", encoding="utf-8") as f:

            writer = csv.writer(f)

            writer.writerow([
                "patient_id",
                "split",
                "file_name",
                "file_path"
            ])

            for patient in same_patient_samples:

                patient_id = patient["patient_id"]

                for sample in patient["samples"]:

                    writer.writerow([
                        patient_id,
                        sample["split"],
                        sample["name"],
                        sample["path"]
                    ])

        print(f"Saved: {patient_csv}")

        # ========================================================
        # SAVE REMOVAL CANDIDATES
        # ========================================================

        removal_csv = "removal_candidates.csv"

        with open(removal_csv, "w", newline="", encoding="utf-8") as f:

            writer = csv.writer(f)

            writer.writerow([
                "reason",
                "recommended_remove",
                "keep"
            ])

            # Exact duplicate removals

            for group in exact_duplicates:

                keep = group[0]["path"]

                for duplicate in group[1:]:

                    writer.writerow([
                        "exact_duplicate",
                        duplicate["path"],
                        keep
                    ])

            # Perceptual duplicate removals

            for pair in perceptual_duplicates:

                writer.writerow([
                    "perceptual_duplicate",
                    pair["img2"]["path"],
                    pair["img1"]["path"]
                ])

        print(f"Saved: {removal_csv}")
    print("\n=================================================")
    print("MERGE + ANALYSIS COMPLETED")
    print("=================================================")

    print(
        f"\nExact Duplicate Groups: "
        f"{len(exact_duplicates)}"
    )

    print(
        f"Perceptual Duplicate Pairs: "
        f"{len(perceptual_duplicates)}"
    )

    print(
        f"Same Patient Overlaps: "
        f"{len(same_patient_samples)}"
    )

    print(f"\nReport Saved To: {REPORT_PATH}")