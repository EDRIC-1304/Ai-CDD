import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import imagehash
import hashlib
from collections import defaultdict
from itertools import combinations
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

DATASET_PATHS = {
    "normal": r"G:\Ai-CDD\updated_class_tb_dataset\TBX11K\imgs\health",
    "sick": r"G:\Ai-CDD\updated_class_tb_dataset\TBX11K\imgs\sick",
    "tb": r"G:\Ai-CDD\updated_class_tb_dataset\TBX11K\imgs\tb"
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
SSIM_THRESHOLD = 0.97
COMPARE_SIZE = 128

# ============================================================
# STORAGE
# ============================================================

all_images = []

exact_duplicates = []
perceptual_duplicates = []
near_identical = []

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

        img = Image.open(path).convert("L")

        return imagehash.phash(img)

    except:

        return None

# ============================================================

def load_gray(path):

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    img = cv2.resize(
        img,
        (COMPARE_SIZE, COMPARE_SIZE)
    )

    return img

# ============================================================

def compute_ssim(img1, img2):

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    mu1 = img1.mean()
    mu2 = img2.mean()

    sigma1 = img1.var()
    sigma2 = img2.var()

    covariance = ((img1 - mu1) * (img2 - mu2)).mean()

    c1 = 6.5025
    c2 = 58.5225

    ssim = (
        (2 * mu1 * mu2 + c1)
        *
        (2 * covariance + c2)
    ) / (
        (mu1**2 + mu2**2 + c1)
        *
        (sigma1 + sigma2 + c2)
    )

    return ssim

# ============================================================
# COLLECT IMAGES
# ============================================================

print("\nCollecting images...")

for class_name, folder_path in DATASET_PATHS.items():

    for file in os.listdir(folder_path):

        if not file.lower().endswith(IMAGE_EXTENSIONS):
            continue

        full_path = os.path.join(folder_path, file)

        all_images.append({

            "class": class_name,
            "name": file,
            "path": full_path
        })

print(f"\nTotal Images: {len(all_images)}")

# ============================================================
# EXACT DUPLICATES
# ============================================================

print("\nChecking exact duplicates...")

hash_map = defaultdict(list)

for item in tqdm(all_images):

    try:

        file_hash = get_file_hash(item["path"])

        hash_map[file_hash].append(item)

    except:
        continue

for file_hash, items in hash_map.items():

    if len(items) > 1:

        for pair in combinations(items, 2):

            exact_duplicates.append(pair)

print(f"\nExact Duplicate Pairs: {len(exact_duplicates)}")

# ============================================================
# COMPUTE PHASHES
# ============================================================

print("\nComputing perceptual hashes...")

phash_data = []

for item in tqdm(all_images):

    phash = get_phash(item["path"])

    if phash is None:
        continue

    phash_data.append((item, phash))

print(f"\nValid pHashes: {len(phash_data)}")

# ============================================================
# FAST PERCEPTUAL CHECK
# ============================================================

print("\nChecking perceptual duplicates...")

candidate_pairs = []

for i in tqdm(range(len(phash_data))):

    item1, hash1 = phash_data[i]

    for j in range(i + 1, len(phash_data)):

        item2, hash2 = phash_data[j]

        distance = hash1 - hash2

        if distance <= PHASH_THRESHOLD:

            perceptual_duplicates.append({

                "distance": distance,
                "img1": item1,
                "img2": item2
            })

            candidate_pairs.append((item1, item2))

print(f"\nPerceptual Duplicate Pairs: {len(perceptual_duplicates)}")

# ============================================================
# SSIM ONLY ON CANDIDATES
# ============================================================

print("\nRunning SSIM on suspicious pairs only...")

image_cache = {}

for item1, item2 in tqdm(candidate_pairs):

    path1 = item1["path"]
    path2 = item2["path"]

    if path1 not in image_cache:

        image_cache[path1] = load_gray(path1)

    if path2 not in image_cache:

        image_cache[path2] = load_gray(path2)

    img1 = image_cache[path1]
    img2 = image_cache[path2]

    if img1 is None or img2 is None:
        continue

    ssim = compute_ssim(img1, img2)

    if ssim >= SSIM_THRESHOLD:

        near_identical.append({

            "ssim": ssim,
            "img1": item1,
            "img2": item2
        })

print(f"\nNear Identical Pairs: {len(near_identical)}")

# ============================================================
# SAVE CSV FILES
# ============================================================

print("\nSaving CSV files...")

# EXACT DUPLICATES

exact_rows = []

for img1, img2 in exact_duplicates:

    exact_rows.append({

        "class_1": img1["class"],
        "file_name_1": img1["name"],
        "file_path_1": img1["path"],

        "class_2": img2["class"],
        "file_name_2": img2["name"],
        "file_path_2": img2["path"]
    })

pd.DataFrame(exact_rows).to_csv(
    "exact_duplicates.csv",
    index=False
)

# PERCEPTUAL DUPLICATES

perceptual_rows = []

for pair in perceptual_duplicates:

    perceptual_rows.append({

        "distance": pair["distance"],

        "class_1": pair["img1"]["class"],
        "file_name_1": pair["img1"]["name"],
        "file_path_1": pair["img1"]["path"],

        "class_2": pair["img2"]["class"],
        "file_name_2": pair["img2"]["name"],
        "file_path_2": pair["img2"]["path"]
    })

pd.DataFrame(perceptual_rows).to_csv(
    "perceptual_duplicates.csv",
    index=False
)

# NEAR IDENTICAL

near_rows = []

for pair in near_identical:

    near_rows.append({

        "ssim": pair["ssim"],

        "class_1": pair["img1"]["class"],
        "file_name_1": pair["img1"]["name"],
        "file_path_1": pair["img1"]["path"],

        "class_2": pair["img2"]["class"],
        "file_name_2": pair["img2"]["name"],
        "file_path_2": pair["img2"]["path"]
    })

pd.DataFrame(near_rows).to_csv(
    "near_identical.csv",
    index=False
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n=================================================")
print("CHECK COMPLETED")
print("=================================================")

print(f"\nTotal Images: {len(all_images)}")

print(f"\nExact Duplicate Pairs: {len(exact_duplicates)}")

print(f"Perceptual Duplicate Pairs: {len(perceptual_duplicates)}")

print(f"Near Identical Pairs: {len(near_identical)}")

print("\nCSV FILES SAVED:")
print("exact_duplicates.csv")
print("perceptual_duplicates.csv")
print("near_identical.csv")

print("\n=================================================")
print("DONE")
print("=================================================")