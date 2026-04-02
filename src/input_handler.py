import os
import cv2
from tqdm import tqdm

# -----------------------------
# PATHS
# -----------------------------
INPUT_DIR_TEST_NORMAL = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/raw/train/tb vs normal classification dataset/test/Normal"
OUTPUT_DIR_TEST_NORMAL = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/processed/xray classification dataset input handled/test/Normal"

INPUT_DIR_TEST_TB = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/raw/train/tb vs normal classification dataset/test/TB"
OUTPUT_DIR_TEST_TB = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/processed/xray classification dataset input handled/test/TB"

INPUT_DIR_TRAIN_NORMAL = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/raw/train/tb vs normal classification dataset/train/normal"
OUTPUT_DIR_TRAIN_NORMAL = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/processed/xray classification dataset input handled/train/normal"

# ✅ FIXED PATH (was wrong before)
INPUT_DIR_TRAIN_TB = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/raw/train/tb vs normal classification dataset/train/TB"
OUTPUT_DIR_TRAIN_TB = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/processed/xray classification dataset input handled/train/TB"

INPUT_DIR_VAL_NORMAL = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/raw/train/tb vs normal classification dataset/validation/Normal"
OUTPUT_DIR_VAL_NORMAL = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/processed/xray classification dataset input handled/validation/Normal"

INPUT_DIR_VAL_TB = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/raw/train/tb vs normal classification dataset/validation/TB"
OUTPUT_DIR_VAL_TB = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/processed/xray classification dataset input handled/validation/TB"

# -----------------------------
# VALID IMAGE EXTENSIONS
# -----------------------------
VALID_EXTENSIONS = [".png", ".jpg", ".jpeg"]

# -----------------------------
# CHECK IF IMAGE IS VALID
# -----------------------------
def is_valid_image(path):
    try:
        img = cv2.imread(path)
        if img is None or img.size == 0:
            return False
        return True
    except:
        return False

# -----------------------------
# MAIN CLEANING PIPELINE
# -----------------------------
def clean_dataset(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # ✅ FIXED

    count = 0
    skipped = 0
    write_attempts = 0
    write_success = 0
    write_fail = 0

    seen_names = set()
    duplicates = []



    for file in tqdm(os.listdir(input_dir)):
        ext = os.path.splitext(file)[1].lower()

        if ext not in VALID_EXTENSIONS:
            continue

        input_path = os.path.join(input_dir, file)

        name = os.path.splitext(file)[0]
        if name in seen_names:
            duplicates.append(name)
        else:
            seen_names.add(name)

        # Validate image
        if not is_valid_image(input_path):
            print(f"❌ Corrupt image removed: {input_path}")
            skipped += 1
            continue

        # Read image
        img = cv2.imread(input_path)

        # Preserve filename (only change extension to PNG)
        filename = os.path.splitext(file)[0] + ".png"
        output_path = os.path.join(output_dir, filename)
        
        success = cv2.imwrite(output_path, img)
        write_attempts +=1
        
        check = cv2.imread(output_path)

        if (not success) or (check is None):
            write_fail += 1
            continue
        else:
            write_success +=1


        count += 1

    print("\n✅ Cleaning Completed:", input_dir)
    print("Unique filenames:", len(seen_names))
    print(f"✔ Valid images saved: {count}")
    print(f"❌ Corrupt/Skipped: {skipped}")
    print(f"Write Attempts: {write_attempts}")
    print(f"Success: {write_success}")
    print(f"Fail: {write_fail}")
    print("Duplicate names:\n", duplicates)
    print("-" * 50)

# -----------------------------
# RUN
# -----------------------------
# clean_dataset(INPUT_DIR_TEST_NORMAL, OUTPUT_DIR_TEST_NORMAL)
# clean_dataset(INPUT_DIR_TEST_TB, OUTPUT_DIR_TEST_TB)
clean_dataset(INPUT_DIR_TRAIN_NORMAL, OUTPUT_DIR_TRAIN_NORMAL)
clean_dataset(INPUT_DIR_TRAIN_TB, OUTPUT_DIR_TRAIN_TB)
clean_dataset(INPUT_DIR_VAL_NORMAL, OUTPUT_DIR_VAL_NORMAL)
clean_dataset(INPUT_DIR_VAL_TB, OUTPUT_DIR_VAL_TB)