import os
import random
import shutil

# =========================================================
# CONFIG
# =========================================================
NORMAL_DIR = r"G:\Ai-CDD\data\preprocessed\stage4\test\NORMAL"

TB_DIR = r"G:\Ai-CDD\data\preprocessed\stage4\test\TUBERCULOSIS"

OUTPUT_DIR = r"G:\Ai-CDD\data\preprocessed\stage4"

SEED = 42

EXTERNAL_COUNT = 100

VALID_EXT = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp"
)

random.seed(SEED)

# =========================================================
# CREATE OUTPUT FOLDERS
# =========================================================
internal_normal = os.path.join(
    OUTPUT_DIR,
    "internal_test",
    "NORMAL"
)

internal_tb = os.path.join(
    OUTPUT_DIR,
    "internal_test",
    "TUBERCULOSIS"
)

external_normal = os.path.join(
    OUTPUT_DIR,
    "external_test",
    "NORMAL"
)

external_tb = os.path.join(
    OUTPUT_DIR,
    "external_test",
    "TUBERCULOSIS"
)

folders = [

    internal_normal,
    internal_tb,

    external_normal,
    external_tb
]

for folder in folders:

    os.makedirs(folder, exist_ok=True)

# =========================================================
# LOAD FILES
# =========================================================
normal_files = [

    f for f in os.listdir(NORMAL_DIR)

    if f.lower().endswith(VALID_EXT)
]

tb_files = [

    f for f in os.listdir(TB_DIR)

    if f.lower().endswith(VALID_EXT)
]

print("\n==========================")
print("ORIGINAL DATA")
print("==========================")

print(f"NORMAL : {len(normal_files)}")
print(f"TB     : {len(tb_files)}")

# =========================================================
# BALANCE DATASET
# =========================================================
min_count = min(
    len(normal_files),
    len(tb_files)
)

random.shuffle(normal_files)
random.shuffle(tb_files)

normal_files = normal_files[:min_count]
tb_files = tb_files[:min_count]

print("\n==========================")
print("BALANCED DATA")
print("==========================")

print(f"NORMAL : {len(normal_files)}")
print(f"TB     : {len(tb_files)}")

# =========================================================
# SPLIT EXTERNAL
# =========================================================
external_normal_files = normal_files[:EXTERNAL_COUNT]
external_tb_files = tb_files[:EXTERNAL_COUNT]

# =========================================================
# REMAINING = INTERNAL
# =========================================================
internal_normal_files = normal_files[EXTERNAL_COUNT:]
internal_tb_files = tb_files[EXTERNAL_COUNT:]

# =========================================================
# COPY EXTERNAL
# =========================================================
print("\nCopying external test files...")

for file in external_normal_files:

    src = os.path.join(NORMAL_DIR, file)

    dst = os.path.join(external_normal, file)

    shutil.copy2(src, dst)

for file in external_tb_files:

    src = os.path.join(TB_DIR, file)

    dst = os.path.join(external_tb, file)

    shutil.copy2(src, dst)

# =========================================================
# COPY INTERNAL
# =========================================================
print("Copying internal test files...")

for file in internal_normal_files:

    src = os.path.join(NORMAL_DIR, file)

    dst = os.path.join(internal_normal, file)

    shutil.copy2(src, dst)

for file in internal_tb_files:

    src = os.path.join(TB_DIR, file)

    dst = os.path.join(internal_tb, file)

    shutil.copy2(src, dst)

# =========================================================
# FINAL STATS
# =========================================================
print("\n==========================")
print("FINAL SPLIT")
print("==========================")

print("\nINTERNAL TEST")

print(
    f"NORMAL : {len(internal_normal_files)}"
)

print(
    f"TB     : {len(internal_tb_files)}"
)

print("\nEXTERNAL TEST")

print(
    f"NORMAL : {len(external_normal_files)}"
)

print(
    f"TB     : {len(external_tb_files)}"
)

print("\n✅ Dataset split complete")