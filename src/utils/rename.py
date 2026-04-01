import os
import re

folders = [
    r"C:\Users\Ant PC\Desktop\22co12 FY_PROJ\AI-CDD\Ai-CDD\data\raw\train\tb vs normal classification dataset\test\Normal",
    r"C:\Users\Ant PC\Desktop\22co12 FY_PROJ\AI-CDD\Ai-CDD\data\raw\train\tb vs normal classification dataset\test\TB",
    r"C:\Users\Ant PC\Desktop\22co12 FY_PROJ\AI-CDD\Ai-CDD\data\raw\train\tb vs normal classification dataset\train\normal",
    r"C:\Users\Ant PC\Desktop\22co12 FY_PROJ\AI-CDD\Ai-CDD\data\raw\train\tb vs normal classification dataset\train\TB",
    r"C:\Users\Ant PC\Desktop\22co12 FY_PROJ\AI-CDD\Ai-CDD\data\raw\train\tb vs normal classification dataset\validation\Normal",
    r"C:\Users\Ant PC\Desktop\22co12 FY_PROJ\AI-CDD\Ai-CDD\data\raw\train\tb vs normal classification dataset\validation\TB"
]

def natural_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

for folder in folders:
    print(f"Processing: {folder}")

    files = sorted(os.listdir(folder), key=natural_key)

    # Filter only files
    files = [f for f in files if os.path.isfile(os.path.join(folder, f))]

    if not files:
        print("  Skipped (empty folder)")
        continue

    # Step 1: temp rename
    temp_names = []
    for i, filename in enumerate(files):
        ext = os.path.splitext(filename)[1]
        temp_name = f"temp_{i}{ext}"

        os.rename(
            os.path.join(folder, filename),
            os.path.join(folder, temp_name)
        )
        temp_names.append(temp_name)

    # Step 2: final rename
    for i, filename in enumerate(temp_names, start=1):
        ext = os.path.splitext(filename)[1]
        new_name = f"img_{i:04d}{ext}"

        os.rename(
            os.path.join(folder, filename),
            os.path.join(folder, new_name)
        )

    print(f"  Done: {len(files)} files renamed\n")