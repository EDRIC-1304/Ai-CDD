# import os
# import torch
# import cv2
# import numpy as np

# from train import CNN, compute_prototypes

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATA_DIR = r"G:\Ai-CDD\data\preprocessed\stage4\train"
# CHECKPOINT = "checkpoints/best.pth"

# IMG_SIZE = 224

# # -----------------------------
# # LOAD MODEL
# # -----------------------------
# model = CNN().to(DEVICE)
# model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
# model.eval()

# # -----------------------------
# # LOAD IMAGE
# # -----------------------------
# def load_image(path):
#     img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
#     img = img.astype(np.float32) / 255.0
#     return torch.tensor(img).unsqueeze(0)

# # -----------------------------
# # LOAD DATA
# # -----------------------------
# data = {"NORMAL": [], "TUBERCULOSIS": []}

# for cls in data:
#     cls_path = os.path.join(DATA_DIR, cls)
#     files = [os.path.join(cls_path, f) for f in os.listdir(cls_path)]
#     data[cls] = files

# # -----------------------------
# # COMPUTE EMBEDDINGS
# # -----------------------------
# embeddings = []
# labels = []

# with torch.no_grad():
#     for label, cls in enumerate(["NORMAL", "TUBERCULOSIS"]):
#         for path in data[cls]:

#             img = load_image(path).unsqueeze(0).to(DEVICE)
#             emb = model(img)

#             embeddings.append(emb)
#             labels.append(label)

# embeddings = torch.cat(embeddings)
# labels = torch.tensor(labels).to(DEVICE)

# # -----------------------------
# # COMPUTE PROTOTYPES
# # -----------------------------
# prototypes = compute_prototypes(embeddings, labels)

# # -----------------------------
# # SAVE
# # -----------------------------
# torch.save(prototypes, "checkpoints/prototypes.pth")

# print("✅ Prototypes saved successfully")






import os
import torch
import cv2
import numpy as np
import torch.nn.functional as F

from train import CNN, compute_prototypes

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

DATA_DIR = r"G:\Ai-CDD\data\preprocessed\stage4\train"

CHECKPOINT = "checkpoints/best.pth"

OUTPUT_PATH = "checkpoints/prototypes.pth"

IMG_SIZE = 224

VALID_EXT = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = CNN().to(DEVICE)

state_dict = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(state_dict)

model.eval()

print("✅ Model loaded successfully")

# -----------------------------
# LOAD IMAGE
# -----------------------------
def load_image(path):

    img = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        raise ValueError(f"❌ Failed to load: {path}")

    img = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE)
    )

    img = img.astype(np.float32) / 255.0

    # -------------------------
    # PROTECT AGAINST
    # COMPLETELY EMPTY IMAGES
    # -------------------------
    if img.mean() < 0.01:
        img = np.zeros(
            (IMG_SIZE, IMG_SIZE),
            dtype=np.float32
        )

    tensor = torch.tensor(
        img,
        dtype=torch.float32
    ).unsqueeze(0)

    return tensor

# -----------------------------
# LOAD DATA
# -----------------------------
data = {
    "NORMAL": [],
    "TUBERCULOSIS": []
}

for cls in data:

    cls_path = os.path.join(
        DATA_DIR,
        cls
    )

    if not os.path.exists(cls_path):
        print(f"❌ Missing folder: {cls_path}")
        continue

    files = [
        os.path.join(cls_path, f)
        for f in os.listdir(cls_path)
        if f.lower().endswith(VALID_EXT)
    ]

    data[cls] = files

    print(f"{cls}: {len(files)} images")

# -----------------------------
# COMPUTE EMBEDDINGS
# -----------------------------
embeddings = []
labels = []

processed = 0
skipped = 0

with torch.no_grad():

    for label, cls in enumerate([
        "NORMAL",
        "TUBERCULOSIS"
    ]):

        for path in data[cls]:

            try:

                img = load_image(path)

                img = img.unsqueeze(0).to(DEVICE)

                emb = model(img)

                embeddings.append(emb)

                labels.append(label)

                processed += 1

            except Exception as e:

                print(f"❌ Error: {path}")
                print(e)

                skipped += 1

# -----------------------------
# STACK
# -----------------------------
embeddings = torch.cat(embeddings)

labels = torch.tensor(
    labels,
    dtype=torch.long
).to(DEVICE)

print(f"\n✔ Processed: {processed}")
print(f"❌ Skipped: {skipped}")

# -----------------------------
# COMPUTE PROTOTYPES
# -----------------------------
prototypes = compute_prototypes(
    embeddings,
    labels
)

# -----------------------------
# NORMALIZE PROTOTYPES
# IMPORTANT
# -----------------------------
prototypes = F.normalize(
    prototypes,
    dim=1
)

# -----------------------------
# SAVE
# -----------------------------
torch.save(
    prototypes,
    OUTPUT_PATH
)

print(f"\n✅ Prototypes saved successfully")
print(f"📁 Saved at: {OUTPUT_PATH}")

# -----------------------------
# DEBUG
# -----------------------------
print("\nPrototype Shape:")
print(prototypes.shape)

print("\nPrototype Norms:")
print(torch.norm(prototypes, dim=1))