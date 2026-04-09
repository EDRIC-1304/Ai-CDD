import os
import torch
import cv2
import numpy as np

from train import CNN, compute_prototypes

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = r"G:\Ai-CDD\data\preprocessed\stage4\train"
CHECKPOINT = "checkpoints/best.pth"

IMG_SIZE = 224

# -----------------------------
# LOAD MODEL
# -----------------------------
model = CNN().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

# -----------------------------
# LOAD IMAGE
# -----------------------------
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    return torch.tensor(img).unsqueeze(0)

# -----------------------------
# LOAD DATA
# -----------------------------
data = {"NORMAL": [], "TUBERCULOSIS": []}

for cls in data:
    cls_path = os.path.join(DATA_DIR, cls)
    files = [os.path.join(cls_path, f) for f in os.listdir(cls_path)]
    data[cls] = files

# -----------------------------
# COMPUTE EMBEDDINGS
# -----------------------------
embeddings = []
labels = []

with torch.no_grad():
    for label, cls in enumerate(["NORMAL", "TUBERCULOSIS"]):
        for path in data[cls]:

            img = load_image(path).unsqueeze(0).to(DEVICE)
            emb = model(img)

            embeddings.append(emb)
            labels.append(label)

embeddings = torch.cat(embeddings)
labels = torch.tensor(labels).to(DEVICE)

# -----------------------------
# COMPUTE PROTOTYPES
# -----------------------------
prototypes = compute_prototypes(embeddings, labels)

# -----------------------------
# SAVE
# -----------------------------
torch.save(prototypes, "checkpoints/prototypes.pth")

print("✅ Prototypes saved successfully")