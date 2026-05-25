import os
import random
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# -----------------------------
# IMPORT FROM TRAINING
# -----------------------------
from train import CNN, load_image  # reuse EXACT same logic

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = r"G:\Ai-CDD\data\preprocessed\stage4\train"
MODEL_PATH = r"checkpoints\best.pth"

SAMPLES_PER_CLASS = 50  # keep small

# -----------------------------
# LOAD MODEL
# -----------------------------
model = CNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -----------------------------
# LOAD DATA (BALANCED)
# -----------------------------
def load_balanced_samples():
    data = {"NORMAL": [], "TUBERCULOSIS": []}

    for cls in data:
        path = os.path.join(DATA_DIR, cls)
        files = os.listdir(path)
        sampled = random.sample(files, min(SAMPLES_PER_CLASS, len(files)))

        for f in sampled:
            full_path = os.path.join(path, f)
            data[cls].append(full_path)

    samples = []
    for label, cls in enumerate(["NORMAL", "TUBERCULOSIS"]):
        for path in data[cls]:
            samples.append((path, label))

    return samples

samples = load_balanced_samples()

# -----------------------------
# EXTRACT EMBEDDINGS
# -----------------------------
embeddings = []
labels = []

with torch.no_grad():
    for path, label in samples:
        img = load_image(path).unsqueeze(0).to(DEVICE)  # EXACT same as training
        emb = model(img)

        embeddings.append(emb.cpu())
        labels.append(label)

embeddings = torch.cat(embeddings)
labels = torch.tensor(labels)

# -----------------------------
# DEBUG CHECKS (CRITICAL)
# -----------------------------
print("Embedding sample:", embeddings[0][:5])
print("Embedding variance:", embeddings.var(dim=0).mean().item())

# -----------------------------
# t-SNE VISUALIZATION
# -----------------------------
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
emb_2d = tsne.fit_transform(embeddings.numpy())

# -----------------------------
# PLOT
# -----------------------------
plt.figure(figsize=(8, 6))

for i in range(len(emb_2d)):
    if labels[i] == 0:
        plt.scatter(emb_2d[i, 0], emb_2d[i, 1], marker='o')
    else:
        plt.scatter(emb_2d[i, 0], emb_2d[i, 1], marker='x')

plt.title("Embedding Space (TB vs NORMAL)")
plt.show()