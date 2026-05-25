import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = r"G:\Ai-CDD\data\preprocessed\stage4\train"
MODEL_PATH = "checkpoints/best.pth"

IMG_SIZE = 224


# -----------------------------
# LOAD MODEL (same as training)
# -----------------------------
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return F.normalize(x, dim=1)  # 🔥 IMPORTANT


# -----------------------------
# LOAD IMAGE
# -----------------------------
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    return torch.tensor(img).unsqueeze(0)


# -----------------------------
# LOAD RANDOM SAMPLES
# -----------------------------
def get_samples(root, n=20):
    data = {"NORMAL": [], "TUBERCULOSIS": []}

    for cls in data:
        path = os.path.join(root, cls)
        files = os.listdir(path)
        chosen = random.sample(files, min(n, len(files)))

        for f in chosen:
            data[cls].append(os.path.join(path, f))

    return data


# -----------------------------
# MAIN DIAGNOSTIC
# -----------------------------
def run_diagnostics():

    print("\n🔍 LOADING MODEL...")
    model = CNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("✅ Model loaded\n")

    data = get_samples(DATA_DIR, n=30)

    # -----------------------------
    # STEP 1: EMBEDDINGS
    # -----------------------------
    print("🔹 STEP 1: Checking embeddings...")

    emb_normal = []
    emb_tb = []

    for path in data["NORMAL"]:
        img = load_image(path).to(DEVICE)
        emb = model(img.unsqueeze(0)).detach().cpu()
        emb_normal.append(emb)

    for path in data["TUBERCULOSIS"]:
        img = load_image(path).to(DEVICE)
        emb = model(img.unsqueeze(0)).detach().cpu()
        emb_tb.append(emb)

    emb_normal = torch.cat(emb_normal)
    emb_tb = torch.cat(emb_tb)

    print("Normal embedding mean:", emb_normal.mean().item())
    print("TB embedding mean:", emb_tb.mean().item())

    # -----------------------------
    # STEP 2: PROTOTYPES
    # -----------------------------
    print("\n🔹 STEP 2: Checking prototypes...")

    proto_normal = emb_normal.mean(0)
    proto_tb = emb_tb.mean(0)

    dist_between = torch.norm(proto_normal - proto_tb).item()

    print("Distance between prototypes:", dist_between)

    if dist_between < 0.5:
        print("❌ BAD: Prototypes too close → model cannot distinguish classes")
    else:
        print("✅ GOOD: Prototypes separated")

    # -----------------------------
    # STEP 3: DISTANCE BEHAVIOR
    # -----------------------------
    print("\n🔹 STEP 3: Checking prediction distances...")

    def dist(x, y):
        return ((x - y) ** 2).sum()

    correct_tb = 0
    total_tb = 0

    for emb in emb_tb:
        d_n = dist(emb, proto_normal)
        d_t = dist(emb, proto_tb)

        print(f"TB sample → d_normal: {d_n:.4f}, d_tb: {d_t:.4f}")

        if d_t < d_n:
            correct_tb += 1
        total_tb += 1

    tb_acc = correct_tb / total_tb
    print(f"\nTB classification accuracy (prototype): {tb_acc:.2f}")

    if tb_acc < 0.5:
        print("❌ MODEL FAILING: TB classified as NORMAL")
    else:
        print("✅ TB separation working")

    # -----------------------------
    # STEP 4: EMBEDDING COLLAPSE CHECK
    # -----------------------------
    print("\n🔹 STEP 4: Checking embedding diversity...")

    var_normal = emb_normal.var().item()
    var_tb = emb_tb.var().item()

    print("Normal variance:", var_normal)
    print("TB variance:", var_tb)

    if var_normal < 1e-3 and var_tb < 1e-3:
        print("❌ EMBEDDING COLLAPSE detected")
    else:
        print("✅ Embeddings have diversity")

    print("\n🏁 DIAGNOSTIC COMPLETE")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    run_diagnostics()