import os
import csv
import cv2
import numpy as np

import torch
import torch.nn as nn

from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

import timm

# =========================================================
# CONFIG
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_DIR = r"G:\Ai-CDD\data\preprocessed\stage4\test"
CHECKPOINT = r"G:\Ai-CDD\convnext_checkpoints\best_convnext.pth"

IMG_SIZE = 224

CLASS_NAMES = ["NORMAL", "TUBERCULOSIS"]

# =========================================================
# MODEL
# =========================================================
class ConvNextTB(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            "convnext_tiny",
            pretrained=False,
            num_classes=2
        )

    def forward(self, x):
        return self.backbone(x)

# =========================================================
# LOAD MODEL
# =========================================================
model = ConvNextTB().to(DEVICE)

model.load_state_dict(
    torch.load(CHECKPOINT, map_location=DEVICE)
)

model.eval()

print("✅ ConvNeXt model loaded")

# =========================================================
# LOAD IMAGE
# =========================================================
def load_image(path):

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img = img.astype(np.float32) / 255.0

    img = np.stack([img, img, img], axis=0)

    tensor = torch.tensor(
        img,
        dtype=torch.float32
    ).unsqueeze(0)

    return tensor

# =========================================================
# RUN INFERENCE
# =========================================================
def run_inference():

    y_true = []
    y_pred = []

    rows = []

    for label, cls in enumerate(CLASS_NAMES):

        class_dir = os.path.join(TEST_DIR, cls)

        files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        for file in tqdm(files, desc=cls):

            path = os.path.join(class_dir, file)

            image = load_image(path)

            if image is None:
                continue

            image = image.to(DEVICE)

            with torch.no_grad():

                output = model(image)

                probs = torch.softmax(output, dim=1)

                conf, pred = torch.max(probs, dim=1)

            pred = pred.item()
            conf = conf.item()

            y_true.append(label)
            y_pred.append(pred)

            rows.append({
                "image": file,
                "true": CLASS_NAMES[label],
                "pred": CLASS_NAMES[pred],
                "confidence": round(conf, 4)
            })

    return y_true, y_pred, rows

# =========================================================
# SAVE CSV
# =========================================================
def save_csv(rows):

    with open("convnext_results.csv", "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)

    print("✅ Results CSV saved")

# =========================================================
# PLOT CONFUSION MATRIX
# =========================================================
def plot_cm(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("ConvNeXt Confusion Matrix")

    plt.savefig("convnext_confusion_matrix.png")

    print("✅ Confusion matrix saved")

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    y_true, y_pred, rows = run_inference()

    print("\n🎯 FINAL RESULTS")

    print("\nAccuracy:")
    print(accuracy_score(y_true, y_pred))

    print("\nClassification Report:\n")
    print(classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES
    ))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_true, y_pred))

    save_csv(rows)

    plot_cm(y_true, y_pred)