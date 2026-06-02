# =========================================================
# RESNET18 TB vs SICK EVALUATION
# =========================================================

import os
import shutil
import tempfile
import warnings

import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =========================================================
# TORCH
# =========================================================

import torch
import torch.nn as nn

from torchvision import datasets
from torchvision import transforms
from torchvision import models

from torch.utils.data import DataLoader

# =========================================================
# SKLEARN
# =========================================================

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# =========================================================
# CONFIG
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMG_SIZE = 224
BATCH_SIZE = 32

VALID_CLASSES = [
    "sick",
    "tb"
]

# =========================================================
# PATHS
# =========================================================

MODEL_PATH = (
    r"G:\Ai-CDD\classification_checkpoints"
    r"\best_resnet18_tb_vs_sick.pth"
)

BASE_DATASET = (
    r"G:\Ai-CDD\data\preprocessed\stage4"
)

TEST_DIR = os.path.join(
    BASE_DATASET,
    "internal_test"
)

# =========================================================
# OUTPUT DIRECTORY
# =========================================================

OUTPUT_DIR = (
    r"G:\Ai-CDD\Evaluation of TB Models"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

CONF_MATRIX_PATH = os.path.join(
    OUTPUT_DIR,
    "2 Label ResNet18 (TB vs Sick).png"
)

REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "2 Label ResNet18 (TB vs Sick).txt"
)

# =========================================================
# CREATE TEMP TEST DATASET
# =========================================================

TEMP_DATASET = tempfile.mkdtemp()

TEMP_TEST = os.path.join(
    TEMP_DATASET,
    "internal_test"
)

os.makedirs(
    TEMP_TEST,
    exist_ok=True
)

for cls in VALID_CLASSES:

    src = os.path.join(
        TEST_DIR,
        cls
    )

    dst = os.path.join(
        TEMP_TEST,
        cls
    )

    if not os.path.exists(src):

        print(f"Missing class folder: {src}")
        continue

    shutil.copytree(
        src,
        dst
    )

print("Temporary dataset created.")

# =========================================================
# TRANSFORMS
# =========================================================

test_transform = transforms.Compose([

    transforms.Grayscale(
        num_output_channels=3
    ),

    transforms.Resize(
        (IMG_SIZE, IMG_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================================
# DATASET
# =========================================================

test_dataset = datasets.ImageFolder(
    TEMP_TEST,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"Test Samples: {len(test_dataset)}")

# =========================================================
# MODEL
# =========================================================

model = models.resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()

print("Model Loaded Successfully")

# =========================================================
# EVALUATION
# =========================================================

all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probs = torch.softmax(
            outputs,
            dim=1
        )

        preds = torch.argmax(
            probs,
            dim=1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_probs.extend(
            probs[:, 1].cpu().numpy()
        )

# =========================================================
# METRICS
# =========================================================

accuracy = accuracy_score(
    all_labels,
    all_preds
)

precision = precision_score(
    all_labels,
    all_preds
)

recall = recall_score(
    all_labels,
    all_preds
)

f1 = f1_score(
    all_labels,
    all_preds
)

auc = roc_auc_score(
    all_labels,
    all_probs
)

cm = confusion_matrix(
    all_labels,
    all_preds
)

tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp)

# =========================================================
# PRINT RESULTS
# =========================================================

print("\n" + "=" * 60)
print("RESNET18 TB vs SICK EVALUATION")
print("=" * 60)

print(f"Accuracy    : {accuracy:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"Specificity : {specificity:.4f}")
print(f"F1 Score    : {f1:.4f}")
print(f"ROC-AUC     : {auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

report = classification_report(
    all_labels,
    all_preds,
    target_names=VALID_CLASSES
)

print("\nClassification Report:\n")
print(report)

# =========================================================
# SAVE REPORT
# =========================================================

with open(
    REPORT_PATH,
    "w"
) as f:

    f.write(
        "2 Label ResNet18 (TB vs Sick)\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        f"Accuracy    : {accuracy:.4f}\n"
    )

    f.write(
        f"Precision   : {precision:.4f}\n"
    )

    f.write(
        f"Recall      : {recall:.4f}\n"
    )

    f.write(
        f"Specificity : {specificity:.4f}\n"
    )

    f.write(
        f"F1 Score    : {f1:.4f}\n"
    )

    f.write(
        f"ROC-AUC     : {auc:.4f}\n\n"
    )

    f.write("Confusion Matrix\n")
    f.write(str(cm))
    f.write("\n\n")

    f.write(report)

# =========================================================
# LIGHT CONFUSION MATRIX
# =========================================================

plt.figure(
    figsize=(8, 7)
)

plt.imshow(
    cm,
    cmap="Pastel2"
)

plt.title(
    "2 Label ResNet18 (TB vs Sick)",
    fontsize=14,
    fontweight="bold"
)

plt.colorbar()

ticks = np.arange(
    len(VALID_CLASSES)
)

plt.xticks(
    ticks,
    VALID_CLASSES,
    fontsize=12
)

plt.yticks(
    ticks,
    VALID_CLASSES,
    fontsize=12
)

plt.xlabel(
    "Predicted Label",
    fontsize=12
)

plt.ylabel(
    "True Label",
    fontsize=12
)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="black"
        )

plt.tight_layout()

plt.savefig(
    CONF_MATRIX_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =========================================================
# CLEANUP
# =========================================================

shutil.rmtree(
    TEMP_DATASET
)

print(
    f"\nConfusion Matrix Saved:\n{CONF_MATRIX_PATH}"
)

print(
    f"\nReport Saved:\n{REPORT_PATH}"
)

print("\nTemporary dataset cleaned.")

print("\nEvaluation Complete.")