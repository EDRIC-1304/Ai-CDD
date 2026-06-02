# =========================================================
# RESNET18 MULTICLASS EVALUATION
# =========================================================

import os
import shutil
import tempfile
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torchvision import datasets
from torchvision import transforms
from torchvision import models

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# CONFIG
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMG_SIZE = 224

BATCH_SIZE = 32

CLASS_NAMES = [
    "health",
    "sick",
    "tb"
]

VALID_CLASSES = [
    "health",
    "sick",
    "tb"
]

# =========================================================
# PATHS
# =========================================================

MODEL_PATH = (
    r"G:\Ai-CDD\classification_checkpoints"
    r"\best_resnet18_multiclass.pth"
)

TEST_DIR = (
    r"G:\Ai-CDD\data\preprocessed\stage4"
    r"\internal_test"
)

OUTPUT_DIR = (
    r"G:\Ai-CDD\Evaluation of TB Models"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# =========================================================
# CREATE FILTERED TEST SET
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

    if os.path.exists(src):

        shutil.copytree(
            src,
            dst
        )

# =========================================================
# TRANSFORMS
# =========================================================

transform = transforms.Compose([

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
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nClass Mapping:")
print(test_dataset.class_to_idx)

print(
    f"\nTotal Test Images: "
    f"{len(test_dataset)}"
)

# =========================================================
# MODEL
# =========================================================

model = models.resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    3
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)

model.eval()

print("\nModel Loaded Successfully")

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
            probs.cpu().numpy()
        )

all_probs = np.array(all_probs)

# =========================================================
# METRICS
# =========================================================

acc = accuracy_score(
    all_labels,
    all_preds
)

precision = precision_score(
    all_labels,
    all_preds,
    average="weighted"
)

recall = recall_score(
    all_labels,
    all_preds,
    average="weighted"
)

f1 = f1_score(
    all_labels,
    all_preds,
    average="weighted"
)

auc = roc_auc_score(
    all_labels,
    all_probs,
    multi_class="ovr"
)

# =========================================================
# PRINT RESULTS
# =========================================================

print("\n" + "="*60)
print("RESNET18 3-LABEL EVALUATION")
print("="*60)

print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {auc:.4f}")

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

report = classification_report(
    all_labels,
    all_preds,
    target_names=CLASS_NAMES
)

print("\n")
print(report)

with open(
    os.path.join(
        OUTPUT_DIR,
        "resnet18_3label_report.txt"
    ),
    "w"
) as f:

    f.write(report)

# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    all_labels,
    all_preds
)

plt.figure(figsize=(8, 7))

plt.imshow(
    cm,
    cmap="Pastel1"
)

plt.title(
    "3 Label ResNet18 Confusion Matrix",
    fontsize=14,
    fontweight="bold"
)

plt.colorbar()

ticks = np.arange(
    len(CLASS_NAMES)
)

plt.xticks(
    ticks,
    CLASS_NAMES
)

plt.yticks(
    ticks,
    CLASS_NAMES
)

plt.xlabel(
    "Predicted Class",
    fontsize=12
)

plt.ylabel(
    "True Class",
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
            color="black",
            fontsize=12,
            fontweight="bold"
        )

plt.tight_layout()

cm_path = os.path.join(
    OUTPUT_DIR,
    "3_label_resnet18_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"\nConfusion Matrix Saved:\n{cm_path}"
)

# =========================================================
# SAVE METRICS
# =========================================================

with open(
    os.path.join(
        OUTPUT_DIR,
        "resnet18_metrics.txt"
    ),
    "w"
) as f:

    f.write(
        f"Accuracy  : {acc:.4f}\n"
        f"Precision : {precision:.4f}\n"
        f"Recall    : {recall:.4f}\n"
        f"F1 Score  : {f1:.4f}\n"
        f"ROC-AUC   : {auc:.4f}\n"
    )

print(
    "\nEvaluation completed successfully."
)