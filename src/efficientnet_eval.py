# =========================================================
# EFFICIENTNET-B3 COMPLETE EVALUATION SCRIPT
# =========================================================
#
# EVALUATES:
#
# 1. EfficientNet-B3 3 Label
#    health vs sick vs tb
#
# 2. EfficientNet-B3 2 Label
#    health vs sick
#
# 3. EfficientNet-B3 2 Label
#    sick vs tb
#
# OUTPUT FOLDER:
# G:\Ai-CDD\Evaluation of TB Models
#
# SAVES:
# - Evaluation TXT files
# - Classification Reports
# - Confusion Matrices
#
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

from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms
from torchvision import models

# =========================================================
# SKLEARN
# =========================================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# =========================================================
# CONFIG
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMG_SIZE = 300
BATCH_SIZE = 16
NUM_WORKERS = 0

print(f"\nUsing Device: {DEVICE}")

# =========================================================
# DATASET
# =========================================================

BASE_DATASET = r"G:\Ai-CDD\data\preprocessed\stage4"

TEST_DIR = os.path.join(
    BASE_DATASET,
    "internal_test"
)

# =========================================================
# OUTPUT DIRECTORY
# =========================================================

OUTPUT_DIR = r"G:\Ai-CDD\Evaluation of TB Models"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# =========================================================
# MODEL PATHS
# =========================================================

MULTICLASS_MODEL = (
    r"G:\Ai-CDD\classification_checkpoints\best_efficientnetb3_multiclass.pth"
)

NORMAL_VS_SICK_MODEL = (
    r"G:\Ai-CDD\classification_checkpoints\best_efficientnetb3_binary.pth"
)

SICK_VS_TB_MODEL = (
    r"G:\Ai-CDD\classification_checkpoints\best_efficientnet_tb_vs_sick.pth"
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
# CONFUSION MATRIX
# =========================================================

def save_confusion_matrix(
    cm,
    class_names,
    title,
    save_path
):

    plt.figure(figsize=(8, 8))

    plt.imshow(
        cm,
        cmap="Pastel1"
    )

    plt.title(
        title,
        fontsize=16,
        fontweight="bold"
    )

    plt.colorbar()

    ticks = np.arange(
        len(class_names)
    )

    plt.xticks(
        ticks,
        class_names,
        fontsize=11
    )

    plt.yticks(
        ticks,
        class_names,
        fontsize=11
    )

    plt.xlabel(
        "Predicted Label",
        fontsize=12
    )

    plt.ylabel(
        "Actual Label",
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
                fontsize=12,
                color="black"
            )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# =========================================================
# MODEL CREATOR
# =========================================================

def create_efficientnet_b3(num_classes):

    model = models.efficientnet_b3(
        weights=None
    )

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        num_classes
    )

    return model

# =========================================================
# DATASET FILTER
# =========================================================

def create_filtered_dataset(
    classes_needed
):

    temp_dir = tempfile.mkdtemp()

    temp_test = os.path.join(
        temp_dir,
        "internal_test"
    )

    os.makedirs(
        temp_test,
        exist_ok=True
    )

    for cls in classes_needed:

        src = os.path.join(
            TEST_DIR,
            cls
        )

        dst = os.path.join(
            temp_test,
            cls
        )

        if os.path.exists(src):

            shutil.copytree(
                src,
                dst
            )

    dataset = datasets.ImageFolder(
        temp_test,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return (
        loader,
        dataset,
        temp_dir
    )

# =========================================================
# EVALUATION FUNCTION
# =========================================================

def evaluate_model(
    model_name,
    model_path,
    classes
):

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    test_loader, dataset, temp_dir = (
        create_filtered_dataset(classes)
    )

    model = create_efficientnet_b3(
        len(classes)
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE
        )
    )

    model = model.to(DEVICE)

    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(
                DEVICE
            )

            outputs = model(
                images
            )

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

    accuracy = accuracy_score(
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

    if len(classes) == 2:

        auc = roc_auc_score(
            all_labels,
            np.array(all_probs)[:, 1]
        )

    else:

        auc = roc_auc_score(
            all_labels,
            all_probs,
            multi_class="ovr"
        )

    report = classification_report(
        all_labels,
        all_preds,
        target_names=classes
    )

    cm = confusion_matrix(
        all_labels,
        all_preds
    )

    txt_path = os.path.join(
        OUTPUT_DIR,
        f"{model_name}.txt"
    )

    with open(
        txt_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "=" * 70 + "\n"
        )

        f.write(
            model_name + "\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            f"Accuracy : {accuracy:.4f}\n"
        )

        f.write(
            f"Precision: {precision:.4f}\n"
        )

        f.write(
            f"Recall   : {recall:.4f}\n"
        )

        f.write(
            f"F1 Score : {f1:.4f}\n"
        )

        f.write(
            f"ROC-AUC  : {auc:.4f}\n\n"
        )

        f.write(
            "Classification Report\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(report)

        f.write(
            "\n\nConfusion Matrix\n"
        )

        f.write(
            "=" * 70 + "\n"
        )

        f.write(
            str(cm)
        )

    matrix_path = os.path.join(
        OUTPUT_DIR,
        f"{model_name}_Confusion_Matrix.png"
    )

    save_confusion_matrix(
        cm=cm,
        class_names=classes,
        title=model_name,
        save_path=matrix_path
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {auc:.4f}"
    )

    shutil.rmtree(
        temp_dir
    )

# =========================================================
# EVALUATE 3 LABEL MODEL
# =========================================================

evaluate_model(
    model_name="EfficientNetB3_3_Label_Health_vs_Sick_vs_TB",
    model_path=MULTICLASS_MODEL,
    classes=[
        "health",
        "sick",
        "tb"
    ]
)

# =========================================================
# EVALUATE NORMAL VS SICK
# =========================================================

evaluate_model(
    model_name="EfficientNetB3_2_Label_Normal_vs_Sick",
    model_path=NORMAL_VS_SICK_MODEL,
    classes=[
        "health",
        "sick"
    ]
)

# =========================================================
# EVALUATE SICK VS TB
# =========================================================

evaluate_model(
    model_name="EfficientNetB3_2_Label_Sick_vs_TB",
    model_path=SICK_VS_TB_MODEL,
    classes=[
        "sick",
        "tb"
    ]
)

print("\n" + "=" * 70)
print("ALL EFFICIENTNET-B3 EVALUATIONS COMPLETED")
print("=" * 70)

print(f"\nSaved To:\n{OUTPUT_DIR}")