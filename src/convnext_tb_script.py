# =========================================================
# CONVNEXT TB vs SICK CLASSIFICATION
# =========================================================
#
# CLASSES:
# 0 -> sick
# 1 -> tb
#
# IMPORTANT:
# THIS SCRIPT IGNORES:
# - health
# - NORMAL
# - TUBERCULOSIS
#
# PURPOSE:
# LEARN DIFFERENCE BETWEEN:
# - GENERIC ABNORMAL LUNG DISEASE
# - TUBERCULOSIS
#
# INPUT:
# ROI EXTRACTED IMAGES (STAGE 4)
#
# =========================================================

import os
import time
import warnings
import shutil
import tempfile

import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =========================================================
# TORCH
# =========================================================

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision import models

# =========================================================
# SKLEARN
# =========================================================

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# =========================================================
# CONFIG
# =========================================================

SEED = 42

BATCH_SIZE = 16

NUM_EPOCHS = 50

LEARNING_RATE = 1e-4

IMG_SIZE = 224

NUM_WORKERS = 0

NUM_CLASSES = 2

VALID_CLASSES = [
    "sick",
    "tb"
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing Device: {DEVICE}")

# =========================================================
# DATA PATHS
# =========================================================

BASE_DATASET = r"G:\Ai-CDD\data\preprocessed\stage4"

TRAIN_DIR = os.path.join(
    BASE_DATASET,
    "train"
)

VAL_DIR = os.path.join(
    BASE_DATASET,
    "val"
)

TEST_DIR = os.path.join(
    BASE_DATASET,
    "internal_test"
)

# =========================================================
# CHECKPOINTS
# =========================================================

CHECKPOINT_DIR = r"G:\Ai-CDD\classification_checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

LATEST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "latest_checkpoint_convnext_tb_vs_sick.pth"
)

BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_convnext_tb_vs_sick.pth"
)

CONF_MATRIX_PATH = os.path.join(
    CHECKPOINT_DIR,
    "confusion_matrix_convnext_tb_vs_sick.png"
)

# =========================================================
# RANDOM SEED
# =========================================================

torch.manual_seed(SEED)
np.random.seed(SEED)

# =========================================================
# CREATE CLEAN TEMP DATASET
# =========================================================

TEMP_DATASET = tempfile.mkdtemp()

print("\nCreating clean temporary dataset...")

def create_filtered_dataset(src_dir, dst_dir):

    os.makedirs(dst_dir, exist_ok=True)

    for class_name in VALID_CLASSES:

        src_class = os.path.join(
            src_dir,
            class_name
        )

        dst_class = os.path.join(
            dst_dir,
            class_name
        )

        if not os.path.exists(src_class):

            print(f"❌ Missing class: {src_class}")
            continue

        shutil.copytree(
            src_class,
            dst_class
        )

# =========================================================
# FILTER TRAIN
# =========================================================

TEMP_TRAIN = os.path.join(
    TEMP_DATASET,
    "train"
)

create_filtered_dataset(
    TRAIN_DIR,
    TEMP_TRAIN
)

# =========================================================
# FILTER VAL
# =========================================================

TEMP_VAL = os.path.join(
    TEMP_DATASET,
    "val"
)

create_filtered_dataset(
    VAL_DIR,
    TEMP_VAL
)

# =========================================================
# FILTER TEST
# =========================================================

TEMP_TEST = os.path.join(
    TEMP_DATASET,
    "internal_test"
)

create_filtered_dataset(
    TEST_DIR,
    TEMP_TEST
)

print("✅ Temporary clean dataset ready")

# =========================================================
# TRANSFORMS
# =========================================================

train_transform = transforms.Compose([

    transforms.Grayscale(num_output_channels=3),

    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(5),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_test_transform = transforms.Compose([

    transforms.Grayscale(num_output_channels=3),

    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================================
# DATASETS
# =========================================================

train_dataset = datasets.ImageFolder(
    root=TEMP_TRAIN,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=TEMP_VAL,
    transform=val_test_transform
)

test_dataset = datasets.ImageFolder(
    root=TEMP_TEST,
    transform=val_test_transform
)

print("\nClass Mapping:")
print(train_dataset.class_to_idx)

print(f"\nTrain Samples: {len(train_dataset)}")
print(f"Val Samples: {len(val_dataset)}")
print(f"Test Samples: {len(test_dataset)}")

# =========================================================
# DATALOADERS
# =========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

# =========================================================
# MODEL
# =========================================================

model = models.convnext_tiny(
    weights=models.ConvNeXt_Tiny_Weights.DEFAULT
)

# =========================================================
# MODIFY CLASSIFIER
# =========================================================

model.classifier[2] = nn.Linear(
    in_features=768,
    out_features=NUM_CLASSES
)

model = model.to(DEVICE)

print("\n✅ ConvNeXt Tiny Loaded")

# =========================================================
# LOSS
# =========================================================

criterion = nn.CrossEntropyLoss()

# =========================================================
# OPTIMIZER
# =========================================================

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

# =========================================================
# LR SCHEDULER
# =========================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)

# =========================================================
# RESUME CHECKPOINT
# =========================================================

start_epoch = 0

best_val_loss = float("inf")

if os.path.exists(LATEST_CHECKPOINT):

    print("\nLoading latest checkpoint...")

    checkpoint = torch.load(
        LATEST_CHECKPOINT,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    start_epoch = checkpoint["epoch"] + 1

    best_val_loss = checkpoint["best_val_loss"]

    print(f"✅ Resuming from epoch {start_epoch}")

# =========================================================
# TRAIN FUNCTION
# =========================================================

def train_one_epoch():

    model.train()

    running_loss = 0

    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    epoch_loss = running_loss / len(train_loader)

    epoch_acc = correct / total

    return epoch_loss, epoch_acc

# =========================================================
# VALIDATION FUNCTION
# =========================================================

def validate():

    model.eval()

    running_loss = 0

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    val_loss = running_loss / len(val_loader)

    val_acc = correct / total

    return val_loss, val_acc

# =========================================================
# TRAINING LOOP
# =========================================================

print("\n🚀 Starting Training...\n")

for epoch in range(start_epoch, NUM_EPOCHS):

    start_time = time.time()

    train_loss, train_acc = train_one_epoch()

    val_loss, val_acc = validate()

    scheduler.step(val_loss)

    checkpoint = {

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "best_val_loss":
            best_val_loss
    }

    torch.save(
        checkpoint,
        LATEST_CHECKPOINT
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

        print("✅ Best model updated")

    elapsed = time.time() - start_time

    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Time: {elapsed:.2f}s"
    )

print("\n🎯 Training Complete")

# =========================================================
# LOAD BEST MODEL
# =========================================================

model.load_state_dict(
    torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE
    )
)

# =========================================================
# TEST EVALUATION
# =========================================================

print("\n📊 Running Internal Test Evaluation...\n")

model.eval()

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

        _, preds = torch.max(
            outputs,
            1
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

acc = accuracy_score(
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

print("=" * 60)
print("TB vs SICK INTERNAL TEST RESULTS")
print("=" * 60)

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=VALID_CLASSES
    )
)

# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    all_labels,
    all_preds
)

print("\nConfusion Matrix:\n")
print(cm)

# =========================================================
# SAVE CONFUSION MATRIX
# =========================================================

plt.figure(figsize=(8, 8))

plt.imshow(cm)

plt.title("ConvNeXt TB vs Sick Confusion Matrix")

plt.colorbar()

tick_marks = np.arange(len(VALID_CLASSES))

plt.xticks(
    tick_marks,
    VALID_CLASSES
)

plt.yticks(
    tick_marks,
    VALID_CLASSES
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(CONF_MATRIX_PATH)

print(f"\n✅ Confusion matrix saved:")
print(CONF_MATRIX_PATH)

print("\n🎯 TB vs Sick classification completed successfully")