# =========================================================
# CONVNEXT BINARY CLASSIFICATION
# =========================================================
#
# TASK:
# health vs sick
#
# INPUT:
# ROI EXTRACTED IMAGES (STAGE 4)
#
# =========================================================

import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

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
    "latest_checkpoint.pth"
)

BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_convnext_binary.pth"
)

CONF_MATRIX_PATH = os.path.join(
    CHECKPOINT_DIR,
    "confusion_matrix.png"
)

# =========================================================
# RANDOM SEED
# =========================================================

torch.manual_seed(SEED)
np.random.seed(SEED)

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
    root=TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=VAL_DIR,
    transform=val_test_transform
)

test_dataset = datasets.ImageFolder(
    root=TEST_DIR,
    transform=val_test_transform
)

# =========================================================
# KEEP ONLY health AND sick
# =========================================================

VALID_CLASSES = ["health", "sick"]

NEW_CLASS_TO_IDX = {
    "health": 0,
    "sick": 1
}

def filter_dataset(dataset):

    filtered_samples = []

    for path, _ in dataset.samples:

        class_name = Path(path).parent.name

        if class_name in VALID_CLASSES:

            filtered_samples.append(
                (
                    path,
                    NEW_CLASS_TO_IDX[class_name]
                )
            )

    dataset.samples = filtered_samples

    dataset.imgs = filtered_samples

    dataset.targets = [
        label for _, label in filtered_samples
    ]

    dataset.class_to_idx = NEW_CLASS_TO_IDX

    dataset.classes = VALID_CLASSES

    return dataset

train_dataset = filter_dataset(train_dataset)
val_dataset = filter_dataset(val_dataset)
test_dataset = filter_dataset(test_dataset)

print("\nFinal Class Mapping:")
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

# Replace classifier
model.classifier[2] = nn.Linear(
    in_features=768,
    out_features=2
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

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

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

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    val_loss = running_loss / len(val_loader)

    val_acc = correct / total

    return val_loss, val_acc

# =========================================================
# TRAIN LOOP
# =========================================================

print("\n🚀 Starting Training...\n")

for epoch in range(start_epoch, NUM_EPOCHS):

    start_time = time.time()

    train_loss, train_acc = train_one_epoch()

    val_loss, val_acc = validate()

    scheduler.step(val_loss)

    # =====================================================
    # SAVE LATEST CHECKPOINT
    # =====================================================

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

    # =====================================================
    # SAVE BEST MODEL
    # =====================================================

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

        _, preds = torch.max(outputs, 1)

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

print("=" * 50)
print("INTERNAL TEST RESULTS")
print("=" * 50)

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
        target_names=[
            "health",
            "sick"
        ]
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

plt.figure(figsize=(6, 6))

plt.imshow(cm)

plt.title("Confusion Matrix")

plt.colorbar()

classes = ["health", "sick"]

tick_marks = np.arange(len(classes))

plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

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

print("\n🎯 Binary classification completed successfully")