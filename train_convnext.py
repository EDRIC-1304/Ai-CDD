import os
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score

import timm

from log import log

# =========================================================
# CONFIG
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_DIR = r"G:\Ai-CDD\data\preprocessed\stage4\train"
VAL_DIR = r"G:\Ai-CDD\data\preprocessed\stage4\val"

CHECKPOINT_DIR = r"G:\Ai-CDD\convnext_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4

CLASS_NAMES = ["NORMAL", "TUBERCULOSIS"]

# =========================================================
# DATASET
# =========================================================
class XrayDataset(Dataset):

    def __init__(self, root_dir):

        self.samples = []

        for label, cls in enumerate(CLASS_NAMES):

            class_dir = os.path.join(root_dir, cls)

            files = [
                f for f in os.listdir(class_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]

            for file in files:
                self.samples.append((
                    os.path.join(class_dir, file),
                    label
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        img = img.astype(np.float32) / 255.0

        # ConvNeXt expects 3 channels
        img = np.stack([img, img, img], axis=0)

        img = torch.tensor(img, dtype=torch.float32)

        return img, torch.tensor(label, dtype=torch.long)

# =========================================================
# MODEL
# ========================================================= 
class ConvNextTB(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            "convnext_tiny",
            pretrained=True,
            num_classes=2
        )

    def forward(self, x):
        return self.backbone(x)

# =========================================================
# VALIDATION
# =========================================================
def validate(model, loader, criterion):

    model.eval()

    total_loss = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)

    return total_loss / len(loader), acc

# =========================================================
# TRAIN
# =========================================================
def train():

    train_dataset = XrayDataset(TRAIN_DIR)
    val_dataset = XrayDataset(VAL_DIR)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    model = ConvNextTB().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=3,
        factor=0.5
    )

    best_val_acc = 0

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0

        train_preds = []
        train_labels = []

        loop = tqdm(train_loader)

        for images, labels in loop:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            train_preds.extend(preds.detach().cpu().numpy())
            train_labels.extend(labels.detach().cpu().numpy())

            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item())

        train_acc = accuracy_score(train_labels, train_preds)

        val_loss, val_acc = validate(model, val_loader, criterion)

        scheduler.step(val_loss)

        log(
            f"Epoch {epoch+1} | "
            f"Train Loss: {running_loss/len(train_loader):.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        latest_path = os.path.join(
            CHECKPOINT_DIR,
            "latest_convnext.pth"
        )

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_acc": val_acc
        }, latest_path)

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            best_path = os.path.join(
                CHECKPOINT_DIR,
                "best_convnext.pth"
            )

            torch.save(model.state_dict(), best_path)

            log(f"✅ New Best Model Saved | Val Acc: {val_acc:.4f}")

    print("\n🎯 ConvNeXt training complete")

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    train()