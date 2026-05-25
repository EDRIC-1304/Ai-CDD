import os
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================================================
# CONFIG
# =========================================================
DATA_DIR = r"G:\Ai-CDD\data\segmentation\preprocessed\stage1"

IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

CHECKPOINT_DIR = "segmentation_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

LATEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")

# =========================================================
# DATASET
# =========================================================
class SegmentationDataset(Dataset):

    def __init__(self, image_dir, mask_dir):

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        valid_ext = (".png", ".jpg", ".jpeg", ".bmp")

        self.files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(valid_ext)
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        file = self.files[idx]

        image_path = os.path.join(self.image_dir, file)
        mask_path = os.path.join(self.mask_dir, file)

        # -------------------------------------------------
        # READ IMAGE
        # -------------------------------------------------
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        image = image.astype(np.float32) / 255.0

        # -------------------------------------------------
        # READ MASK
        # -------------------------------------------------
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        mask = (mask > 0).astype(np.float32)

        # -------------------------------------------------
        # CHANNEL DIMENSION
        # -------------------------------------------------
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32)
        )

# =========================================================
# DOUBLE CONV BLOCK
# =========================================================
class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

# =========================================================
# U-NET
# =========================================================
class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder
        self.down1 = DoubleConv(1, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv1 = DoubleConv(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv2 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = DoubleConv(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        # Output
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):

        # Encoder
        d1 = self.down1(x)
        p1 = self.pool1(d1)

        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        d3 = self.down3(p2)
        p3 = self.pool3(d3)

        d4 = self.down4(p3)
        p4 = self.pool4(d4)

        # Bottleneck
        bn = self.bottleneck(p4)

        # Decoder
        u1 = self.up1(bn)
        u1 = torch.cat([u1, d4], dim=1)
        u1 = self.conv1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d3], dim=1)
        u2 = self.conv2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, d2], dim=1)
        u3 = self.conv3(u3)

        u4 = self.up4(u3)
        u4 = torch.cat([u4, d1], dim=1)
        u4 = self.conv4(u4)

        return self.out(u4)

# =========================================================
# DICE LOSS
# =========================================================
def dice_loss(pred, target, smooth=1e-6):

    pred = torch.sigmoid(pred)

    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()

    dice = (
        (2. * intersection + smooth)
        /
        (pred.sum() + target.sum() + smooth)
    )

    return 1 - dice

# =========================================================
# COMBINED LOSS
# =========================================================
bce_loss = nn.BCEWithLogitsLoss()

def combined_loss(pred, target):

    bce = bce_loss(pred, target)
    dice = dice_loss(pred, target)

    return bce + dice

# =========================================================
# DATA LOADERS
# =========================================================
train_dataset = SegmentationDataset(
    image_dir=os.path.join(DATA_DIR, "train", "images"),
    mask_dir=os.path.join(DATA_DIR, "train", "masks")
)

val_dataset = SegmentationDataset(
    image_dir=os.path.join(DATA_DIR, "val", "images"),
    mask_dir=os.path.join(DATA_DIR, "val", "masks")
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# =========================================================
# MODEL
# =========================================================
model = UNet().to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

# =========================================================
# RESUME TRAINING
# =========================================================
start_epoch = 0
best_val_loss = float("inf")

if os.path.exists(LATEST_CHECKPOINT):

    print(f"\n🔄 Resuming from checkpoint: {LATEST_CHECKPOINT}")

    checkpoint = torch.load(LATEST_CHECKPOINT, map_location=DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"]
    best_val_loss = checkpoint["best_val_loss"]

    print(f"✅ Resumed from Epoch {start_epoch}")

# =========================================================
# TRAINING LOOP
# =========================================================
for epoch in range(start_epoch, EPOCHS):

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------
    model.train()

    train_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for images, masks in loop:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = combined_loss(outputs, masks)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        loop.set_postfix(loss=loss.item())

    train_loss /= len(train_loader)

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------
    model.eval()

    val_loss = 0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)

            loss = combined_loss(outputs, masks)

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"\nEpoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f}"
    )

    # -----------------------------------------------------
    # SAVE LATEST CHECKPOINT
    # -----------------------------------------------------
    latest_checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
    }

    torch.save(latest_checkpoint, LATEST_CHECKPOINT)

    # -----------------------------------------------------
    # SAVE BEST MODEL
    # -----------------------------------------------------
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        }

        save_path = os.path.join(
            CHECKPOINT_DIR,
            "best_unet.pth"
        )

        torch.save(checkpoint, save_path)

        print(f"✅ Best model saved: {save_path}")

# =========================================================
# FINAL SAVE
# =========================================================
torch.save(
    model.state_dict(),
    os.path.join(CHECKPOINT_DIR, "final_unet.pth")
)

print("\n🎯 Training Completed")