import os
import cv2
import numpy as np
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import (
    Dataset,
    DataLoader,
    Subset,
    WeightedRandomSampler
)

from sklearn.model_selection import train_test_split

# ============================================================
# FIX RANDOMNESS
# ============================================================

SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

np.random.seed(SEED)
random.seed(SEED)

torch.backends.cudnn.benchmark = True

# ============================================================
# CONFIG
# ============================================================

IMAGE_DIR = (
    r"G:\Ai-CDD\data\preprocessed"
    r"\segmentation\images_stage1_split"
)

MASK_DIR = (
    r"G:\Ai-CDD\data\preprocessed"
    r"\segmentation\masks_split"
)

CHECKPOINT_DIR = (
    r"G:\Ai-CDD\checkpoints"
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

IMG_SIZE = 256

BATCH_SIZE = 4

EPOCHS = 60

LR = 1e-4

VAL_SPLIT = 0.2

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"\nUsing Device: {DEVICE}")

# ============================================================
# DATASET
# ============================================================

class TBSegmentationDataset(Dataset):

    def __init__(self, image_dir, mask_dir):

        self.samples = []

        healthy_image_dir = os.path.join(
            image_dir,
            "healthy"
        )

        healthy_mask_dir = os.path.join(
            mask_dir,
            "healthy"
        )

        tb_image_dir = os.path.join(
            image_dir,
            "tb"
        )

        tb_mask_dir = os.path.join(
            mask_dir,
            "tb"
        )

        healthy_images = sorted(
            os.listdir(healthy_image_dir)
        )

        healthy_count = 0

        for img_name in healthy_images:

            image_path = os.path.join(
                healthy_image_dir,
                img_name
            )

            mask_path = os.path.join(
                healthy_mask_dir,
                img_name
            )

            if not os.path.exists(mask_path):
                continue

            image = cv2.imread(
                image_path,
                cv2.IMREAD_GRAYSCALE
            )

            mask = cv2.imread(
                mask_path,
                cv2.IMREAD_GRAYSCALE
            )

            if image is None or mask is None:
                continue

            self.samples.append(
                (
                    image_path,
                    mask_path,
                    0
                )
            )

            healthy_count += 1

        tb_images = sorted(
            os.listdir(tb_image_dir)
        )

        tb_count = 0

        for img_name in tb_images:

            image_path = os.path.join(
                tb_image_dir,
                img_name
            )

            mask_path = os.path.join(
                tb_mask_dir,
                img_name
            )

            if not os.path.exists(mask_path):
                continue

            image = cv2.imread(
                image_path,
                cv2.IMREAD_GRAYSCALE
            )

            mask = cv2.imread(
                mask_path,
                cv2.IMREAD_GRAYSCALE
            )

            if image is None or mask is None:
                continue

            self.samples.append(
                (
                    image_path,
                    mask_path,
                    1
                )
            )

            tb_count += 1

        print("\n===================================")
        print(f"Healthy Samples : {healthy_count}")
        print(f"TB Samples      : {tb_count}")
        print(f"Total Samples   : {len(self.samples)}")
        print("===================================")

    def __len__(self):

        return len(self.samples)

    # ========================================================
    # STRONG TB AUGMENTATION
    # ========================================================

    def augment_tb(self, image, mask):

        # Horizontal flip
        if random.random() > 0.5:

            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        # Vertical flip
        if random.random() > 0.85:

            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)

        # Rotation
        if random.random() > 0.3:

            angle = random.randint(-20, 20)

            h, w = image.shape

            M = cv2.getRotationMatrix2D(
                (w // 2, h // 2),
                angle,
                1.0
            )

            image = cv2.warpAffine(
                image,
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )

            mask = cv2.warpAffine(
                mask,
                M,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_REFLECT
            )

        # Brightness / Contrast
        if random.random() > 0.4:

            alpha = random.uniform(0.85, 1.20)
            beta = random.randint(-15, 15)

            image = cv2.convertScaleAbs(
                image,
                alpha=alpha,
                beta=beta
            )

        # Gaussian Blur
        if random.random() > 0.75:

            image = cv2.GaussianBlur(
                image,
                (3, 3),
                0
            )

        # CLAHE
        if random.random() > 0.5:

            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8)
            )

            image = clahe.apply(image)

        return image, mask

    # ========================================================
    # LIGHT HEALTHY AUGMENTATION
    # ========================================================

    def augment_healthy(self, image, mask):

        # Only mild augmentation
        # Avoid over-distorting healthy lungs

        if random.random() > 0.5:

            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        if random.random() > 0.6:

            angle = random.randint(-8, 8)

            h, w = image.shape

            M = cv2.getRotationMatrix2D(
                (w // 2, h // 2),
                angle,
                1.0
            )

            image = cv2.warpAffine(
                image,
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )

            mask = cv2.warpAffine(
                mask,
                M,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_REFLECT
            )

        # Small brightness variation
        if random.random() > 0.5:

            alpha = random.uniform(0.95, 1.08)
            beta = random.randint(-8, 8)

            image = cv2.convertScaleAbs(
                image,
                alpha=alpha,
                beta=beta
            )

        return image, mask

    def __getitem__(self, idx):

        image_path, mask_path, label = self.samples[idx]

        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        image = cv2.resize(
            image,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_NEAREST
        )

        # ====================================================
        # AUGMENTATION
        # ====================================================

        if label == 1:

            image, mask = self.augment_tb(
                image,
                mask
            )

        else:

            image, mask = self.augment_healthy(
                image,
                mask
            )

        # ====================================================
        # NORMALIZATION
        # ====================================================

        image = image.astype(np.float32) / 255.0

        image = (
            image - image.mean()
        ) / (
            image.std() + 1e-8
        )

        mask = (
            mask > 127
        ).astype(np.float32)

        image = np.expand_dims(
            image,
            axis=0
        )

        mask = np.expand_dims(
            mask,
            axis=0
        )

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        mask = torch.tensor(
            mask,
            dtype=torch.float32
        )

        return image, mask

# ============================================================
# MODEL BLOCKS
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.conv(x)

# ============================================================
# ATTENTION BLOCK
# ============================================================

class AttentionBlock(nn.Module):

    def __init__(
        self,
        g_channels,
        x_channels,
        inter_channels
    ):

        super().__init__()

        self.W_g = nn.Sequential(

            nn.Conv2d(
                g_channels,
                inter_channels,
                kernel_size=1
            ),

            nn.BatchNorm2d(inter_channels)
        )

        self.W_x = nn.Sequential(

            nn.Conv2d(
                x_channels,
                inter_channels,
                kernel_size=1
            ),

            nn.BatchNorm2d(inter_channels)
        )

        self.psi = nn.Sequential(

            nn.Conv2d(
                inter_channels,
                1,
                kernel_size=1
            ),

            nn.BatchNorm2d(1),

            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):

        g1 = self.W_g(g)

        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)

        psi = self.psi(psi)

        return x * psi

# ============================================================
# ATTENTION U-NET
# ============================================================

class AttentionUNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.enc1 = DoubleConv(1, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)

        self.bottleneck = DoubleConv(256, 512)

        self.up4 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.att4 = AttentionBlock(
            256,
            256,
            128
        )

        self.dec4 = DoubleConv(
            512,
            256
        )

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.att3 = AttentionBlock(
            128,
            128,
            64
        )

        self.dec3 = DoubleConv(
            256,
            128
        )

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.att2 = AttentionBlock(
            64,
            64,
            32
        )

        self.dec2 = DoubleConv(
            128,
            64
        )

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.att1 = AttentionBlock(
            32,
            32,
            16
        )

        self.dec1 = DoubleConv(
            64,
            32
        )

        self.final = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(self.pool(e1))

        e3 = self.enc3(self.pool(e2))

        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)

        e4 = self.att4(d4, e4)

        d4 = torch.cat([d4, e4], dim=1)

        d4 = self.dec4(d4)

        d3 = self.up3(d4)

        e3 = self.att3(d3, e3)

        d3 = torch.cat([d3, e3], dim=1)

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        e2 = self.att2(d2, e2)

        d2 = torch.cat([d2, e2], dim=1)

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        e1 = self.att1(d1, e1)

        d1 = torch.cat([d1, e1], dim=1)

        d1 = self.dec1(d1)

        return self.final(d1)

# ============================================================
# LOSS
# ============================================================

bce_loss = nn.BCEWithLogitsLoss()

def dice_loss(pred, target, smooth=1):

    pred = torch.sigmoid(pred)

    intersection = (pred * target).sum()

    dice = (
        (2.0 * intersection + smooth)
        /
        (pred.sum() + target.sum() + smooth)
    )

    return 1 - dice

def combined_loss(pred, target):

    return (
        0.5 * bce_loss(pred, target)
        +
        0.5 * dice_loss(pred, target)
    )

def dice_score(pred, target, smooth=1):

    pred = torch.sigmoid(pred)

    pred = (pred > 0.5).float()

    intersection = (pred * target).sum()

    dice = (
        (2.0 * intersection + smooth)
        /
        (pred.sum() + target.sum() + smooth)
    )

    return dice.item()

# ============================================================
# MAIN
# ============================================================

def main():

    dataset = TBSegmentationDataset(
        IMAGE_DIR,
        MASK_DIR
    )

    labels = []

    for sample in dataset.samples:
        labels.append(sample[2])

    labels = np.array(labels)

    indices = np.arange(len(dataset))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=VAL_SPLIT,
        stratify=labels,
        random_state=SEED,
        shuffle=True
    )

    train_dataset = Subset(
        dataset,
        train_indices
    )

    val_dataset = Subset(
        dataset,
        val_indices
    )

    # ========================================================
    # BALANCED SAMPLER
    # ========================================================

    train_labels = labels[train_indices]

    class_counts = np.bincount(train_labels)

    class_weights = 1.0 / class_counts

    sample_weights = [
        class_weights[label]
        for label in train_labels
    ]

    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    print(f"\nTrain Samples: {len(train_dataset)}")
    print(f"Validation Samples: {len(val_dataset)}")

    model = AttentionUNet().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    USE_AMP = DEVICE == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP
    )

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        "attention_unet_checkpoint.pth"
    )

    best_model_path = os.path.join(
        CHECKPOINT_DIR,
        "best_attention_unet.pth"
    )

    start_epoch = 0

    best_val_dice = 0

    if os.path.exists(checkpoint_path):

        print("\nLoading checkpoint...")

        checkpoint = torch.load(
            checkpoint_path,
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

        best_val_dice = checkpoint["best_val_dice"]

        print(f"\nResuming from epoch {start_epoch}")

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(start_epoch, EPOCHS):

        model.train()

        train_loss = 0

        loop = tqdm(train_loader)

        for images, masks in loop:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            masks = masks.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad()

            with torch.amp.autocast(
                device_type="cuda",
                enabled=USE_AMP
            ):

                outputs = model(images)

                loss = combined_loss(
                    outputs,
                    masks
                )

            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            scaler.step(optimizer)

            scaler.update()

            train_loss += loss.item()

            loop.set_description(
                f"Epoch [{epoch+1}/{EPOCHS}]"
            )

            loop.set_postfix(
                loss=loss.item()
            )

        model.eval()

        val_loss = 0
        val_dice = 0

        with torch.no_grad():

            for images, masks in val_loader:

                images = images.to(
                    DEVICE,
                    non_blocking=True
                )

                masks = masks.to(
                    DEVICE,
                    non_blocking=True
                )

                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=USE_AMP
                ):

                    outputs = model(images)

                    loss = combined_loss(
                        outputs,
                        masks
                    )

                val_loss += loss.item()

                val_dice += dice_score(
                    outputs,
                    masks
                )

        avg_train_loss = (
            train_loss / len(train_loader)
        )

        avg_val_loss = (
            val_loss / len(val_loader)
        )

        avg_val_dice = (
            val_dice / len(val_loader)
        )

        scheduler.step()

        print(f"\nTrain Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Val Dice: {avg_val_dice:.4f}")

        checkpoint = {

            "epoch": epoch,

            "model_state_dict": model.state_dict(),

            "optimizer_state_dict": optimizer.state_dict(),

            "scheduler_state_dict": scheduler.state_dict(),

            "best_val_dice": best_val_dice
        }

        torch.save(
            checkpoint,
            checkpoint_path
        )

        if avg_val_dice > best_val_dice:

            best_val_dice = avg_val_dice

            torch.save(
                model.state_dict(),
                best_model_path
            )

            print("\nBest model updated.")

    print("\nTraining Completed.")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()