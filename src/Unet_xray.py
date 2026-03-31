# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# import re

# IMG_SIZE = 256

# IMAGE_DIR = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/raw/train/preprocessed_xray_images"
# MASK_DIR  = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/raw/train/xray-mask"

# # -----------------------------
# # LOAD DATA
# # -----------------------------

# def extract_id(filename):
#     match = re.search(r'\d+', filename)
#     return match.group() if match else None


# def load_data(image_dir, mask_dir):
#     images = []
#     masks = []

#     image_files = os.listdir(image_dir)
#     mask_files  = os.listdir(mask_dir)

#     image_dict = {}
#     mask_dict = {}

#     for file in image_files:
#         img_id = extract_id(file)
#         if img_id:
#             image_dict[img_id] = file

#     for file in mask_files:
#         mask_id = extract_id(file)
#         if mask_id:
#             mask_dict[mask_id] = file

#     common_ids = sorted(set(image_dict.keys()) & set(mask_dict.keys()))

#     print(f"✅ Total matched pairs: {len(common_ids)}")

#     if len(common_ids) == 0:
#         raise ValueError("❌ No matching image-mask pairs found!")

#     for idx in common_ids:
#         img_path = os.path.join(image_dir, image_dict[idx])
#         mask_path = os.path.join(mask_dir, mask_dict[idx])

#         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#         if img is None:
#             continue

#         img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
#         img = img.astype(np.float32) / 255.0
#         img = np.expand_dims(img, axis=-1)

#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         if mask is None:
#             continue

#         mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
#         mask = (mask > 127).astype(np.float32)
#         mask = np.expand_dims(mask, axis=-1)

#         images.append(img)
#         masks.append(mask)

#     print(f"✅ Loaded images: {len(images)}")

#     return np.array(images), np.array(masks)


# # -----------------------------
# # LOAD + SPLIT
# # -----------------------------
# X, y = load_data(IMAGE_DIR, MASK_DIR)

# X_train, X_val, y_train, y_val = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# print("Train:", X_train.shape)
# print("Val:", X_val.shape)

# # -----------------------------
# # AUGMENTATION (TRAIN ONLY)
# # -----------------------------
# data_augmentation = tf.keras.Sequential([
#     tf.keras.layers.RandomFlip("horizontal"),
#     tf.keras.layers.RandomRotation(0.05)
# ])

# # Apply augmentation ONLY on training images
# def augment(x, y):
#     x = data_augmentation(x)
#     return x, y

# train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
# train_dataset = train_dataset.map(lambda x, y: augment(x, y),
#                                   num_parallel_calls=tf.data.AUTOTUNE)
# train_dataset = train_dataset.batch(8).prefetch(tf.data.AUTOTUNE)

# val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
# val_dataset = val_dataset.batch(8).prefetch(tf.data.AUTOTUNE)

# # -----------------------------
# # SIMPLE U-NET MODEL
# # -----------------------------
# def conv_block(x, filters):
#     x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.ReLU()(x)

#     x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.ReLU()(x)
#     return x

# def build_unet(input_shape=(256,256,1)):
#     inputs = tf.keras.Input(input_shape)

#     c1 = conv_block(inputs, 32)
#     p1 = tf.keras.layers.MaxPooling2D()(c1)

#     c2 = conv_block(p1, 64)
#     p2 = tf.keras.layers.MaxPooling2D()(c2)

#     c3 = conv_block(p2, 128)
#     p3 = tf.keras.layers.MaxPooling2D()(c3)

#     c4 = conv_block(p3, 256)
#     p4 = tf.keras.layers.MaxPooling2D()(c4)

#     bn = conv_block(p4, 512)

#     u1 = tf.keras.layers.UpSampling2D()(bn)
#     u1 = tf.keras.layers.Concatenate()([u1, c4])
#     c5 = conv_block(u1, 256)

#     u2 = tf.keras.layers.UpSampling2D()(c5)
#     u2 = tf.keras.layers.Concatenate()([u2, c3])
#     c6 = conv_block(u2, 128)

#     u3 = tf.keras.layers.UpSampling2D()(c6)
#     u3 = tf.keras.layers.Concatenate()([u3, c2])
#     c7 = conv_block(u3, 64)

#     u4 = tf.keras.layers.UpSampling2D()(c7)
#     u4 = tf.keras.layers.Concatenate()([u4, c1])
#     c8 = conv_block(u4, 32)

#     outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(c8)

#     return tf.keras.Model(inputs, outputs)

# model = build_unet()

# # -----------------------------
# # LOSS + METRICS
# # -----------------------------
# def dice_coef(y_true, y_pred):
#     smooth = 1e-6
#     intersection = tf.reduce_sum(y_true * y_pred)
#     union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
#     return (2. * intersection + smooth) / (union + smooth)

# def dice_loss(y_true, y_pred):
#     return 1 - dice_coef(y_true, y_pred)

# def iou_metric(y_true, y_pred):
#     smooth = 1e-6
#     intersection = tf.reduce_sum(y_true * y_pred)
#     union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
#     return (intersection + smooth) / (union + smooth)

# def combined_loss(y_true, y_pred):
#     bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
#     return bce + dice_loss(y_true, y_pred)

# # -----------------------------
# # COMPILE
# # -----------------------------
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(1e-4),
#     loss=combined_loss,
#     metrics=[dice_coef, iou_metric]
# )

# model.summary()

# # -----------------------------
# # TRAIN
# # -----------------------------
# history = model.fit(
#     train_dataset,
#     validation_data=val_dataset,
#     epochs=50
# )

# # -----------------------------
# # SAVE MODEL
# # -----------------------------
# model.save("xray_unet_lung.h5")

# print("✅ U-Net training completed")




import os
import cv2
import re
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 50
CHECKPOINT_DIR = "checkpoints"
LOG_FILE = "training_log.txt"

IMAGE_DIR = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/raw/train/preprocessed_xray_images"
MASK_DIR  = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/raw/train/xray-mask"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -----------------------------
# GPU CHECK
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# -----------------------------
# LOGGING
# -----------------------------
def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

# -----------------------------
# DATA LOADING
# -----------------------------
def extract_id(filename):
    match = re.search(r'\d+', filename)
    return match.group() if match else None


def load_data(image_dir, mask_dir):
    images = []
    masks = []

    image_files = os.listdir(image_dir)
    mask_files  = os.listdir(mask_dir)

    image_dict = {}
    mask_dict = {}

    for file in image_files:
        img_id = extract_id(file)
        if img_id:
            image_dict[img_id] = file

    for file in mask_files:
        mask_id = extract_id(file)
        if mask_id:
            mask_dict[mask_id] = file

    common_ids = sorted(set(image_dict.keys()) & set(mask_dict.keys()))

    log(f"Total matched pairs: {len(common_ids)}")

    if len(common_ids) == 0:
        raise ValueError("No matching pairs found")

    for idx in tqdm(common_ids):
        img_path = os.path.join(image_dir, image_dict[idx])
        mask_path = os.path.join(mask_dir, mask_dict[idx])

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        mask = (mask > 127).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)


# -----------------------------
# DATASET
# -----------------------------
class XrayDataset(Dataset):
    def __init__(self, images, masks, augment=False):
        self.images = images
        self.masks = masks
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]

        if self.augment:
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=2).copy()
                mask = np.flip(mask, axis=2).copy()

        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


# -----------------------------
# LOAD DATA
# -----------------------------
X, y = load_data(IMAGE_DIR, MASK_DIR)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_loader = DataLoader(
    XrayDataset(X_train, y_train, augment=True),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    XrayDataset(X_val, y_val, augment=False),
    batch_size=BATCH_SIZE
)

# -----------------------------
# MODEL (U-NET)
# -----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.c1 = ConvBlock(1, 32)
        self.p1 = nn.MaxPool2d(2)

        self.c2 = ConvBlock(32, 64)
        self.p2 = nn.MaxPool2d(2)

        self.c3 = ConvBlock(64, 128)
        self.p3 = nn.MaxPool2d(2)

        self.c4 = ConvBlock(128, 256)
        self.p4 = nn.MaxPool2d(2)

        self.bn = ConvBlock(256, 512)

        self.u1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.c5 = ConvBlock(512, 256)

        self.u2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.c6 = ConvBlock(256, 128)

        self.u3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.c7 = ConvBlock(128, 64)

        self.u4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.c8 = ConvBlock(64, 32)

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(self.p1(c1))
        c3 = self.c3(self.p2(c2))
        c4 = self.c4(self.p3(c3))

        bn = self.bn(self.p4(c4))

        u1 = self.u1(bn)
        u1 = torch.cat([u1, c4], dim=1)
        c5 = self.c5(u1)

        u2 = self.u2(c5)
        u2 = torch.cat([u2, c3], dim=1)
        c6 = self.c6(u2)

        u3 = self.u3(c6)
        u3 = torch.cat([u3, c2], dim=1)
        c7 = self.c7(u3)

        u4 = self.u4(c7)
        u4 = torch.cat([u4, c1], dim=1)
        c8 = self.c8(u4)

        return torch.sigmoid(self.out(c8))


model = UNet().to(DEVICE)

# -----------------------------
# LOSS + METRICS
# -----------------------------
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    intersection = (y_true * y_pred).sum()
    return (2. * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)


def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + smooth) / (union + smooth)


bce_loss = nn.BCELoss()

def combined_loss(y_true, y_pred):
    return bce_loss(y_pred, y_true) + (1 - dice_coef(y_true, y_pred))


optimizer = optim.Adam(model.parameters(), lr=1e-4)

# -----------------------------
# CHECKPOINT LOAD
# -----------------------------
start_epoch = 0
best_loss = float("inf")

checkpoint_path = os.path.join(CHECKPOINT_DIR, "latest.pth")

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint["best_loss"]
    log(f"Resuming from epoch {start_epoch}")

# -----------------------------
# TRAIN LOOP
# -----------------------------
for epoch in range(start_epoch, EPOCHS):
    model.train()
    train_loss = 0

    for imgs, masks in train_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        preds = model(imgs)
        loss = combined_loss(masks, preds)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # VALIDATION
    model.eval()
    val_loss = 0
    dice_score = 0
    iou_score = 0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            preds = model(imgs)
            loss = combined_loss(masks, preds)

            val_loss += loss.item()
            dice_score += dice_coef(masks, preds).item()
            iou_score += iou_metric(masks, preds).item()

    val_loss /= len(val_loader)
    dice_score /= len(val_loader)
    iou_score /= len(val_loader)

    log(f"Epoch {epoch+1}/{EPOCHS}")
    log(f"Train Loss: {train_loss:.4f}")
    log(f"Val Loss: {val_loss:.4f}")
    log(f"Dice: {dice_score:.4f} | IoU: {iou_score:.4f}")
    log("-"*40)

    # SAVE CHECKPOINT
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_loss": best_loss
    }, checkpoint_path)

    # SAVE BEST MODEL
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best.pth"))

log("Training completed")