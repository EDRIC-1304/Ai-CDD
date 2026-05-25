import os
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = (
    r"G:\Ai-CDD\data\preprocessed"
    r"\classification_stage1"
)

OUTPUT_DIR = (
    r"G:\Ai-CDD\data\preprocessed"
    r"\classification_segmented"
)

MODEL_PATH = (
    r"G:\Ai-CDD\checkpoints"
    r"\best_attention_unet.pth"
)

# MUST MATCH TRAINING
IMG_SIZE = 256

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\nUsing Device: {DEVICE}")

# ============================================================
# DOUBLE CONV
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

        # MUST MATCH TRAINING MODEL EXACTLY
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
# LOAD MODEL
# ============================================================

model = AttentionUNet().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

# HANDLE BOTH DIRECT STATE_DICT AND CHECKPOINT FORMAT
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(checkpoint)

model.eval()

print("\nSegmentation model loaded successfully.")

# ============================================================
# SEGMENT SINGLE IMAGE
# ============================================================

def segment_image(image_path):

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        print(f"Failed to load: {image_path}")

        return None

    original = image.copy()

    original_h, original_w = original.shape

    # ========================================================
    # RESIZE
    # ========================================================

    resized = cv2.resize(
        image,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    resized = resized.astype(np.float32) / 255.0

    resized = (
        resized - resized.mean()
    ) / (
        resized.std() + 1e-8
    )

    # ========================================================
    # TO TENSOR
    # ========================================================

    resized = np.expand_dims(
        resized,
        axis=0
    )

    resized = np.expand_dims(
        resized,
        axis=0
    )

    tensor = torch.tensor(
        resized,
        dtype=torch.float32
    ).to(DEVICE)

    # ========================================================
    # INFERENCE
    # ========================================================

    with torch.no_grad():

        with torch.amp.autocast(
            device_type="cuda",
            enabled=(DEVICE == "cuda")
        ):

            pred_mask = model(tensor)

            pred_mask = torch.sigmoid(pred_mask)

    pred_mask = (
        pred_mask.squeeze()
        .cpu()
        .numpy()
    )

    # ========================================================
    # THRESHOLD
    # ========================================================

    pred_mask = (
        pred_mask > 0.5
    ).astype(np.uint8)

    # ========================================================
    # MORPHOLOGICAL CLEANING
    # ========================================================

    kernel = np.ones((5, 5), np.uint8)

    pred_mask = cv2.morphologyEx(
        pred_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    pred_mask = cv2.morphologyEx(
        pred_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # ========================================================
    # REMOVE SMALL COMPONENTS
    # ========================================================

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        pred_mask,
        connectivity=8
    )

    clean_mask = np.zeros_like(pred_mask)

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        # LOWER THIS IF LUNGS GET CUT OFF
        if area > 1000:

            clean_mask[labels == i] = 1

    pred_mask = clean_mask

    # ========================================================
    # RESIZE BACK
    # ========================================================

    pred_mask = cv2.resize(
        pred_mask,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST
    )

    pred_mask = pred_mask.astype(np.uint8)

    # ========================================================
    # APPLY MASK
    # ========================================================

    segmented = original * pred_mask

    return segmented

# ============================================================
# PROCESS DATASET
# ============================================================

valid_extensions = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff"
)

for split in ["train", "val", "test"]:

    split_input = os.path.join(
        INPUT_DIR,
        split
    )

    if not os.path.exists(split_input):

        print(f"\nSkipping missing split: {split}")

        continue

    split_output = os.path.join(
        OUTPUT_DIR,
        split
    )

    os.makedirs(split_output, exist_ok=True)

    classes = os.listdir(split_input)

    for cls in classes:

        class_input = os.path.join(
            split_input,
            cls
        )

        if not os.path.isdir(class_input):
            continue

        class_output = os.path.join(
            split_output,
            cls
        )

        os.makedirs(class_output, exist_ok=True)

        images = [

            img for img in os.listdir(class_input)

            if img.lower().endswith(valid_extensions)
        ]

        print(f"\nProcessing {split}/{cls}")
        print(f"Images Found: {len(images)}")

        for img_name in tqdm(images):

            img_path = os.path.join(
                class_input,
                img_name
            )

            segmented = segment_image(img_path)

            if segmented is None:
                continue

            save_path = os.path.join(
                class_output,
                img_name
            )

            cv2.imwrite(
                save_path,
                segmented
            )

print("\nClassification dataset segmentation completed.")