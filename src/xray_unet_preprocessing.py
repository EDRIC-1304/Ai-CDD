import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 256

MODEL_PATH = r"G:\Ai-CDD\segmentation_checkpoints\final_unet.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# -----------------------------
# DEBUG
# -----------------------------
DEBUG_DIR = r"G:\Ai-CDD\src\debug_stage2"
FAILED_DIR = r"G:\Ai-CDD\src\failed_masks"

os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(FAILED_DIR, exist_ok=True)

# -----------------------------
# UPDATED PATHS
# -----------------------------

# INPUT = OUTPUT OF STAGE 1 PREPROCESSING
BASE_IN = r"G:\Ai-CDD\data\preprocessed\stage1"

# OUTPUT = STAGE 2 PREPROCESSING
BASE_OUT = r"G:\Ai-CDD\data\preprocessed\stage2"

# -----------------------------
# DOUBLE CONV BLOCK
# -----------------------------
class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

# -----------------------------
# U-NET
# -----------------------------
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
        self.up1 = nn.ConvTranspose2d(
            1024,
            512,
            kernel_size=2,
            stride=2
        )

        self.conv1 = DoubleConv(1024, 512)

        self.up2 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.conv2 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.conv3 = DoubleConv(256, 128)

        self.up4 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.conv4 = DoubleConv(128, 64)

        # Output
        self.out = nn.Conv2d(
            64,
            1,
            kernel_size=1
        )

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

        # IMPORTANT:
        # NO SIGMOID HERE
        return self.out(u4)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = UNet().to(DEVICE)

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(state_dict)

model.eval()

print("✅ UNet model loaded successfully")

# -----------------------------
# PROCESS FUNCTION
# -----------------------------
def process_split(split):

    # -----------------------------
    # UPDATED CLASSES
    # -----------------------------
    for class_name in [
        "health",
        "sick",
        "tb"
    ]:

        input_dir = os.path.join(
            BASE_IN,
            split,
            class_name
        )

        output_dir = os.path.join(
            BASE_OUT,
            split,
            class_name
        )

        if not os.path.exists(input_dir):
            print(f"❌ Missing: {input_dir}")
            continue

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        valid_ext = (
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp"
        )

        files = [
            f for f in os.listdir(input_dir)
            if f.lower().endswith(valid_ext)
        ]

        processed_count = 0
        failed_count = 0

        for i, file in enumerate(
            tqdm(files, desc=f"{split}/{class_name}")
        ):

            path = os.path.join(
                input_dir,
                file
            )

            # -------------------------
            # READ IMAGE
            # -------------------------
            img = cv2.imread(
                path,
                cv2.IMREAD_GRAYSCALE
            )

            if img is None:
                failed_count += 1
                continue

            # -------------------------
            # RESIZE
            # -------------------------
            img = cv2.resize(
                img,
                (IMG_SIZE, IMG_SIZE)
            )

            # -------------------------
            # NORMALIZE
            # -------------------------
            img_norm = (
                img.astype(np.float32) / 255.0
            )

            # -------------------------
            # TO TENSOR
            # -------------------------
            tensor = torch.tensor(
                img_norm,
                dtype=torch.float32
            ).unsqueeze(0).unsqueeze(0).to(DEVICE)

            # -------------------------
            # PREDICT
            # -------------------------
            with torch.no_grad():

                pred = model(tensor)

                # APPLY SIGMOID HERE
                pred = torch.sigmoid(pred)

                pred = pred[0, 0].cpu().numpy()

            # -------------------------
            # BINARIZE MASK
            # -------------------------
            binary_mask = (
                pred > 0.5
            ).astype(np.uint8) * 255

            # -------------------------
            # CHECK FAILED MASK
            # -------------------------
            white_pixels = np.sum(binary_mask > 0)

            if white_pixels < 500:

                failed_count += 1

                failed_path = os.path.join(
                    FAILED_DIR,
                    f"{split}_{class_name}_{file}"
                )

                cv2.imwrite(
                    failed_path,
                    binary_mask
                )

            # -------------------------
            # SAVE MASK
            # -------------------------
            out_path = os.path.join(
                output_dir,
                file
            )

            cv2.imwrite(
                out_path,
                binary_mask
            )

            processed_count += 1

            # -------------------------
            # DEBUG SAVE
            # -------------------------
            if i < 20:

                combined = np.hstack([
                    img,
                    binary_mask
                ])

                debug_path = os.path.join(
                    DEBUG_DIR,
                    f"{split}_{class_name}_{i}.png"
                )

                cv2.imwrite(
                    debug_path,
                    combined
                )

        # -------------------------
        # SUMMARY
        # -------------------------
        print(f"\n✅ {split}/{class_name}")
        print(f"✔ Processed: {processed_count}")
        print(f"⚠️ Weak Masks: {failed_count}")
        print("-" * 40)

# -----------------------------
# RUN
# -----------------------------
for split in [
    "train",
    "val",
    "internal_test",
    "external_test"
]:

    process_split(split)

print("\n🎯 Stage 2 preprocessing complete")