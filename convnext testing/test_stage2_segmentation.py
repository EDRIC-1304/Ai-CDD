import os
import cv2
import numpy as np
import torch
import torch.nn as nn

# =========================================
# CONFIG
# =========================================

IMG_SIZE = 256

MODEL_PATH = (
    r"G:\Ai-CDD\segmentation_checkpoints\final_unet.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =========================================
# INPUT / OUTPUT
# =========================================

INPUT_IMAGE = (
    r"G:\Ai-CDD\convnext testing\outputs\stage1_preprocessed.png"
)

OUTPUT_DIR = (
    r"G:\Ai-CDD\convnext testing\outputs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

OUTPUT_MASK = os.path.join(
    OUTPUT_DIR,
    "stage2_mask.png"
)

OUTPUT_DEBUG = os.path.join(
    OUTPUT_DIR,
    "stage2_debug.png"
)

# =========================================
# DOUBLE CONV BLOCK
# =========================================

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

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.conv(x)

# =========================================
# U-NET
# =========================================

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

        d1 = self.down1(x)
        p1 = self.pool1(d1)

        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        d3 = self.down3(p2)
        p3 = self.pool3(d3)

        d4 = self.down4(p3)
        p4 = self.pool4(d4)

        bn = self.bottleneck(p4)

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

# =========================================
# LOAD MODEL
# =========================================

model = UNet().to(DEVICE)

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(state_dict)

model.eval()

print("✅ UNet model loaded successfully")

# =========================================
# MAIN
# =========================================

def main():

    # =====================================
    # LOAD IMAGE
    # =====================================

    img = cv2.imread(
        INPUT_IMAGE,
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:

        raise ValueError(
            f"❌ Failed to load image:\n{INPUT_IMAGE}"
        )

    print("✅ Stage1 image loaded")

    # =====================================
    # RESIZE
    # =====================================

    img = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE)
    )

    # =====================================
    # NORMALIZE
    # =====================================

    img_norm = (
        img.astype(np.float32) / 255.0
    )

    # =====================================
    # TO TENSOR
    # =====================================

    tensor = torch.tensor(
        img_norm,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # =====================================
    # PREDICT
    # =====================================

    with torch.no_grad():

        pred = model(tensor)

        pred = torch.sigmoid(pred)

        pred = pred[0, 0].cpu().numpy()

    # =====================================
    # BINARIZE
    # =====================================

    binary_mask = (
        pred > 0.5
    ).astype(np.uint8) * 255

    # =====================================
    # SAVE MASK
    # =====================================

    cv2.imwrite(
        OUTPUT_MASK,
        binary_mask
    )

    # =====================================
    # DEBUG IMAGE
    # =====================================

    combined = np.hstack([
        img,
        binary_mask
    ])

    cv2.imwrite(
        OUTPUT_DEBUG,
        combined
    )

    print("\n✅ Stage 2 complete")

    print(f"\nSaved mask:")
    print(OUTPUT_MASK)

    print(f"\nSaved debug:")
    print(OUTPUT_DEBUG)

# =========================================
# RUN
# =========================================

if __name__ == "__main__":
    main()