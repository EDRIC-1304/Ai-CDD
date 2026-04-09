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
MODEL_PATH = r"models_saved\Unet.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# PATHS
# -----------------------------
BASE_IN = r"G:\Ai-CDD\data\preprocessed\stage1"
BASE_OUT = r"G:\Ai-CDD\data\preprocessed\stage2"

# -----------------------------
# MODEL
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


# -----------------------------
# LOAD MODEL
# -----------------------------
model = UNet().to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

# -----------------------------
# PROCESS FUNCTION
# -----------------------------
def process_split(split):
    for class_name in ["NORMAL", "TUBERCULOSIS"]:
        input_dir = os.path.join(BASE_IN, split, class_name)
        output_dir = os.path.join(BASE_OUT, split, class_name)

        if not os.path.exists(input_dir):
            print(f"(xray_unet_preprocessing.py)❌ Missing: {input_dir}")
            continue

        os.makedirs(output_dir, exist_ok=True)

        files = os.listdir(input_dir)

        for file in tqdm(files, desc=f"{split}/{class_name}"):
            path = os.path.join(input_dir, file)

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype(np.float32) / 255.0

            tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                pred = model(tensor)[0, 0].cpu().numpy()

            save = (pred * 255).astype(np.uint8)

            out_path = os.path.join(output_dir, file)  # keep same name
            cv2.imwrite(out_path, save)

# -----------------------------
# RUN
# -----------------------------
for split in ["train", "val", "test"]:
    process_split(split)

print("\n🎯 Stage 2 preprocessing complete")