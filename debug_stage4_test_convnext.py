import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import timm

# =========================================
# CONFIG
# =========================================

IMAGE_PATH = (
    r"G:\Ai-CDD\data\preprocessed\stage4\test\TUBERCULOSIS\tb_00001.png"
)

MODEL_PATH = (
    r"G:\Ai-CDD\convnext_checkpoints\best_convnext.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMG_SIZE = 224

# =========================================
# MODEL
# =========================================

class ConvNextTB(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = timm.create_model(
            "convnext_tiny",
            pretrained=False,
            num_classes=2
        )

    def forward(self, x):

        return self.backbone(x)

# =========================================
# LOAD MODEL
# =========================================

model = ConvNextTB().to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.eval()

print("✅ Model loaded")

# =========================================
# LOAD IMAGE
# =========================================

img = cv2.imread(
    IMAGE_PATH,
    cv2.IMREAD_GRAYSCALE
)

if img is None:

    raise ValueError("Image not found")

img = cv2.resize(
    img,
    (IMG_SIZE, IMG_SIZE)
)

img = img.astype(np.float32) / 255.0

img = np.stack([img, img, img], axis=0)

tensor = torch.tensor(
    img,
    dtype=torch.float32
).unsqueeze(0).to(DEVICE)

# =========================================
# PREDICTION
# =========================================

with torch.no_grad():

    output = model(tensor)

    probs = torch.softmax(output, dim=1)

    conf, pred = torch.max(probs, dim=1)

classes = {
    0: "NORMAL",
    1: "TUBERCULOSIS"
}

print("\n========== RESULT ==========")

print(f"Prediction : {classes[pred.item()]}")
print(f"Confidence : {conf.item():.4f}")

print("============================")