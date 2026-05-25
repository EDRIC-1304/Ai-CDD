import os
import cv2
import numpy as np

import torch
import torch.nn as nn

import timm

# =========================================================
# CONFIG
# =========================================================
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_PATH = (
    r"G:\Ai-CDD\convnext testing\outputs\4_roi.png"
)

MODEL_PATH = (
    r"G:\Ai-CDD\convnext_checkpoints\best_convnext.pth"
)

IMG_SIZE = 224

CLASS_NAMES = [
    "NORMAL",
    "TUBERCULOSIS"
]

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
# LOAD MODEL
# =========================================================
model = ConvNextTB().to(DEVICE)

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(state_dict)

model.eval()

print("✅ ConvNeXt model loaded")

# =========================================================
# LOAD IMAGE
# =========================================================
img = cv2.imread(
    IMAGE_PATH,
    cv2.IMREAD_GRAYSCALE
)

if img is None:

    raise ValueError(
        "❌ Failed to load ROI image"
    )

# =========================================================
# PREPARE IMAGE
# =========================================================
img = cv2.resize(
    img,
    (IMG_SIZE, IMG_SIZE)
)

# IMPORTANT:
# EXACT SAME NORMALIZATION
# USED DURING TRAINING
img = img.astype(np.float32) / 255.0

# IMPORTANT:
# EXACT SAME CHANNEL FORMAT
# USED DURING TRAINING
img = np.stack(
    [img, img, img],
    axis=0
)

tensor = torch.tensor(
    img,
    dtype=torch.float32
).unsqueeze(0).to(DEVICE)

# =========================================================
# PREDICTION
# =========================================================
with torch.no_grad():

    output = model(tensor)

    probs = torch.softmax(
        output,
        dim=1
    )

    confidence, pred = torch.max(
        probs,
        dim=1
    )

pred_class = pred.item()

confidence = confidence.item()

# =========================================================
# RESULTS
# =========================================================
print("\n========== RESULT ==========")

print(
    f"Prediction : "
    f"{CLASS_NAMES[pred_class]}"
)

print(
    f"Confidence : "
    f"{confidence:.4f}"
)

print("============================")

# =========================================================
# FULL PROBABILITIES
# =========================================================
print("\nClass Probabilities:\n")

for i, class_name in enumerate(CLASS_NAMES):

    print(
        f"{class_name}: "
        f"{probs[0][i].item():.4f}"
    )