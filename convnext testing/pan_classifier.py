import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# CONFIG
# =========================================================
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_PATH = (
    r"G:\Ai-CDD\convnext testing\outputs\4_roi.png"
)

CHECKPOINT_PATH = (
    r"G:\Ai-CDD\checkpoints\best.pth"
)

PROTOTYPE_PATH = (
    r"G:\Ai-CDD\checkpoints\prototypes.pth"
)

IMG_SIZE = 224

CLASS_NAMES = [
    "NORMAL",
    "TUBERCULOSIS"
]

# =========================================================
# MODEL
# =========================================================
class CNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):

        x = self.net(x)

        x = x.view(x.size(0), -1)

        x = F.normalize(x, dim=1)

        return x

# =========================================================
# LOAD MODEL
# =========================================================
print("Loading model...")

model = CNN().to(DEVICE)

model.load_state_dict(
    torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )
)

model.eval()

print("✅ Model loaded")

# =========================================================
# LOAD PROTOTYPES
# =========================================================
print("Loading prototypes...")

prototypes = torch.load(
    PROTOTYPE_PATH,
    map_location=DEVICE
)

prototypes = F.normalize(
    prototypes,
    dim=1
)

print("✅ Prototypes loaded")

print("Prototype Shape:")
print(prototypes.shape)

# =========================================================
# IMAGE LOADER
# =========================================================
def load_image(path):

    img = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:

        raise ValueError(
            f"❌ Failed to load image: {path}"
        )

    img = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE)
    )

    # SAME PREPROCESSING AS TRAINING
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    img = clahe.apply(img)

    img = img.astype(np.float32) / 255.0

    tensor = torch.tensor(
        img,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    return tensor, img

# =========================================================
# COSINE LOGITS
# =========================================================
def cosine_logits(x, y):

    return F.cosine_similarity(
        x.unsqueeze(1),
        y.unsqueeze(0),
        dim=2
    )

# =========================================================
# PREDICT
# =========================================================
def predict(image_path):

    tensor, processed = load_image(
        image_path
    )

    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        embedding = model(tensor)

        embedding = F.normalize(
            embedding,
            dim=1
        )

        logits = cosine_logits(
            embedding,
            prototypes
        )

        probs = torch.softmax(
            logits,
            dim=1
        )

        pred = torch.argmax(
            probs,
            dim=1
        ).item()

        confidence = probs[
            0,
            pred
        ].item()

    return (
        pred,
        confidence,
        probs.cpu().numpy(),
        processed
    )

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    print("\nRunning prediction...\n")

    pred, confidence, probs, processed = predict(
        IMAGE_PATH
    )

    print("==============================")
    print("FINAL PREDICTION")
    print("==============================")

    print(f"\nImage:")
    print(IMAGE_PATH)

    print(f"\nPredicted Class:")
    print(CLASS_NAMES[pred])

    print(f"\nConfidence:")
    print(f"{confidence:.4f}")

    print("\nClass Probabilities:")

    for i, cls in enumerate(CLASS_NAMES):

        print(
            f"{cls}: "
            f"{probs[0][i]:.4f}"
        )

    # =====================================================
    # SHOW IMAGE
    # =====================================================
    display = (processed * 255).astype(np.uint8)

    cv2.imshow(
        "Processed ROI",
        display
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()