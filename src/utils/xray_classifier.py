# import os
# import random
# import cv2
# import numpy as np
# from src.xray_preprocessing import preprocess_image
# from train import CNN, compute_prototypes, euclidean_dist
# import torch.nn.functional as F
# import torch
# import torch.nn as nn

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # load trained encoder
# class FixedCNN(CNN):
#     def forward(self, x):
#         x = self.net(x)
#         x = x.view(x.size(0), -1)
#         return x

# model = FixedCNN().to(DEVICE)
# model.load_state_dict(torch.load("checkpoints/best.pth", map_location=DEVICE))
# model.eval()


# IMG_SIZE = 224

# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# def preprocess(image):
#     img = np.array(image)

#     # convert to grayscale
#     if len(img.shape) == 3:
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

#     # CLAHE
#     img = clahe.apply(img)

#     return img



# # 🔧 SAME UNET YOU TRAINED
# class ConvBlock(nn.Module):
#     def __init__(self, in_c, out_c):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_c, out_c, 3, padding=1),
#             nn.BatchNorm2d(out_c),
#             nn.ReLU(),
#             nn.Conv2d(out_c, out_c, 3, padding=1),
#             nn.BatchNorm2d(out_c),
#             nn.ReLU()
#         )

#     def forward(self, x):
#         return self.conv(x)


# class UNet(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.c1 = ConvBlock(1, 32)
#         self.p1 = nn.MaxPool2d(2)

#         self.c2 = ConvBlock(32, 64)
#         self.p2 = nn.MaxPool2d(2)

#         self.c3 = ConvBlock(64, 128)
#         self.p3 = nn.MaxPool2d(2)

#         self.c4 = ConvBlock(128, 256)
#         self.p4 = nn.MaxPool2d(2)

#         self.bn = ConvBlock(256, 512)

#         self.u1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
#         self.c5 = ConvBlock(512, 256)

#         self.u2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
#         self.c6 = ConvBlock(256, 128)

#         self.u3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
#         self.c7 = ConvBlock(128, 64)

#         self.u4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
#         self.c8 = ConvBlock(64, 32)

#         self.out = nn.Conv2d(32, 1, 1)

#     def forward(self, x):
#         c1 = self.c1(x)
#         c2 = self.c2(self.p1(c1))
#         c3 = self.c3(self.p2(c2))
#         c4 = self.c4(self.p3(c3))

#         bn = self.bn(self.p4(c4))

#         u1 = self.u1(bn)
#         u1 = torch.cat([u1, c4], dim=1)
#         c5 = self.c5(u1)

#         u2 = self.u2(c5)
#         u2 = torch.cat([u2, c3], dim=1)
#         c6 = self.c6(u2)

#         u3 = self.u3(c6)
#         u3 = torch.cat([u3, c2], dim=1)
#         c7 = self.c7(u3)

#         u4 = self.u4(c7)
#         u4 = torch.cat([u4, c1], dim=1)
#         c8 = self.c8(u4)

#         return torch.sigmoid(self.out(c8))


# # 🔥 LOAD MODEL ONCE (IMPORTANT)
# unet_model = UNet().to(DEVICE)
# unet_model.load_state_dict(torch.load("models_saved/Unet.pth", map_location=DEVICE))
# unet_model.eval()


# def unet_predict(img):
#     img_norm = img.astype(np.float32) / 255.0

#     tensor = torch.tensor(img_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

#     with torch.no_grad():
#         pred = unet_model(tensor)[0, 0].cpu().numpy()

#     return pred

# def clean_mask(pred):

#     pred = pred.astype(np.float32)

#     # threshold
#     mask = (pred > 0.5).astype(np.uint8) * 255

#     # opening
#     kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=2)

#     # closing
#     kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

#     # keep largest components
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

#     if num_labels > 1:
#         areas = stats[1:, cv2.CC_STAT_AREA]
#         largest = np.argsort(areas)[-2:] + 1

#         clean = np.zeros_like(mask)
#         for idx in largest:
#             clean[labels == idx] = 255
#         mask = clean

#     # fill holes
#     h, w = mask.shape
#     flood = mask.copy()
#     flood_mask = np.zeros((h + 2, w + 2), np.uint8)
#     cv2.floodFill(flood, flood_mask, (0, 0), 255)
#     flood_inv = cv2.bitwise_not(flood)
#     mask = mask | flood_inv

#     return mask

# def get_support_embeddings(k=5):
#     support = []
#     labels = []

#     for label, cls in enumerate(["NORMAL", "TUBERCULOSIS"]):
#         path = os.path.join(r"G:\Ai-CDD\data\preprocessed\stage4\train", cls)
#         files = random.sample(os.listdir(path), k)

#         for f in files:
#             img = cv2.imread(os.path.join(path, f), cv2.IMREAD_GRAYSCALE)
#             img = cv2.resize(img, (224,224))
#             img = img.astype(np.float32)/255.0

#             tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

#             with torch.no_grad():
#                 emb = model(tensor)
#                 emb = F.normalize(emb, dim=1)

#             support.append(emb)
#             labels.append(label)

#     support = torch.cat(support)
#     labels = torch.tensor(labels).to(DEVICE)

#     return support, labels

# def compute_prototypes_from_support(support, labels):
#     prototypes = []
#     for c in [0, 1]:
#         class_emb = support[labels == c]
#         prototypes.append(class_emb.mean(0))
#     return torch.stack(prototypes)


# def pan_predict(roi):

#     support, labels = get_support_embeddings()
#     prototypes = compute_prototypes_from_support(support, labels)
#     prototypes = F.normalize(prototypes, dim=1)

#     # prototypes = prototypes[[1, 0]]

#     img = cv2.resize(roi, (224, 224))
#     img = img.astype(np.float32) / 255.0

#     tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

#     with torch.no_grad():
#         emb = model(tensor)
#         emb = F.normalize(emb, dim=1)

#         dists = euclidean_dist(emb, prototypes)
        
#         dists = dists / 0.1
        
#         print("Distances:", dists.cpu().numpy())  # 🔥 debug

#         probs = torch.softmax(-dists, dim=1)

#         pred = torch.argmax(probs, dim=1).item()
#         confidence = probs[0, pred].item()

#     return pred, confidence

# def xray_full_pipeline(image):

#     # Stage 0: original
#     original = np.array(image)

#     # Stage 1: preprocessing
#     stage1 = preprocess_image(original)

#     # Stage 2: UNet
#     stage2 = unet_predict(stage1)

#     # Stage 3: mask cleaning
#     stage3 = clean_mask(stage2)

#     # Stage 4: ROI
#     stage4 = cv2.bitwise_and(stage1, stage1, mask=stage3)

#     # Stage 5: classification
#     pred, confidence = pan_predict(stage4)

#     return original, stage1, stage2, stage3, stage4, pred, confidence






import os
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.xray_preprocessing import preprocess_image
from train import CNN

# =========================================================
# DEVICE
# =========================================================
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMG_SIZE = 224

# =========================================================
# DISTANCE FUNCTION
# =========================================================
def euclidean_dist(x, y):

    return (
        (x.unsqueeze(1) - y.unsqueeze(0)) ** 2
    ).sum(2)

# =========================================================
# LOAD CLASSIFIER
# =========================================================
model = CNN().to(DEVICE)

model.load_state_dict(
    torch.load(
        "checkpoints/best.pth",
        map_location=DEVICE
    )
)

model.eval()

# =========================================================
# CLAHE
# =========================================================
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

# =========================================================
# PREPROCESS
# =========================================================
def preprocess(image):

    img = np.array(image)

    if len(img.shape) == 3:

        img = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2GRAY
        )

    img = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE)
    )

    img = clahe.apply(img)

    return img

# =========================================================
# UNET BLOCKS
# =========================================================
class ConvBlock(nn.Module):

    def __init__(self, in_c, out_c):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                in_c,
                out_c,
                3,
                padding=1
            ),

            nn.BatchNorm2d(out_c),

            nn.ReLU(),

            nn.Conv2d(
                out_c,
                out_c,
                3,
                padding=1
            ),

            nn.BatchNorm2d(out_c),

            nn.ReLU()
        )

    def forward(self, x):

        return self.conv(x)

# =========================================================
# UNET
# =========================================================
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

        self.u1 = nn.ConvTranspose2d(
            512,
            256,
            2,
            stride=2
        )

        self.c5 = ConvBlock(512, 256)

        self.u2 = nn.ConvTranspose2d(
            256,
            128,
            2,
            stride=2
        )

        self.c6 = ConvBlock(256, 128)

        self.u3 = nn.ConvTranspose2d(
            128,
            64,
            2,
            stride=2
        )

        self.c7 = ConvBlock(128, 64)

        self.u4 = nn.ConvTranspose2d(
            64,
            32,
            2,
            stride=2
        )

        self.c8 = ConvBlock(64, 32)

        self.out = nn.Conv2d(
            32,
            1,
            1
        )

    def forward(self, x):

        c1 = self.c1(x)

        c2 = self.c2(
            self.p1(c1)
        )

        c3 = self.c3(
            self.p2(c2)
        )

        c4 = self.c4(
            self.p3(c3)
        )

        bn = self.bn(
            self.p4(c4)
        )

        u1 = self.u1(bn)

        u1 = torch.cat(
            [u1, c4],
            dim=1
        )

        c5 = self.c5(u1)

        u2 = self.u2(c5)

        u2 = torch.cat(
            [u2, c3],
            dim=1
        )

        c6 = self.c6(u2)

        u3 = self.u3(c6)

        u3 = torch.cat(
            [u3, c2],
            dim=1
        )

        c7 = self.c7(u3)

        u4 = self.u4(c7)

        u4 = torch.cat(
            [u4, c1],
            dim=1
        )

        c8 = self.c8(u4)

        return torch.sigmoid(
            self.out(c8)
        )

# =========================================================
# LOAD UNET
# =========================================================
unet_model = UNet().to(DEVICE)

unet_model.load_state_dict(
    torch.load(
        "models_saved/Unet.pth",
        map_location=DEVICE
    )
)

unet_model.eval()

# =========================================================
# UNET PREDICT
# =========================================================
def unet_predict(img):

    img = img.astype(np.float32) / 255.0

    tensor = torch.tensor(
        img,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        pred = unet_model(tensor)[0, 0]

        pred = pred.cpu().numpy()

    return pred

# =========================================================
# CLEAN MASK
# =========================================================
def clean_mask(pred):

    mask = (pred > 0.5).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask

# =========================================================
# BUILD FIXED PROTOTYPES ONCE
# =========================================================
def build_prototypes(k=15):

    embeddings = []
    labels = []

    train_root = (
        r"G:\Ai-CDD\data\preprocessed\stage4\train"
    )

    for label, cls in enumerate([
        "NORMAL",
        "TUBERCULOSIS"
    ]):

        cls_path = os.path.join(
            train_root,
            cls
        )

        files = random.sample(
            os.listdir(cls_path),
            min(k, len(os.listdir(cls_path)))
        )

        for file in files:

            path = os.path.join(
                cls_path,
                file
            )

            img = cv2.imread(
                path,
                cv2.IMREAD_GRAYSCALE
            )

            img = cv2.resize(
                img,
                (IMG_SIZE, IMG_SIZE)
            )

            img = clahe.apply(img)

            img = img.astype(np.float32) / 255.0

            tensor = torch.tensor(
                img,
                dtype=torch.float32
            ).unsqueeze(0).unsqueeze(0).to(DEVICE)

            with torch.no_grad():

                emb = model(tensor)

            embeddings.append(emb)

            labels.append(label)

    embeddings = torch.cat(embeddings)

    labels = torch.tensor(labels).to(DEVICE)

    prototypes = []

    for c in [0, 1]:

        proto = embeddings[
            labels == c
        ].mean(0)

        proto = F.normalize(
            proto,
            dim=0
        )

        prototypes.append(proto)

    prototypes = torch.stack(prototypes)

    return prototypes

# =========================================================
# CREATE PROTOTYPES ONCE
# =========================================================
PROTOTYPES = build_prototypes()

# =========================================================
# CLASSIFICATION
# =========================================================
def pan_predict(roi):

    img = cv2.resize(
        roi,
        (IMG_SIZE, IMG_SIZE)
    )

    img = clahe.apply(img)

    img = img.astype(np.float32) / 255.0

    tensor = torch.tensor(
        img,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        emb = model(tensor)

        dists = euclidean_dist(
            emb,
            PROTOTYPES
        )

        probs = torch.softmax(
            -dists / 0.15,
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

    return pred, confidence

# =========================================================
# FULL PIPELINE
# =========================================================
def xray_full_pipeline(image):

    original = np.array(image)

    stage1 = preprocess_image(original)

    stage2 = unet_predict(stage1)

    stage3 = clean_mask(stage2)

    stage4 = cv2.bitwise_and(
        stage1,
        stage1,
        mask=stage3
    )

    pred, confidence = pan_predict(stage4)

    return (
        original,
        stage1,
        stage2,
        stage3,
        stage4,
        pred,
        confidence
    )