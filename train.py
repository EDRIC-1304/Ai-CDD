
# import os
# import random
# import cv2
# import numpy as np

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F

# from log import log

# # -----------------------------
# # CONFIG
# # -----------------------------
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATA_DIR = r"G:\Ai-CDD\data\preprocessed\stage4\train"
# VAL_DIR = r"G:\Ai-CDD\data\preprocessed\stage4\val"

# CHECKPOINT_DIR = "checkpoints"
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# IMG_SIZE = 224

# LR = 1e-3

# EPISODES_PER_EPOCH = 100
# EPOCHS = 100

# SEED = 42

# VALID_EXT = (".png", ".jpg", ".jpeg", ".bmp")

# # -----------------------------
# # REPRODUCIBILITY
# # -----------------------------
# random.seed(SEED)
# np.random.seed(SEED)

# torch.manual_seed(SEED)

# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(SEED)

# # -----------------------------
# # DATASET
# # -----------------------------
# class Dataset:

#     def __init__(self, root):

#         self.data = {
#             "NORMAL": [],
#             "TUBERCULOSIS": []
#         }

#         for cls in self.data:

#             path = os.path.join(root, cls)

#             self.data[cls] = [
#                 os.path.join(path, f)
#                 for f in os.listdir(path)
#                 if f.lower().endswith(VALID_EXT)
#             ]

#             print(f"{cls}: {len(self.data[cls])}")

#     def sample_episode(self, k=5):

#         support = []
#         query = []

#         for label, cls in enumerate(["NORMAL", "TUBERCULOSIS"]):

#             samples = random.sample(self.data[cls], k * 2)

#             for i in range(k):
#                 support.append((samples[i], label))

#             for i in range(k, 2 * k):
#                 query.append((samples[i], label))

#         return support, query


# class FullDataset:

#     def __init__(self, root):

#         self.samples = []

#         for label, cls in enumerate(["NORMAL", "TUBERCULOSIS"]):

#             path = os.path.join(root, cls)

#             files = [
#                 f for f in os.listdir(path)
#                 if f.lower().endswith(VALID_EXT)
#             ]

#             for f in files:
#                 self.samples.append(
#                     (os.path.join(path, f), label)
#                 )

# # -----------------------------
# # IMAGE LOADER
# # -----------------------------
# def load_image(path):

#     img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

#     if img is None:
#         raise ValueError(f"Failed to load: {path}")

#     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

#     img = img.astype(np.float32) / 255.0

#     if img.mean() < 0.01:
#         img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

#     tensor = torch.tensor(
#         img,
#         dtype=torch.float32
#     ).unsqueeze(0)

#     return tensor

# # -----------------------------
# # MODEL
# # -----------------------------
# class CNN(nn.Module):

#     def __init__(self):
#         super().__init__()

#         self.net = nn.Sequential(

#             nn.Conv2d(1, 32, 3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),

#             nn.Conv2d(32, 32, 3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),

#             nn.MaxPool2d(2),

#             nn.Conv2d(32, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),

#             nn.Conv2d(64, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),

#             nn.MaxPool2d(2),

#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),

#             nn.Conv2d(128, 128, 3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),

#             nn.MaxPool2d(2),

#             nn.Conv2d(128, 256, 3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),

#             nn.AdaptiveAvgPool2d(1)
#         )

#     def forward(self, x):

#         x = self.net(x)

#         x = x.view(x.size(0), -1)

#         x = F.normalize(x, dim=1)

#         return x

# # -----------------------------
# # PROTOTYPES
# # -----------------------------
# def compute_prototypes(embeddings, labels):

#     prototypes = []

#     for c in [0, 1]:

#         class_emb = embeddings[labels == c]

#         proto = class_emb.mean(0)

#         proto = F.normalize(proto, dim=0)

#         prototypes.append(proto)

#     return torch.stack(prototypes)

# # -----------------------------
# # DISTANCE
# # -----------------------------
# def euclidean_dist(x, y):

#     return ((x.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(2)

# # -----------------------------
# # VALIDATION
# # -----------------------------
# def validate(model, train_data, val_data):

#     model.eval()

#     # -------------------------
#     # BUILD PROTOTYPES
#     # FROM TRAIN SET
#     # -------------------------
#     train_embeddings = []
#     train_labels = []

#     with torch.no_grad():

#         for cls_idx, cls_name in enumerate(["NORMAL", "TUBERCULOSIS"]):

#             sample_paths = random.sample(
#                 train_data.data[cls_name],
#                 min(200, len(train_data.data[cls_name]))
#             )

#             for path in sample_paths:

#                 img = load_image(path).unsqueeze(0).to(DEVICE)

#                 emb = model(img)

#                 train_embeddings.append(emb)

#                 train_labels.append(cls_idx)

#     train_embeddings = torch.cat(train_embeddings)

#     train_labels = torch.tensor(train_labels).to(DEVICE)

#     prototypes = compute_prototypes(
#         train_embeddings,
#         train_labels
#     )

#     # -------------------------
#     # VALIDATE
#     # -------------------------
#     correct = 0
#     total = 0

#     tb_tp = 0
#     tb_fn = 0

#     with torch.no_grad():

#         for path, label in val_data.samples:

#             img = load_image(path).unsqueeze(0).to(DEVICE)

#             emb = model(img)

#             dists = euclidean_dist(emb, prototypes)

#             pred = torch.argmax(-dists, dim=1).item()

#             if pred == label:
#                 correct += 1

#             if label == 1:

#                 if pred == 1:
#                     tb_tp += 1
#                 else:
#                     tb_fn += 1

#             total += 1

#     acc = correct / total

#     recall_tb = tb_tp / (tb_tp + tb_fn + 1e-8)

#     return acc, recall_tb

# # -----------------------------
# # TRAIN
# # -----------------------------
# def train():

#     train_data = Dataset(DATA_DIR)

#     val_data = FullDataset(VAL_DIR)

#     model = CNN().to(DEVICE)

#     optimizer = optim.Adam(
#         model.parameters(),
#         lr=LR
#     )

#     best_acc = 0

#     checkpoint_path = os.path.join(
#         CHECKPOINT_DIR,
#         "latest.pth"
#     )

#     for epoch in range(EPOCHS):

#         model.train()

#         epoch_loss = 0

#         correct = 0
#         total = 0

#         for episode in range(EPISODES_PER_EPOCH):

#             support, query = train_data.sample_episode()

#             sx = torch.stack([
#                 load_image(x[0]) for x in support
#             ]).to(DEVICE)

#             sy = torch.tensor([
#                 x[1] for x in support
#             ]).to(DEVICE)

#             qx = torch.stack([
#                 load_image(x[0]) for x in query
#             ]).to(DEVICE)

#             qy = torch.tensor([
#                 x[1] for x in query
#             ]).to(DEVICE)

#             emb_s = model(sx)

#             emb_q = model(qx)

#             prototypes = compute_prototypes(
#                 emb_s,
#                 sy
#             )

#             dists = euclidean_dist(
#                 emb_q,
#                 prototypes
#             )

#             loss = nn.CrossEntropyLoss()(
#                 -dists / 0.1,
#                 qy
#             )

#             optimizer.zero_grad()

#             loss.backward()

#             optimizer.step()

#             epoch_loss += loss.item()

#             preds = torch.argmax(-dists, dim=1)

#             correct += (preds == qy).sum().item()

#             total += qy.size(0)

#         train_loss = epoch_loss / EPISODES_PER_EPOCH

#         train_acc = correct / total

#         val_acc, val_tb_recall = validate(
#             model,
#             train_data,
#             val_data
#         )

#         log(
#             f"Epoch {epoch+1}/{EPOCHS} | "
#             f"Train Loss: {train_loss:.4f} | "
#             f"Train Acc: {train_acc:.4f} | "
#             f"Val Acc: {val_acc:.4f} | "
#             f"TB Recall: {val_tb_recall:.4f}"
#         )

#         # -------------------------
#         # SAVE LATEST
#         # -------------------------
#         torch.save({
#             "model": model.state_dict(),
#             "optimizer": optimizer.state_dict(),
#             "epoch": epoch
#         }, checkpoint_path)

#         # -------------------------
#         # SAVE BEST
#         # -------------------------
#         if val_acc > best_acc:

#             best_acc = val_acc

#             torch.save(
#                 model.state_dict(),
#                 os.path.join(
#                     CHECKPOINT_DIR,
#                     "best.pth"
#                 )
#             )

#             log(f"✅ New best model saved")

# # -----------------------------
# # MAIN
# # -----------------------------
# if __name__ == "__main__":
#     train()















import os
import random
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import transforms

from log import log

# =========================================================
# CONFIG
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = r"G:\Ai-CDD\data\preprocessed\stage4\train"
VAL_DIR = r"G:\Ai-CDD\data\preprocessed\stage4\val"

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

IMG_SIZE = 224

LR = 1e-3

EPISODES_PER_EPOCH = 100
EPOCHS = 60

K_SHOT = 5

SEED = 42

VALID_EXT = (".png", ".jpg", ".jpeg", ".bmp")

# =========================================================
# REPRODUCIBILITY
# =========================================================
random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =========================================================
# AUGMENTATION
# =========================================================
augment = transforms.Compose([

    transforms.ToPILImage(),

    transforms.RandomRotation(5),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.03, 0.03),
        scale=(0.97, 1.03)
    ),

    transforms.ToTensor()
])

# =========================================================
# DATASET
# =========================================================
class Dataset:

    def __init__(self, root):

        self.data = {
            "NORMAL": [],
            "TUBERCULOSIS": []
        }

        for cls in self.data:

            path = os.path.join(root, cls)

            self.data[cls] = [

                os.path.join(path, f)

                for f in os.listdir(path)

                if f.lower().endswith(VALID_EXT)
            ]

            print(f"{cls}: {len(self.data[cls])}")

    def sample_episode(self, k=K_SHOT):

        support = []
        query = []

        for label, cls in enumerate(
            ["NORMAL", "TUBERCULOSIS"]
        ):

            samples = random.sample(
                self.data[cls],
                k * 2
            )

            for i in range(k):
                support.append((samples[i], label))

            for i in range(k, k * 2):
                query.append((samples[i], label))

        return support, query


class FullDataset:

    def __init__(self, root):

        self.samples = []

        for label, cls in enumerate(
            ["NORMAL", "TUBERCULOSIS"]
        ):

            path = os.path.join(root, cls)

            files = [

                f for f in os.listdir(path)

                if f.lower().endswith(VALID_EXT)
            ]

            for f in files:

                self.samples.append(
                    (os.path.join(path, f), label)
                )

# =========================================================
# IMAGE LOADER
# =========================================================
def load_image(path, train=False):

    img = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        raise ValueError(f"Failed to load: {path}")

    img = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE)
    )

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    img = clahe.apply(img)

    if train:

        img = augment(img)

    else:

        img = torch.tensor(
            img / 255.0,
            dtype=torch.float32
        ).unsqueeze(0)

    return img

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
# PROTOTYPES
# =========================================================
def compute_prototypes(
    embeddings,
    labels
):

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

    return torch.stack(prototypes)

# =========================================================
# COSINE SIMILARITY
# =========================================================
def cosine_logits(x, y):

    return F.cosine_similarity(
        x.unsqueeze(1),
        y.unsqueeze(0),
        dim=2
    )

# =========================================================
# VALIDATION
# =========================================================
def validate(model, train_data, val_data):

    model.eval()

    train_emb = []
    train_labels = []

    with torch.no_grad():

        for cls_idx, cls_name in enumerate(
            ["NORMAL", "TUBERCULOSIS"]
        ):

            paths = random.sample(
                train_data.data[cls_name],
                min(200, len(train_data.data[cls_name]))
            )

            for path in paths:

                img = load_image(path)

                img = img.unsqueeze(0).to(DEVICE)

                emb = model(img)

                train_emb.append(emb)

                train_labels.append(cls_idx)

    train_emb = torch.cat(train_emb)

    train_labels = torch.tensor(
        train_labels
    ).to(DEVICE)

    prototypes = compute_prototypes(
        train_emb,
        train_labels
    )

    correct = 0
    total = 0

    with torch.no_grad():

        for path, label in val_data.samples:

            img = load_image(path)

            img = img.unsqueeze(0).to(DEVICE)

            emb = model(img)

            logits = cosine_logits(
                emb,
                prototypes
            )

            pred = torch.argmax(
                logits,
                dim=1
            ).item()

            if pred == label:
                correct += 1

            total += 1

    return correct / total

# =========================================================
# TRAIN
# =========================================================
def train():

    train_data = Dataset(DATA_DIR)

    val_data = FullDataset(VAL_DIR)

    model = CNN().to(DEVICE)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LR
    )

    criterion = nn.CrossEntropyLoss()

    best_acc = 0

    for epoch in range(EPOCHS):

        model.train()

        epoch_loss = 0

        correct = 0
        total = 0

        for _ in range(
            EPISODES_PER_EPOCH
        ):

            support, query = train_data.sample_episode()

            sx = torch.stack([

                load_image(
                    x[0],
                    train=True
                )

                for x in support

            ]).to(DEVICE)

            sy = torch.tensor([
                x[1]
                for x in support
            ]).to(DEVICE)

            qx = torch.stack([

                load_image(
                    x[0],
                    train=True
                )

                for x in query

            ]).to(DEVICE)

            qy = torch.tensor([
                x[1]
                for x in query
            ]).to(DEVICE)

            emb_s = model(sx)

            emb_q = model(qx)

            prototypes = compute_prototypes(
                emb_s,
                sy
            )

            logits = cosine_logits(
                emb_q,
                prototypes
            )

            loss = criterion(
                logits,
                qy
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

            preds = torch.argmax(
                logits,
                dim=1
            )

            correct += (
                preds == qy
            ).sum().item()

            total += qy.size(0)

        train_acc = correct / total

        val_acc = validate(
            model,
            train_data,
            val_data
        )

        log(
            f"Epoch {epoch+1} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        torch.save(
            model.state_dict(),
            os.path.join(
                CHECKPOINT_DIR,
                "latest.pth"
            )
        )

        if val_acc > best_acc:

            best_acc = val_acc

            torch.save(
                model.state_dict(),
                os.path.join(
                    CHECKPOINT_DIR,
                    "best.pth"
                )
            )

            log(
                "✅ Best model updated"
            )

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    train()