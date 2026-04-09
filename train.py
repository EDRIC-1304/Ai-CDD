# import os
# import random
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import cv2

# from log import log

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATA_DIR = r"G:\Ai-CDD\data\preprocessed\stage4\train"
# VAL_DIR = r"G:\Ai-CDD\data\preprocessed\stage4\val"

# CHECKPOINT_DIR = "checkpoints"
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# IMG_SIZE = 224
# EPISODES = 10000
# LR = 1e-3

# # -----------------------------
# # DATA LOADER (EPISODIC)
# # -----------------------------
# class Dataset:
#     def __init__(self, root):
#         self.data = {"NORMAL": [], "TUBERCULOSIS": []}

#         for cls in self.data:
#             path = os.path.join(root, cls)
#             self.data[cls] = [os.path.join(path, f) for f in os.listdir(path)]

#     def sample_episode(self, k=5):
#         support = []
#         query = []

#         for label, cls in enumerate(["NORMAL", "TUBERCULOSIS"]):
#             samples = random.sample(self.data[cls], k + k)

#             for i in range(k):
#                 support.append((samples[i], label))
#             for i in range(k, 2*k):
#                 query.append((samples[i], label))

#         return support, query


# def load_image(path):
#     img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
#     img = img.astype(np.float32) / 255.0
#     return torch.tensor(img).unsqueeze(0)


# # -----------------------------
# # MODEL
# # -----------------------------
# class CNN(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.net = nn.Sequential(
#             nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
#             nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
#             nn.MaxPool2d(2),

#             nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
#             nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
#             nn.MaxPool2d(2),

#             nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
#             nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
#             nn.MaxPool2d(2),

#             nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
#             nn.AdaptiveAvgPool2d(1)
#         )

#     def forward(self, x):
#         x = self.net(x)
#         return x.view(x.size(0), -1)


# # -----------------------------
# # PROTOTYPE LOGIC
# # -----------------------------
# def compute_prototypes(embeddings, labels):
#     prototypes = []
#     for c in [0, 1]:
#         class_emb = embeddings[labels == c]
#         prototypes.append(class_emb.mean(0))
#     return torch.stack(prototypes)


# def euclidean_dist(x, y):
#     return ((x.unsqueeze(1) - y.unsqueeze(0))**2).sum(2)


# # -----------------------------
# # TRAIN LOOP
# # -----------------------------
# def train():
#     dataset = Dataset(DATA_DIR)
#     model = CNN().to(DEVICE)
#     optimizer = optim.Adam(model.parameters(), lr=LR)

#     start_episode = 0
#     best_loss = float("inf")

#     # RESUME
#     checkpoint_path = os.path.join(CHECKPOINT_DIR, "latest.pth")
#     if os.path.exists(checkpoint_path):
#         ckpt = torch.load(checkpoint_path)
#         model.load_state_dict(ckpt["model"])
#         optimizer.load_state_dict(ckpt["opt"])
#         start_episode = ckpt["episode"]
#         best_loss = ckpt["best_loss"]
#         log(f"Resumed from episode {start_episode}")

#     for episode in range(start_episode, EPISODES):

#         support, query = dataset.sample_episode()

#         sx = torch.stack([load_image(x[0]) for x in support]).to(DEVICE)
#         sy = torch.tensor([x[1] for x in support]).to(DEVICE)

#         qx = torch.stack([load_image(x[0]) for x in query]).to(DEVICE)
#         qy = torch.tensor([x[1] for x in query]).to(DEVICE)

#         emb_s = model(sx)
#         emb_q = model(qx)

#         prototypes = compute_prototypes(emb_s, sy)

#         dists = euclidean_dist(emb_q, prototypes)
#         loss = nn.CrossEntropyLoss()(-dists, qy)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if episode % 50 == 0:
#             log(f"Episode {episode} | Loss: {loss.item():.4f}")

#         # SAVE LATEST
#         torch.save({
#             "model": model.state_dict(),
#             "opt": optimizer.state_dict(),
#             "episode": episode,
#             "best_loss": best_loss
#         }, checkpoint_path)

#         # SAVE BEST
#         if loss.item() < best_loss:
#             best_loss = loss.item()
#             torch.save(model.state_dict(),
#                        os.path.join(CHECKPOINT_DIR, "best.pth"))
#             log(f"🔥 New best model saved at episode {episode}")


# if __name__ == "__main__":
#     train()


import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import torch.nn.functional as F

from log import log

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = r"G:\Ai-CDD\data\preprocessed\stage4\train"
VAL_DIR = r"G:\Ai-CDD\data\preprocessed\stage4\val"

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

IMG_SIZE = 224
LR = 1e-3

EPISODES_PER_EPOCH = 100
EPOCHS = 100

# -----------------------------
# DATASET
# -----------------------------
class Dataset:
    def __init__(self, root):
        self.data = {"NORMAL": [], "TUBERCULOSIS": []}

        for cls in self.data:
            path = os.path.join(root, cls)
            self.data[cls] = [os.path.join(path, f) for f in os.listdir(path)]

    def sample_episode(self, k=5):
        support = []
        query = []

        for label, cls in enumerate(["NORMAL", "TUBERCULOSIS"]):
            samples = random.sample(self.data[cls], k * 2)

            for i in range(k):
                support.append((samples[i], label))
            for i in range(k, 2*k):
                query.append((samples[i], label))

        return support, query


class FullDataset:
    def __init__(self, root):
        self.samples = []

        for label, cls in enumerate(["NORMAL", "TUBERCULOSIS"]):
            path = os.path.join(root, cls)
            for f in os.listdir(path):
                self.samples.append((os.path.join(path, f), label))


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    return torch.tensor(img).unsqueeze(0)


# -----------------------------
# MODEL
# -----------------------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

    # def forward(self, x):
    #     x = self.net(x)
    #     return x.view(x.size(0), -1)
    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return F.normalize(x, dim=1)  


# -----------------------------
# PROTOTYPE LOGIC
# -----------------------------
def compute_prototypes(embeddings, labels):
    prototypes = []
    for c in [0, 1]:
        class_emb = embeddings[labels == c]
        prototypes.append(class_emb.mean(0))
    return torch.stack(prototypes)


def euclidean_dist(x, y):
    return ((x.unsqueeze(1) - y.unsqueeze(0))**2).sum(2)


# -----------------------------
# VALIDATION
# -----------------------------
def validate(model, dataset):
    model.eval()

    correct = 0
    total = 0

    tb_tp = 0
    tb_fn = 0

    # 🔥 Build prototypes from entire validation set
    embeddings = []
    labels = []

    with torch.no_grad():
        for path, label in dataset.samples:
            img = load_image(path).unsqueeze(0).to(DEVICE)
            emb = model(img)

            embeddings.append(emb)
            labels.append(label)

    embeddings = torch.cat(embeddings)
    labels = torch.tensor(labels).to(DEVICE)

    prototypes = compute_prototypes(embeddings, labels)

    # 🔥 Now classify using distance
    with torch.no_grad():
        for path, label in dataset.samples:

            img = load_image(path).unsqueeze(0).to(DEVICE)
            emb = model(img)

            dists = euclidean_dist(emb, prototypes)
            pred = torch.argmax(-dists, dim=1).item()

            if pred == label:
                correct += 1

            if label == 1:
                if pred == 1:
                    tb_tp += 1
                else:
                    tb_fn += 1

            total += 1

    acc = correct / total
    recall_tb = tb_tp / (tb_tp + tb_fn + 1e-8)

    return acc, recall_tb

# -----------------------------
# TRAIN
# -----------------------------
def train():
    train_data = Dataset(DATA_DIR)
    val_data = FullDataset(VAL_DIR)

    model = CNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_loss = float("inf")

    checkpoint_path = os.path.join(CHECKPOINT_DIR, "latest.pth")

    for epoch in range(EPOCHS):

        model.train()

        epoch_loss = 0
        correct = 0
        total = 0

        for _ in range(EPISODES_PER_EPOCH):

            support, query = train_data.sample_episode()

            sx = torch.stack([load_image(x[0]) for x in support]).to(DEVICE)
            sy = torch.tensor([x[1] for x in support]).to(DEVICE)

            qx = torch.stack([load_image(x[0]) for x in query]).to(DEVICE)
            qy = torch.tensor([x[1] for x in query]).to(DEVICE)

            emb_s = model(sx)
            emb_q = model(qx)

            prototypes = compute_prototypes(emb_s, sy)
            prototypes = F.normalize(prototypes, dim=1)
            if _ == 0:  # first episode only
                print("Proto distance:", torch.norm(prototypes[0] - prototypes[1]).item())

            dists = euclidean_dist(emb_q, prototypes)
            # loss = nn.CrossEntropyLoss()(-dists, qy)
            loss = nn.CrossEntropyLoss()(-dists / 0.1, qy)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            preds = torch.argmax(-dists, dim=1)
            correct += (preds == qy).sum().item()
            total += qy.size(0)

        train_loss = epoch_loss / EPISODES_PER_EPOCH
        train_acc = correct / total

        # VALIDATION
        val_acc, val_recall_tb = validate(model, val_data)

        log(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | TB Recall: {val_recall_tb:.4f}")

        # SAVE
        torch.save({
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss
        }, checkpoint_path)

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best.pth"))
            log(f"New best model at epoch {epoch}")


if __name__ == "__main__":
    train()
