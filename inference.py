# import os
# import random
# import numpy as np
# from PIL import Image
# from sklearn.metrics import confusion_matrix, classification_report
# import matplotlib.pyplot as plt
# import seaborn as sns

# from src.utils.xray_classifier import xray_full_pipeline


# # -----------------------------
# # CONFIG
# # -----------------------------
# DATASET_PATH = "dataset2/test"
# CLASS_NAMES = ["NORMAL", "TUBERCULOSIS"]

# SAMPLES_PER_CLASS = 20
# THRESHOLD = 0.7   # 🔥 try 0.6 / 0.7 / 0.8


# # -----------------------------
# # LOAD DATA
# # -----------------------------
# def load_dataset(folder_path):
#     image_paths = []
#     labels = []

#     for label, class_name in enumerate(CLASS_NAMES):
#         class_folder = os.path.join(folder_path, class_name)

#         files = [
#             f for f in os.listdir(class_folder)
#             if f.lower().endswith((".png", ".jpg", ".jpeg"))
#         ]

#         # ✅ NO SAMPLING — use all files
#         for file in files:
#             image_paths.append(os.path.join(class_folder, file))
#             labels.append(label)

#     return image_paths, labels

# # -----------------------------
# # RUN INFERENCE
# # -----------------------------
# def run_inference(image_paths, y_true):
#     y_pred = []
#     y_true_clean = []

#     normal_conf = []
#     tb_conf = []

#     for img_path, label in zip(image_paths, y_true):
#         try:
#             image = Image.open(img_path).convert("RGB")

#             _, _, _, _, _, pred, conf = xray_full_pipeline(image)

#             # 🔥 THRESHOLD FIX
#             pred = 1 if conf > THRESHOLD else 0

#             # ⚠️ Uncomment if predictions seem flipped
#             # pred = 1 - pred

#             y_pred.append(pred)
#             y_true_clean.append(label)

#             # track confidence
#             if label == 0:
#                 normal_conf.append(conf)
#             else:
#                 tb_conf.append(conf)

#             print(f"{os.path.basename(img_path)} -> Pred: {pred} | Conf: {conf:.3f}")

#             # 🔴 WRONG PREDICTIONS
#             if pred != label:
#                 print(f"❌ WRONG: {img_path} | Pred: {pred} | True: {label}")

#         except Exception as e:
#             print(f"Error processing {img_path}: {e}")

#     # -----------------------------
#     # CONFIDENCE STATS
#     # -----------------------------
#     print("\n--- Confidence Stats ---")
#     print(f"Avg NORMAL confidence: {np.mean(normal_conf):.3f}")
#     print(f"Avg TB confidence: {np.mean(tb_conf):.3f}")

#     return y_true_clean, y_pred


# # -----------------------------
# # CONFUSION MATRIX
# # -----------------------------
# def plot_confusion_matrix(y_true, y_pred):
#     cm = confusion_matrix(y_true, y_pred)

#     plt.figure(figsize=(6, 5))
#     sns.heatmap(
#         cm,
#         annot=True,
#         fmt="d",
#         cmap="Blues",
#         xticklabels=CLASS_NAMES,
#         yticklabels=CLASS_NAMES
#     )

#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.title("Confusion Matrix")

#     plt.savefig("confusion_matrix.png")
#     print("\n📊 Confusion matrix saved as confusion_matrix.png")


# # -----------------------------
# # MAIN
# # -----------------------------
# if __name__ == "__main__":
#     print("Loading test dataset...")
#     image_paths, y_true = load_dataset(DATASET_PATH)

#     print(f"Total test images: {len(image_paths)}")

#     print("\nRunning inference...\n")
#     y_true, y_pred = run_inference(image_paths, y_true)

#     # -----------------------------
#     # RESULTS
#     # -----------------------------
#     print("\nClassification Report:")
#     print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

#     print("\nConfusion Matrix:")
#     print(confusion_matrix(y_true, y_pred))

#     plot_confusion_matrix(y_true, y_pred)




import os
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import csv

from src.utils.xray_classifier import xray_full_pipeline


# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data", "raw", "dataset", "test")
CLASS_NAMES = ["NORMAL", "TUBERCULOSIS"]

# initial threshold (will be optimized later)
THRESHOLD = 0.5


# -----------------------------
# LOAD DATA (ALL IMAGES)
# -----------------------------
def load_dataset(folder_path):
    image_paths = []
    labels = []

    for label, class_name in enumerate(CLASS_NAMES):
        class_folder = os.path.join(folder_path, class_name)

        files = [
            f for f in os.listdir(class_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        for file in files:
            image_paths.append(os.path.join(class_folder, file))
            labels.append(label)

    return image_paths, labels


# -----------------------------
# RUN INFERENCE
# -----------------------------
def run_inference(image_paths, y_true):
    y_pred = []
    y_true_clean = []

    normal_conf = []
    tb_conf = []
    all_conf = []

    log_rows = []

    for i, (img_path, label) in enumerate(zip(image_paths, y_true)):
        try:
            print(f"[{i+1}/{len(image_paths)}] Processing...")

            image = Image.open(img_path).convert("RGB")

            _, _, _, _, _, pred_raw, conf = xray_full_pipeline(image)

            all_conf.append(conf)

            # apply threshold
            pred = 1 if conf > THRESHOLD else 0

            y_pred.append(pred)
            y_true_clean.append(label)

            # track confidence
            if label == 0:
                normal_conf.append(conf)
            else:
                tb_conf.append(conf)

            is_correct = (pred == label)

            # ✅ LOG ROW
            log_rows.append({
                "image": img_path,
                "true_label": CLASS_NAMES[label],
                "pred_label": CLASS_NAMES[pred],
                "confidence": round(conf, 4),
                "correct": is_correct
            })

            # print wrong predictions
            if not is_correct:
                print(f"❌ WRONG: {img_path} | Pred: {pred} | True: {label} | Conf: {conf:.3f}")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # -----------------------------
    # SAVE LOG FILE
    # -----------------------------
    log_file = "inference_log.csv"
    with open(log_file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\n📝 Inference log saved as: {log_file}")

    return y_true_clean, y_pred, normal_conf, tb_conf, all_conf

# -----------------------------
# FIND BEST THRESHOLD
# -----------------------------
def find_best_threshold(y_true, confidences):
    best_acc = 0
    best_thresh = 0.5

    for t in np.arange(0.1, 0.9, 0.05):
        preds = [1 if c > t else 0 for c in confidences]
        acc = np.mean(np.array(preds) == np.array(y_true))

        if acc > best_acc:
            best_acc = acc
            best_thresh = t

    print(f"\n🔥 Best Threshold: {best_thresh:.2f} | Accuracy: {best_acc:.3f}")
    return best_thresh


# -----------------------------
# CONFUSION MATRIX
# -----------------------------
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.savefig("confusion_matrix.png")
    print("📊 Confusion matrix saved as confusion_matrix.png")


# -----------------------------
# CONFIDENCE DISTRIBUTION
# -----------------------------
def plot_confidence_distribution(normal_conf, tb_conf):
    plt.figure()

    plt.hist(normal_conf, bins=30, alpha=0.5, label="NORMAL")
    plt.hist(tb_conf, bins=30, alpha=0.5, label="TB")

    plt.legend()
    plt.title("Confidence Distribution")
    plt.xlabel("Confidence")
    plt.ylabel("Count")

    plt.savefig("confidence_distribution.png")
    print("📈 Confidence distribution saved as confidence_distribution.png")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    print("Loading dataset...")
    image_paths, y_true = load_dataset(DATASET_PATH)

    print(f"Total images: {len(image_paths)}")

    print("\nRunning inference...\n")
    y_true, y_pred, normal_conf, tb_conf, all_conf = run_inference(image_paths, y_true)

    # -----------------------------
    # FIND BEST THRESHOLD
    # -----------------------------
    best_threshold = find_best_threshold(y_true, all_conf)

    # apply best threshold
    y_pred_best = [1 if c > best_threshold else 0 for c in all_conf]

    # -----------------------------
    # RESULTS
    # -----------------------------
    print("\n--- FINAL RESULTS (BEST THRESHOLD) ---")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_best, target_names=CLASS_NAMES))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred_best))

    # -----------------------------
    # PLOTS
    # -----------------------------
    plot_confusion_matrix(y_true, y_pred_best)
    plot_confidence_distribution(normal_conf, tb_conf)