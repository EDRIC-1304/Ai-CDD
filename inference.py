
# import os
# import csv
# import numpy as np

# from PIL import Image

# from sklearn.metrics import (
#     confusion_matrix,
#     classification_report,
#     accuracy_score
# )

# import matplotlib.pyplot as plt
# import seaborn as sns

# from src.utils.xray_classifier import xray_full_pipeline

# # -----------------------------
# # CONFIG
# # -----------------------------
# BASE_DIR = os.path.dirname(
#     os.path.abspath(__file__)
# )

# DATASET_PATH = os.path.join(
#     BASE_DIR,
#     "data",
#     "raw",
#     "dataset",
#     "test"
# )

# CLASS_NAMES = [
#     "NORMAL",
#     "TUBERCULOSIS"
# ]

# VALID_EXT = (
#     ".png",
#     ".jpg",
#     ".jpeg",
#     ".bmp"
# )

# # -----------------------------
# # LOAD DATASET
# # -----------------------------
# def load_dataset(folder_path):

#     image_paths = []
#     labels = []

#     for label, class_name in enumerate(CLASS_NAMES):

#         class_folder = os.path.join(
#             folder_path,
#             class_name
#         )

#         if not os.path.exists(class_folder):

#             print(f"❌ Missing: {class_folder}")

#             continue

#         files = sorted([
#             f for f in os.listdir(class_folder)
#             if f.lower().endswith(VALID_EXT)
#         ])

#         for file in files:

#             image_paths.append(
#                 os.path.join(class_folder, file)
#             )

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

#     all_conf = []

#     log_rows = []

#     processed = 0
#     failed = 0

#     for i, (img_path, label) in enumerate(
#         zip(image_paths, y_true)
#     ):

#         try:

#             print(
#                 f"[{i+1}/{len(image_paths)}] "
#                 f"Processing..."
#             )

#             # -------------------------
#             # LOAD AS GRAYSCALE
#             # IMPORTANT
#             # -------------------------
#             image = Image.open(
#                 img_path
#             ).convert("L")

#             # -------------------------
#             # FULL PIPELINE
#             # -------------------------
#             results = xray_full_pipeline(image)

#             # -------------------------
#             # SAFETY CHECK
#             # -------------------------
#             if results is None:
#                 failed += 1
#                 continue

#             _, _, _, _, _, pred_raw, conf = results

#             pred = int(pred_raw)

#             conf = float(conf)

#             y_pred.append(pred)

#             y_true_clean.append(label)

#             all_conf.append(conf)

#             if label == 0:
#                 normal_conf.append(conf)
#             else:
#                 tb_conf.append(conf)

#             is_correct = (pred == label)

#             log_rows.append({
#                 "image": img_path,
#                 "true_label": CLASS_NAMES[label],
#                 "pred_label": CLASS_NAMES[pred],
#                 "confidence": round(conf, 4),
#                 "correct": is_correct
#             })

#             if not is_correct:

#                 print(
#                     f"❌ WRONG | "
#                     f"Pred: {CLASS_NAMES[pred]} | "
#                     f"True: {CLASS_NAMES[label]} | "
#                     f"Conf: {conf:.4f}"
#                 )

#             processed += 1

#         except Exception as e:

#             failed += 1

#             print(f"\n❌ Error processing:")
#             print(img_path)
#             print(e)

#     # -----------------------------
#     # SAVE LOG
#     # -----------------------------
#     if len(log_rows) > 0:

#         log_file = "inference_log.csv"

#         with open(
#             log_file,
#             mode="w",
#             newline="",
#             encoding="utf-8"
#         ) as file:

#             writer = csv.DictWriter(
#                 file,
#                 fieldnames=log_rows[0].keys()
#             )

#             writer.writeheader()

#             writer.writerows(log_rows)

#         print(f"\n📝 Log saved:")
#         print(log_file)

#     print("\n-----------------------------")
#     print(f"✔ Processed: {processed}")
#     print(f"❌ Failed: {failed}")
#     print("-----------------------------")

#     return (
#         y_true_clean,
#         y_pred,
#         normal_conf,
#         tb_conf,
#         all_conf
#     )

# # -----------------------------
# # CONFUSION MATRIX
# # -----------------------------
# def plot_confusion_matrix(y_true, y_pred):

#     cm = confusion_matrix(
#         y_true,
#         y_pred
#     )

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

#     plt.tight_layout()

#     plt.savefig(
#         "confusion_matrix.png"
#     )

#     print(
#         "📊 Saved: confusion_matrix.png"
#     )

# # -----------------------------
# # CONFIDENCE DISTRIBUTION
# # -----------------------------
# def plot_confidence_distribution(
#     normal_conf,
#     tb_conf
# ):

#     plt.figure(figsize=(7, 5))

#     plt.hist(
#         normal_conf,
#         bins=30,
#         alpha=0.5,
#         label="NORMAL"
#     )

#     plt.hist(
#         tb_conf,
#         bins=30,
#         alpha=0.5,
#         label="TB"
#     )

#     plt.legend()

#     plt.title(
#         "Confidence Distribution"
#     )

#     plt.xlabel("Confidence")

#     plt.ylabel("Count")

#     plt.tight_layout()

#     plt.savefig(
#         "confidence_distribution.png"
#     )

#     print(
#         "📈 Saved: confidence_distribution.png"
#     )

# # -----------------------------
# # MAIN
# # -----------------------------
# if __name__ == "__main__":

#     print("Loading dataset...")

#     image_paths, y_true = load_dataset(
#         DATASET_PATH
#     )

#     print(f"Total images: {len(image_paths)}")

#     print("\nRunning inference...\n")

#     (
#         y_true,
#         y_pred,
#         normal_conf,
#         tb_conf,
#         all_conf
#     ) = run_inference(
#         image_paths,
#         y_true
#     )

#     # -----------------------------
#     # SAFETY
#     # -----------------------------
#     if len(y_pred) == 0:

#         print("\n❌ No predictions generated")

#         exit()

#     # -----------------------------
#     # RESULTS
#     # -----------------------------
#     print("\n=============================")
#     print("FINAL RESULTS")
#     print("=============================")

#     acc = accuracy_score(
#         y_true,
#         y_pred
#     )

#     print(f"\nAccuracy: {acc:.4f}")

#     print("\nClassification Report:\n")

#     print(
#         classification_report(
#             y_true,
#             y_pred,
#             target_names=CLASS_NAMES,
#             digits=4
#         )
#     )

#     print("\nConfusion Matrix:\n")

#     print(
#         confusion_matrix(
#             y_true,
#             y_pred
#         )
#     )

#     # -----------------------------
#     # CONFIDENCE STATS
#     # -----------------------------
#     print("\nConfidence Statistics")

#     print(
#         f"Mean Confidence: "
#         f"{np.mean(all_conf):.4f}"
#     )

#     print(
#         f"Min Confidence: "
#         f"{np.min(all_conf):.4f}"
#     )

#     print(
#         f"Max Confidence: "
#         f"{np.max(all_conf):.4f}"
#     )

#     # -----------------------------
#     # PLOTS
#     # -----------------------------
#     plot_confusion_matrix(
#         y_true,
#         y_pred
#     )

#     plot_confidence_distribution(
#         normal_conf,
#         tb_conf
#     )

#     print("\n✅ Inference complete")










import os
import csv
import numpy as np

from PIL import Image

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.xray_classifier import xray_full_pipeline

# =========================================================
# CONFIG
# =========================================================
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "dataset",
    "test"
)

CLASS_NAMES = [
    "NORMAL",
    "TUBERCULOSIS"
]

VALID_EXT = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp"
)

# =========================================================
# LOAD DATASET
# =========================================================
def load_dataset(folder_path):

    image_paths = []
    labels = []

    for label, class_name in enumerate(
        CLASS_NAMES
    ):

        class_folder = os.path.join(
            folder_path,
            class_name
        )

        if not os.path.exists(
            class_folder
        ):

            print(
                f"❌ Missing: {class_folder}"
            )

            continue

        files = sorted([

            f for f in os.listdir(
                class_folder
            )

            if f.lower().endswith(
                VALID_EXT
            )
        ])

        for file in files:

            image_paths.append(
                os.path.join(
                    class_folder,
                    file
                )
            )

            labels.append(label)

    return image_paths, labels

# =========================================================
# RUN INFERENCE
# =========================================================
def run_inference(
    image_paths,
    y_true
):

    y_pred = []
    y_true_clean = []

    normal_conf = []
    tb_conf = []

    all_conf = []

    log_rows = []

    processed = 0
    failed = 0

    for i, (
        img_path,
        label
    ) in enumerate(
        zip(image_paths, y_true)
    ):

        try:

            print(
                f"[{i+1}/{len(image_paths)}] "
                f"Processing..."
            )

            # ---------------------------------
            # LOAD IMAGE
            # ---------------------------------
            image = Image.open(
                img_path
            ).convert("L")

            # ---------------------------------
            # FULL PIPELINE
            # ---------------------------------
            results = xray_full_pipeline(
                image
            )

            if results is None:

                failed += 1

                continue

            # ---------------------------------
            # EXPECTED RETURNS
            # ---------------------------------
            (
                _,
                _,
                _,
                _,
                _,
                pred_raw,
                conf
            ) = results

            pred = int(pred_raw)

            conf = float(conf)

            conf = max(
                0.0,
                min(conf, 1.0)
            )

            y_pred.append(pred)

            y_true_clean.append(
                label
            )

            all_conf.append(conf)

            if pred == 0:
                normal_conf.append(conf)

            else:
                tb_conf.append(conf)

            is_correct = (
                pred == label
            )

            log_rows.append({

                "image": img_path,

                "true_label":
                CLASS_NAMES[label],

                "pred_label":
                CLASS_NAMES[pred],

                "confidence":
                round(conf, 4),

                "correct":
                is_correct
            })

            # ---------------------------------
            # WRONG PREDICTION
            # ---------------------------------
            if not is_correct:

                print(
                    f"❌ WRONG | "
                    f"Pred: {CLASS_NAMES[pred]} | "
                    f"True: {CLASS_NAMES[label]} | "
                    f"Conf: {conf:.4f}"
                )

            processed += 1

        except Exception as e:

            failed += 1

            print("\n❌ Error processing")

            print(img_path)

            print(e)

    # =====================================================
    # SAVE LOG
    # =====================================================
    if len(log_rows) > 0:

        log_file = (
            "inference_log.csv"
        )

        with open(
            log_file,
            mode="w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=log_rows[0].keys()
            )

            writer.writeheader()

            writer.writerows(
                log_rows
            )

        print(
            f"\n📝 Log saved:"
        )

        print(log_file)

    print("\n-------------------")

    print(
        f"✔ Processed: {processed}"
    )

    print(
        f"❌ Failed: {failed}"
    )

    print("-------------------")

    return (
        y_true_clean,
        y_pred,
        normal_conf,
        tb_conf,
        all_conf
    )

# =========================================================
# CONFUSION MATRIX
# =========================================================
def plot_confusion_matrix(
    y_true,
    y_pred
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

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

    plt.tight_layout()

    plt.savefig(
        "confusion_matrix.png"
    )

    print(
        "📊 Saved: confusion_matrix.png"
    )

# =========================================================
# CONFIDENCE DISTRIBUTION
# =========================================================
def plot_confidence_distribution(
    normal_conf,
    tb_conf
):

    plt.figure(figsize=(7, 5))

    plt.hist(
        normal_conf,
        bins=30,
        alpha=0.5,
        label="NORMAL"
    )

    plt.hist(
        tb_conf,
        bins=30,
        alpha=0.5,
        label="TB"
    )

    plt.legend()

    plt.title(
        "Confidence Distribution"
    )

    plt.xlabel("Confidence")

    plt.ylabel("Count")

    plt.tight_layout()

    plt.savefig(
        "confidence_distribution.png"
    )

    print(
        "📈 Saved: confidence_distribution.png"
    )

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    print(
        "Loading dataset..."
    )

    image_paths, y_true = (
        load_dataset(
            DATASET_PATH
        )
    )

    print(
        f"Total images: {len(image_paths)}"
    )

    print(
        "\nRunning inference...\n"
    )

    (
        y_true,
        y_pred,
        normal_conf,
        tb_conf,
        all_conf

    ) = run_inference(
        image_paths,
        y_true
    )

    # =====================================================
    # SAFETY
    # =====================================================
    if len(y_pred) == 0:

        print(
            "\n❌ No predictions generated"
        )

        exit()

    # =====================================================
    # METRICS
    # =====================================================
    acc = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (
        tn + fp + 1e-8
    )

    sensitivity = tp / (
        tp + fn + 1e-8
    )

    # =====================================================
    # RESULTS
    # =====================================================
    print("\n====================")
    print("FINAL RESULTS")
    print("====================")

    print(
        f"\nAccuracy      : {acc:.4f}"
    )

    print(
        f"TB Precision  : {precision:.4f}"
    )

    print(
        f"TB Recall     : {recall:.4f}"
    )

    print(
        f"TB F1 Score   : {f1:.4f}"
    )

    print(
        f"Specificity   : {specificity:.4f}"
    )

    print(
        f"Sensitivity   : {sensitivity:.4f}"
    )

    print(
        "\nClassification Report:\n"
    )

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0
        )
    )

    print(
        "\nConfusion Matrix:\n"
    )

    print(cm)

    # =====================================================
    # PREDICTION DISTRIBUTION
    # =====================================================
    normal_pred = np.sum(
        np.array(y_pred) == 0
    )

    tb_pred = np.sum(
        np.array(y_pred) == 1
    )

    print(
        "\nPrediction Distribution"
    )

    print(
        f"NORMAL Predictions : {normal_pred}"
    )

    print(
        f"TB Predictions     : {tb_pred}"
    )

    # =====================================================
    # CONFIDENCE STATS
    # =====================================================
    print(
        "\nConfidence Statistics"
    )

    print(
        f"Mean Confidence: "
        f"{np.mean(all_conf):.4f}"
    )

    print(
        f"Min Confidence: "
        f"{np.min(all_conf):.4f}"
    )

    print(
        f"Max Confidence: "
        f"{np.max(all_conf):.4f}"
    )

    # =====================================================
    # PLOTS
    # =====================================================
    plot_confusion_matrix(
        y_true,
        y_pred
    )

    plot_confidence_distribution(
        normal_conf,
        tb_conf
    )

    print(
        "\n✅ Inference complete"
    )