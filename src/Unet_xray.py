import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import re


IMG_SIZE = 256

IMAGE_DIR = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/raw/train/preprocessed_xray_images"
MASK_DIR  = "C:/Users/Ant PC/Desktop/22co12 FY_PROJ/AI-CDD/Ai-CDD/data/raw/train/xray-mask"

# -----------------------------
# LOAD DATA
# -----------------------------

def extract_id(filename):
    # Extract numeric ID from filename
    match = re.search(r'\d+', filename)
    return match.group() if match else None


def load_data(image_dir, mask_dir):
    images = []
    masks = []

    image_files = os.listdir(image_dir)
    mask_files  = os.listdir(mask_dir)

    # Create mapping: ID → filename
    image_dict = {}
    mask_dict = {}

    for file in image_files:
        img_id = extract_id(file)
        if img_id:
            image_dict[img_id] = file

    for file in mask_files:
        mask_id = extract_id(file)
        if mask_id:
            mask_dict[mask_id] = file

    common_ids = sorted(set(image_dict.keys()) & set(mask_dict.keys()))

    print(f"✅ Total matched pairs: {len(common_ids)}")

    if len(common_ids) == 0:
        raise ValueError("❌ No matching image-mask pairs found!")

    for idx in common_ids:
        img_path = os.path.join(image_dir, image_dict[idx])
        mask_path = os.path.join(mask_dir, mask_dict[idx])

        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)

        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        mask = (mask > 127).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)

        images.append(img)
        masks.append(mask)

    print(f"✅ Loaded images: {len(images)}")

    return np.array(images), np.array(masks)
# -----------------------------
# LOAD + SPLIT
# -----------------------------
X, y = load_data(IMAGE_DIR, MASK_DIR)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train:", X_train.shape)
print("Val:", X_val.shape)

# -----------------------------
# SIMPLE U-NET MODEL
# -----------------------------
def conv_block(x, filters):
    x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x

def build_unet(input_shape=(256,256,1)):
    inputs = tf.keras.Input(input_shape)

    # Encoder
    c1 = conv_block(inputs, 32)
    p1 = tf.keras.layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 64)
    p2 = tf.keras.layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 128)
    p3 = tf.keras.layers.MaxPooling2D()(c3)

    c4 = conv_block(p3, 256)
    p4 = tf.keras.layers.MaxPooling2D()(c4)

    # Bottleneck
    bn = conv_block(p4, 512)

    # Decoder
    u1 = tf.keras.layers.UpSampling2D()(bn)
    u1 = tf.keras.layers.Concatenate()([u1, c4])
    c5 = conv_block(u1, 256)

    u2 = tf.keras.layers.UpSampling2D()(c5)
    u2 = tf.keras.layers.Concatenate()([u2, c3])
    c6 = conv_block(u2, 128)

    u3 = tf.keras.layers.UpSampling2D()(c6)
    u3 = tf.keras.layers.Concatenate()([u3, c2])
    c7 = conv_block(u3, 64)

    u4 = tf.keras.layers.UpSampling2D()(c7)
    u4 = tf.keras.layers.Concatenate()([u4, c1])
    c8 = conv_block(u4, 32)

    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(c8)

    return tf.keras.Model(inputs, outputs)

model = build_unet()

# -----------------------------
# LOSS (IMPORTANT)
# -----------------------------
def dice_loss(y_true, y_pred):
    smooth = 1e-6
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return 1 - (2. * intersection + smooth) / (union + smooth)

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

# -----------------------------
# COMPILE
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=combined_loss,
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# TRAIN
# -----------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=8
)

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save("xray_unet_lung.h5")

print("✅ U-Net training completed")