import cv2
import numpy as np
import tensorflow as tf
from skimage.morphology import opening, disk

HOME_DIR = "D:/FY_Project/Ai-CDD"
IMG_SIZE = 256

# -------------------------------
# LOAD CT U-NET MODEL
# -------------------------------
model = tf.keras.models.load_model(
    HOME_DIR + "/models_saved/ct_unet_lung.keras"
)
print("✅ CT U-Net loaded")

# -------------------------------
# SEGMENTATION FUNCTION
# -------------------------------
def segment_ct_lung(image, threshold=0.5, morph_radius=3):
    # PIL → NumPy
    image = np.array(image)

    # Convert RGB → GRAY
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    orig = image.copy()

    # Preprocess
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = img[np.newaxis, ..., np.newaxis]  # (1,256,256,1)

    # Predict
    pred = model.predict(img, verbose=0)[0, ..., 0]

    # Binary mask
    binary_mask = (pred > threshold).astype(np.uint8)

    # Morphological opening
    morph_mask = opening(binary_mask, disk(morph_radius))

    # Resize masks back
    binary_mask = cv2.resize(
        binary_mask,
        (orig.shape[1], orig.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    morph_mask = cv2.resize(
        morph_mask,
        (orig.shape[1], orig.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    cv2.imwrite(HOME_DIR + "/data/processed/CT/morph_mask.png", morph_mask * 255)
    print("Segmentation masks saved to disk")
    
    return orig, binary_mask, morph_mask
