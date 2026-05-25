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
# KEEP LARGEST COMPONENTS
# -------------------------------
def keep_largest_components(mask, num_components=2):

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = np.argsort(areas)[-num_components:] + 1

    clean = np.zeros_like(mask)

    for idx in largest:
        clean[labels == idx] = 1

    return clean


# -------------------------------
# FILL HOLES
# -------------------------------
def fill_holes(mask):

    mask = (mask * 255).astype(np.uint8)

    h, w = mask.shape
    flood = mask.copy()

    flood_mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(flood, flood_mask, (0, 0), 255)

    flood_inv = cv2.bitwise_not(flood)

    filled = mask | flood_inv

    return (filled > 0).astype(np.uint8)


# -------------------------------
# SEGMENTATION FUNCTION
# -------------------------------
def segment_ct_lung(image, threshold=0.5, morph_radius=3):

    image = np.array(image)

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    orig = image.copy()


    # -------------------------------
    # PREPROCESS
    # -------------------------------
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = img[np.newaxis, ..., np.newaxis]


    # -------------------------------
    # MODEL PREDICTION
    # -------------------------------
    pred = model.predict(img, verbose=0)[0, ..., 0]


    # -------------------------------
    # BINARY MASK
    # -------------------------------
    binary_mask = (pred > threshold).astype(np.uint8)


    # -------------------------------
    # MORPHOLOGY
    # -------------------------------
    morph_mask = opening(binary_mask, disk(morph_radius))


    # -------------------------------
    # KEEP ONLY LUNGS
    # -------------------------------
    morph_mask = keep_largest_components(morph_mask, 2)


    # -------------------------------
    # FILL HOLES
    # -------------------------------
    morph_mask = fill_holes(morph_mask)


    # -------------------------------
    # RESIZE BACK
    # -------------------------------
    morph_mask = cv2.resize(
        morph_mask,
        (orig.shape[1], orig.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )


    # -------------------------------
    # APPLY MASK
    # -------------------------------
    mask_uint8 = (morph_mask * 255).astype(np.uint8)

    masked_ct = cv2.bitwise_and(orig, orig, mask=mask_uint8)


    # -------------------------------
    # SAVE OUTPUT
    # -------------------------------
    cv2.imwrite(
        HOME_DIR + "/data/processed/CT/masked_ct.png",
        masked_ct
    )

    print("✅ CT lung masking completed")

    return orig, binary_mask, masked_ct