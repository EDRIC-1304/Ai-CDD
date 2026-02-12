import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import opening, disk

HOME_DIR = "D:/FY_Project/Ai-CDD"
model = tf.keras.models.load_model(HOME_DIR+"/models_saved/ct_unet_lung.keras")



TEST_IMAGE_PATH = HOME_DIR+"/data/raw/test/ct/downloactsample4.jfif"

# 1. Load image
img = cv2.imread(TEST_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
assert img is not None, "Image not loaded. Path is wrong."

orig = img.copy()

# 2. Resize to model input size
img_resized = cv2.resize(img, (256, 256))

# 3. Normalize
img_norm = img_resized / 255.0
img_norm = img_norm[np.newaxis, ..., np.newaxis]  # (1,256,256,1)

# 4. Predict
pred = model.predict(img_norm, verbose=0)[0, ..., 0]

# 5. Threshold
mask = (pred > 0.5).astype(np.uint8)

# 6. Morphological opening
mask_clean = opening(mask, disk(3))

# 7. Resize mask back to original size
mask_full = cv2.resize(
    mask_clean,
    (orig.shape[1], orig.shape[0]),
    interpolation=cv2.INTER_NEAREST
)

# 8. Overlay
overlay = orig.copy()
overlay[mask_full == 1] = 255

# 9. DISPLAY RESULTS
plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.title("Original")
plt.imshow(orig, cmap="gray")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.title("U-Net Segmentation")
plt.imshow(pred, cmap="gray")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.title("Morphological Opening")
plt.imshow(mask_full, cmap="gray")
plt.axis("off")

# plt.subplot(1, 4, 4)
# plt.title("Overlay")
# plt.imshow(overlay, cmap="gray")
# plt.axis("off")

plt.show()

# 10. SAVE OUTPUT
cv2.imwrite(HOME_DIR+"/data/processed/CT/output_mask.png", mask_full * 255)
# cv2.imwrite("/kaggle/working/output_overlay.png", overlay)

print("Saved:")
print("/data/processed/CT/output_mask.png")
# print("/kaggle/working/output_overlay.png")
