import cv2
import numpy as np
from tensorflow.keras.models import load_model

HOME_DIR = "D:/FY_Project/Ai-CDD"
IMG_SIZE = 224

# --------------------------------------------------
# LOAD MODEL ONCE
# --------------------------------------------------
model = load_model(HOME_DIR + "/models_saved/modality_classifier.h5")
print("✅ Modality classifier loaded")

# --------------------------------------------------
# PREPROCESS IMAGE OBJECT (NOT PATH)
# --------------------------------------------------
def preprocess_image_object(image):
    if image is None:
        raise ValueError("Input image is None")
    image = np.array(image)
    # If image is BGR (cv2 default), convert to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.astype(np.float32) / 255.0

    image = np.expand_dims(image, axis=-1)  # (224,224,1)
    image = np.expand_dims(image, axis=0)   # (1,224,224,1)

    return image

# --------------------------------------------------
# PREDICTION FUNCTION (IMAGE OBJECT)
# --------------------------------------------------
def predict_scan_type(image):
    img = preprocess_image_object(image)
    pred = model.predict(img, verbose=0)[0][0]

    print("Raw model output:", pred)

    if pred >= 0.5:
        return 1
    else:
        return 2
