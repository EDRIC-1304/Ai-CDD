from tensorflow.keras.models import load_model
import numpy as np
import cv2
HOME_DIR = "D:/FY_Project/Ai-CDD"
model = load_model(HOME_DIR+"/models_saved/modality_classifier.h5")
print("Model loaded successfully")

IMG_SIZE = 224

def preprocess_test_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Image not found or unreadable")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)  # (224,224,1)
    img = np.expand_dims(img, axis=0)   # (1,224,224,1)

    return img


TEST_IMAGE_PATH = HOME_DIR+"/data/raw/test/xray/sample4.jfif"

img = preprocess_test_image(TEST_IMAGE_PATH)

pred = model.predict(img)[0][0]

print("Raw model output:", pred)

if pred >= 0.5:
    print("Prediction: CT scan")
else:
    print("Prediction: X-ray")