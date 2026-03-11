import torch
import cv2
import sys
import os
import numpy as np
from typing import Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_loader import TrainedModelLoader


def preprocess_image(img_path: str) -> torch.Tensor:
    """Preprocess image for model prediction."""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
        
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")
    
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = torch.tensor(img).permute(2, 0, 1).float()
    img = img.unsqueeze(0)  # Add batch dimension
    
    return img


def predict(image_path: str, model_path: str = "models_saved/tb_moet.pth") -> Tuple[int, float]:
    """Predict disease from image using trained model."""
    # Initialize model loader
    loader = TrainedModelLoader(model_path)
    
    # Load the trained model
    model = loader.load_model()
    
    # Preprocess image
    img = preprocess_image(image_path)
    
    # Make prediction
    prediction, confidence = loader.get_confidence_and_prediction(img)
    
    return prediction, confidence


def main():
    """Main prediction function."""
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path> [model_path]")
        print("Example: python predict.py image.jpg models_saved/tb_moet.pth")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "models_saved/tb_moet.pth"
    
    try:
        prediction, confidence = predict(image_path, model_path)
        confidence *= 100
        
        # Determine result
        result = "NORMAL" if prediction == 0 else "DISEASE DETECTED"
        
        # Output
        print("\n==============================")
        print("   MoET Diagnosis Result")
        print("==============================")
        print(f"Image      : {image_path}")
        print(f"Model      : {os.path.basename(model_path)}")
        print(f"Prediction : {result}")
        print(f"Confidence : {confidence:.2f}%")
        print("==============================\n")
        
    except Exception as e:
        print(f" Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
