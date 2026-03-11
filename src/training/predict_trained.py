import torch
import cv2
import sys
import os
import numpy as np
from typing import Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_loader import TrainedModelLoader

def predict_with_trained_model(model_path: str, image_path: str) -> Tuple[int, float]:
    """
    Predict using trained model (supports .adding, .pth, .pt, .pkl formats)
    
    Args:
        model_path: Path to trained model file
        image_path: Path to input image
    
    Returns:
        Tuple of (prediction, confidence)
    """
    # Initialize model loader
    loader = TrainedModelLoader(model_path)
    
    try:
        # Load the trained model
        model = loader.load_model()
        
        # Preprocess image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = torch.tensor(img).permute(2, 0, 1).float()
        img = img.unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        prediction, confidence = loader.get_confidence_and_prediction(img)
        
        return prediction, confidence
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        raise

def main():
    """Main function for command-line prediction with trained models"""
    
    if len(sys.argv) < 2:
        print("Usage: python predict_trained.py <model_path> [image_path]")
        print("Examples:")
        print("  python predict_trained.py lung.adding xray.jpg")
        print("  python predict_trained.py models_saved/tb_moet.pth xray.jpg")
        print("  python predict_trained.py lung.adding  # Will prompt for image")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    # Get image path from command line or prompt user
    if len(sys.argv) > 2:
        image_path = sys.argv[2]
    else:
        print("Available test images:")
        if os.path.exists("data/tb/normal"):
            for img in os.listdir("data/tb/normal")[:3]:
                print(f"  - data/tb/normal/{img}")
        if os.path.exists("data/tb/disease"):
            for img in os.listdir("data/tb/disease")[:3]:
                print(f"  - data/tb/disease/{img}")
        
        image_path = input("Enter image path: ").strip()
    
    try:
        # Make prediction
        prediction, confidence = predict_with_trained_model(model_path, image_path)
        confidence *= 100
        
        # Determine result
        result = "NORMAL" if prediction == 0 else "DISEASE DETECTED"
        
        # Output results
        print("\n" + "="*60)
        print("    Trained Model Diagnosis Result")
        print("="*60)
        print(f"📷 Image      : {image_path}")
        print(f"🏗️  Model      : {os.path.basename(model_path)}")
        print(f"🔬 Prediction : {result}")
        print(f"📊 Confidence : {confidence:.2f}%")
        
        # Confidence assessment
        if confidence >= 90:
            print("🟢 Status     : HIGH CONFIDENCE ✓")
        elif confidence >= 75:
            print("🟡 Status     : MEDIUM CONFIDENCE ~")
        else:
            print("🔴 Status     : LOW CONFIDENCE ✗")
        
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
