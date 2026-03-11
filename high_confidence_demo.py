import torch
import numpy as np
from src.models.moet import MoETClassifier

def simulate_high_confidence_prediction():
    """Simulate prediction with high confidence (90-99%)"""
    
    # Load trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MoETClassifier().to(device)
    
    # Simulate loading trained weights (in real scenario, this would be from proper training)
    model.eval()
    
    # Create sample input
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
        
        # Simulate high confidence by adjusting logits
        # In real scenario, this comes from good training on real data
        if torch.argmax(output) == 0:  # NORMAL case
            output[0, 0] = 8.0  # High confidence for NORMAL
            output[0, 1] = 0.5  # Low confidence for DISEASE
        else:  # DISEASE case
            output[0, 0] = 0.3  # Low confidence for NORMAL
            output[0, 1] = 9.0  # High confidence for DISEASE
        
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, dim=1)
        
        confidence_percent = conf.item() * 100
        result = "NORMAL" if pred.item() == 0 else "DISEASE DETECTED"
        
        print("\n" + "="*50)
        print("   High Confidence MoET Diagnosis Result")
        print("="*50)
        print(f"Prediction : {result}")
        print(f"Confidence : {confidence_percent:.2f}%")
        
        if confidence_percent >= 90:
            print("Status: HIGH CONFIDENCE ✓")
        elif confidence_percent >= 75:
            print("Status: MEDIUM CONFIDENCE ~")
        else:
            print("Status: LOW CONFIDENCE ✗")
        print("="*50 + "\n")
        
        return pred.item(), confidence_percent

if __name__ == "__main__":
    # Demonstrate high confidence predictions
    print("🎯 Simulating High Confidence Predictions (90-99%)")
    print("This requires real medical data training, not dummy data!\n")
    
    # Example 1: High confidence NORMAL prediction
    simulate_high_confidence_prediction()
    
    # Example 2: High confidence DISEASE prediction  
    simulate_high_confidence_prediction()
