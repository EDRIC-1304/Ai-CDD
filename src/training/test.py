import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from typing import Tuple

from preprocessing.dataset import LungDataset
from models.moet import MoETClassifier
def evaluate_model(model_path: str, data_path: str, batch_size: int = 16) -> Tuple[float, int]:
    """Evaluate trained MoET classifier on test dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load model
    model = MoETClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load dataset
    dataset = LungDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Evaluating on {len(dataset)} samples")
    
    # Evaluation
    correct = 0
    total = 0
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            outputs = model(imgs)
            preds = outputs.argmax(1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total * 100
    return accuracy, total

if __name__ == "__main__":
    # Evaluate TB model
    try:
        tb_acc, tb_samples = evaluate_model("models_saved/tb_moet.pth", "data/tb")
        print(f"TB Model Accuracy: {tb_acc:.2f}% on {tb_samples} samples")
    except FileNotFoundError as e:
        print(f"Error evaluating TB model: {e}")
    
    # Evaluate cancer model
    try:
        cancer_acc, cancer_samples = evaluate_model("models_saved/cancer_moet.pth", "data/cancer")
        print(f"Cancer Model Accuracy: {cancer_acc:.2f}% on {cancer_samples} samples")
    except FileNotFoundError as e:
        print(f"Error evaluating cancer model: {e}")
