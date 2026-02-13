import os
import cv2
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional

class LungDataset(Dataset):
    """Dataset for lung X-ray images with normal/disease classification."""
    
    def __init__(self, root: str):
        if not os.path.exists(root):
            raise ValueError(f"Root directory {root} does not exist")
            
        self.data: List[str] = []
        self.labels: List[int] = []
        
        for label, folder in enumerate(["normal", "disease"]):
            path = os.path.join(root, folder)
            if not os.path.exists(path):
                print(f"Warning: {path} does not exist, skipping")
                continue
                
            for file in os.listdir(path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    self.data.append(os.path.join(path, file))
                    self.labels.append(label)
        
        if len(self.data) == 0:
            raise ValueError(f"No valid image files found in {root}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.data[idx]
        
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
            
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        
        img = torch.tensor(img).permute(2, 0, 1).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return img, label
