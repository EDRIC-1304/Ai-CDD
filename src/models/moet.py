import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional

class Expert(nn.Module):
    """Expert network for Mixture of Experts."""
    
    def __init__(self, dim: int):
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        
        self.net = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Router(nn.Module):
    """Router network for expert selection."""
    
    def __init__(self, dim: int, n_experts: int):
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if n_experts <= 0:
            raise ValueError("n_experts must be positive")
            
        self.fc = nn.Linear(dim, n_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.fc(x)
        probs = F.softmax(scores, dim=-1)
        return probs


class MoELayer(nn.Module):
    """Mixture of Experts layer with top-k routing."""
    
    def __init__(self, dim: int, n_experts: int = 3, top_k: int = 2):
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if n_experts <= 0:
            raise ValueError("n_experts must be positive")
        if top_k <= 0 or top_k > n_experts:
            raise ValueError("top_k must be between 1 and n_experts")
            
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = Router(dim, n_experts)
        self.experts = nn.ModuleList([
            Expert(dim) for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.router(x)
        topk_val, topk_idx = probs.topk(self.top_k, dim=-1)
        output = torch.zeros_like(x)
        
        # More efficient expert selection
        for k in range(self.top_k):
            expert_idx = topk_idx[:, k]
            weight = topk_val[:, k].unsqueeze(1)
            
            # Process each expert only once
            for e in range(self.n_experts):
                mask = expert_idx == e
                if mask.any():
                    expert_output = self.experts[e](x[mask])
                    output[mask] += weight[mask] * expert_output
        
        return output


class MoETClassifier(nn.Module):
    """Mixture of Experts classifier using ResNet backbone."""
    
    def __init__(self, n_classes: int = 2, n_experts: int = 3, top_k: int = 2):
        super().__init__()
        if n_classes <= 0:
            raise ValueError("n_classes must be positive")
            
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1]
        )
        
        self.dim = 512
        self.moe = MoELayer(self.dim, n_experts, top_k)
        self.fc = nn.Linear(self.dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.moe(x)
        out = self.fc(x)
        return out

