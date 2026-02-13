import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


# ======================================================
# Expert Network
# ======================================================

class Expert(nn.Module):
    """Expert network for Mixture of Experts."""
    
    def __init__(self, dim: int):
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
            
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ======================================================
# Router Network
# ======================================================

class Router(nn.Module):
    """Router for expert selection"""
    
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


# ======================================================
# MoE Layer
# ======================================================

class MoELayer(nn.Module):
    """Top-k Mixture of Experts"""
    
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
        
        for k in range(self.top_k):
            expert_ids = topk_idx[:, k]
            weights = topk_val[:, k].unsqueeze(1)
            
            for e in range(self.n_experts):
                mask = expert_ids == e
                if mask.any():
                    expert_out = self.experts[e](x[mask])
                    output[mask] += weights[mask] * expert_out
        
        return output


# ======================================================
# Patch Embedding
# ======================================================

class PatchEmbedding(nn.Module):
    """Convert image into patches"""
    
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 embed_dim: int = 512):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


# ======================================================
# Transformer Encoder Block
# ======================================================

class TransformerBlock(nn.Module):
    """Transformer encoder block with multi-head attention"""
    
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if heads <= 0:
            raise ValueError("heads must be positive")
            
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.attn = nn.MultiheadAttention(
            dim,
            heads,
            dropout=dropout,
            batch_first=True
        )
        
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = self.norm1(x)
        
        mlp_out = self.mlp(x)
        x = x + mlp_out
        x = self.norm2(x)
        
        return x


# ======================================================
# Vision Transformer Backbone
# ======================================================

class VisionTransformer(nn.Module):
    """Vision Transformer backbone"""
    
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 embed_dim: int = 512,
                 depth: int = 6,
                 heads: int = 8):
        super().__init__()
        if img_size <= 0 or patch_size <= 0:
            raise ValueError("img_size and patch_size must be positive")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if depth <= 0 or heads <= 0:
            raise ValueError("depth and heads must be positive")
            
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        self._init_weights()

    def _init_weights(self):
        """Initialize positional embeddings and class token"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        cls_output = x[:, 0]  # Take CLS token
        
        return cls_output


# ======================================================
# MoET Classifier (ViT + MoE)
# ======================================================

class MoETClassifier(nn.Module):
    """
    Vision Transformer + Mixture of Experts Classifier
    """
    
    def __init__(self,
                 n_classes: int = 2,
                 n_experts: int = 3,
                 top_k: int = 2):
        super().__init__()
        if n_classes <= 0:
            raise ValueError("n_classes must be positive")
        if n_experts <= 0:
            raise ValueError("n_experts must be positive")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
            
        self.dim = 512
        
        # Transformer Backbone
        self.backbone = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=self.dim,
            depth=6,
            heads=8
        )
        
        # MoE Layer
        self.moe = MoELayer(self.dim, n_experts, top_k)
        
        # Classifier Head
        self.fc = nn.Linear(self.dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transformer Feature Extraction
        x = self.backbone(x)
        
        # Mixture of Experts
        x = self.moe(x)
        
        # Classification
        out = self.fc(x)
        
        return out


# ======================================================
# Testing
# ======================================================

if __name__ == "__main__":

    model = MoETClassifier(
        n_classes=2,
        n_experts=3,
        top_k=2
    )

    dummy = torch.randn(4, 3, 224, 224)

    out = model(dummy)

    print("Output shape:", out.shape)
