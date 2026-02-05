import torch
import torch.nn as nn

from .encoder import CNNEncoder
from .attention import AttentionModule


def compute_prototype(support_embeddings, attention_weights):
    """
    support_embeddings: (N, D)
    attention_weights: (B, N)
    """
    return torch.matmul(attention_weights, support_embeddings)


class PAN(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        self.encoder = CNNEncoder(embedding_dim)
        self.attention = AttentionModule(embedding_dim)

    def forward(self, support_images, support_labels, query_images):
        """
        support_images: (N, 1, H, W)
        support_labels: (N,)
        query_images: (B, 1, H, W)
        """

        support_embeddings = self.encoder(support_images)
        query_embeddings = self.encoder(query_images)

        classes = torch.unique(support_labels)
        prototypes = []

        for cls in classes:
            cls_support = support_embeddings[support_labels == cls]
            attn_weights = self.attention(query_embeddings, cls_support)
            proto = compute_prototype(cls_support, attn_weights)
            prototypes.append(proto)

        prototypes = torch.stack(prototypes, dim=1)  # (B, C, D)

        distances = torch.cdist(
            query_embeddings.unsqueeze(1),
            prototypes
        )

        logits = -distances.squeeze(1)
        return logits
