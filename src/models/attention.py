import torch
import torch.nn.functional as F

class AttentionModule(torch.nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.scale = embedding_dim ** -0.5

    def forward(self, query_embeddings, support_embeddings):
        """
        query_embeddings: (B, D)
        support_embeddings: (N, D)
        """
        scores = torch.matmul(query_embeddings, support_embeddings.T) * self.scale
        weights = F.softmax(scores, dim=-1)
        return weights
