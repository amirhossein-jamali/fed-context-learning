import torch
import torch.nn as nn

class EmbeddingMapper(nn.Module):
    """Maps small embeddings to CLIP embedding space."""
    def __init__(self, small_dim=64, hidden_dim=256, clip_dim=512):
        super(EmbeddingMapper, self).__init__()
        self.mapper = nn.Sequential(
            nn.Linear(small_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, clip_dim),
            nn.LayerNorm(clip_dim)  # Normalize to match CLIP's embedding space
        )
    
    def forward(self, z_small):
        return self.mapper(z_small) 