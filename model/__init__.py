"""
Model module for federated learning with CLIP alignment.
Contains CNN model, mapper and CLIP alignment components.
"""

from .cnn import get_model, SimpleCNN
from .mapper import EmbeddingMapper
from .clip_alignment import CLIPAligner, CombinedLoss

__all__ = [
    'get_model', 'SimpleCNN', 
    'EmbeddingMapper',
    'CLIPAligner', 'CombinedLoss'
] 