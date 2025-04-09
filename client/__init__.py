"""
Client module for federated learning.
Contains client implementations for training.
"""

from .trainer import ClientTrainer
from .clip_alignment_client import CLIPAlignmentClient

__all__ = ['ClientTrainer', 'CLIPAlignmentClient'] 