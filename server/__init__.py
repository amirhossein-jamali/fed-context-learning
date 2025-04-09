"""
Server module for federated learning.
Contains server-side aggregation and coordination components.
"""

from .aggregator import FedAvgAggregator
from .coordinator import FederatedServer
from .mapper_aggregator import MapperAggregator
from .clip_coordinator import CLIPFederatedServer

__all__ = [
    'FedAvgAggregator', 'FederatedServer',
    'MapperAggregator', 'CLIPFederatedServer'
] 