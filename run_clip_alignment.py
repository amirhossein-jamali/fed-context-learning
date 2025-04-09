import torch
import os
import argparse
import yaml
import random
import numpy as np
from pathlib import Path

# Import project modules
from data.pacs_loader import get_dataloader, load_config
from data.download_pacs import download_and_extract_pacs
from model.cnn import get_model
from model.mapper import EmbeddingMapper
from client.clip_alignment_client import CLIPAlignmentClient
from server.aggregator import FedAvgAggregator
from server.mapper_aggregator import MapperAggregator
from server.clip_coordinator import CLIPFederatedServer
from utils.metrics import print_summary

def set_seed(seed):
    """Set the random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def parse_args():
    parser = argparse.ArgumentParser(description='Federated Learning with CLIP Alignment on PACS Dataset')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--test-domain', type=str, default=None,
                        help='Domain to use for testing (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (overrides config)')
    parser.add_argument('--download-data', action='store_true',
                        help='Download PACS dataset if not exists')
    parser.add_argument('--save-client-models', action='store_true',
                        help='Save individual client models for each domain')
    parser.add_argument('--lambda-align', type=float, default=0.1,
                        help='Weight for alignment loss (default: 0.1)')
    parser.add_argument('--use-dp-noise', action='store_true',
                        help='Apply differential privacy noise to small embeddings')
    parser.add_argument('--dp-noise-std', type=float, default=0.01,
                        help='Standard deviation for DP noise (default: 0.01)')
    parser.add_argument('--small-dim', type=int, default=64,
                        help='Dimension of small embedding (default: 64)')
    parser.add_argument('--mapper-hidden-dim', type=int, default=256,
                        help='Hidden dimension of mapper network (default: 256)')
    return parser.parse_args()

def main():
    # Parse arguments and load config
    args = parse_args()
    config = load_config(args.config)
    
    # Override config with command line arguments if provided
    if args.test_domain:
        config['data']['test_domain'] = args.test_domain
    if args.device:
        config['federated_learning']['device'] = args.device
    
    # Add CLIP alignment specific config
    config['lambda_align'] = args.lambda_align
    config['use_dp_noise'] = args.use_dp_noise
    config['dp_noise_std'] = args.dp_noise_std
    config['model']['small_dim'] = args.small_dim
    config['model']['mapper_hidden_dim'] = args.mapper_hidden_dim
    config['model']['clip_dim'] = 512  # CLIP ViT-B/32 has 512-dim embeddings
    
    # Force saving of client models if specified
    if args.save_client_models:
        config['logging']['save_model'] = True
    
    # Download dataset if requested
    if args.download_data:
        print("Checking for PACS dataset...")
        download_and_extract_pacs(config['data']['data_path'])
    
    # Set random seed
    set_seed(args.seed)
    
    # Check if CUDA is available
    if config['federated_learning']['device'] == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Using CPU instead.")
        config['federated_learning']['device'] = 'cpu'
    
    print(f"Using device: {config['federated_learning']['device']}")
    
    # Create checkpoints directory if it doesn't exist
    if config['logging']['save_model']:
        os.makedirs(config['logging']['save_path'], exist_ok=True)
    
    # Initialize model and mapper
    global_model = get_model(config)
    global_mapper = EmbeddingMapper(
        small_dim=config['model']['small_dim'],
        hidden_dim=config['model']['mapper_hidden_dim'],
        clip_dim=config['model']['clip_dim']
    )
    
    # Get the list of domains
    domains = config['data']['domains']
    test_domain = config['data']['test_domain']
    
    # Verify test domain is in the list of domains
    if test_domain not in domains:
        raise ValueError(f"Test domain {test_domain} not in domains list {domains}")
    
    # Get training domains (all except test domain)
    train_domains = [d for d in domains if d != test_domain]
    
    # Check if we have the right number of clients
    if len(train_domains) != config['federated_learning']['num_clients']:
        print(f"Warning: Number of clients in config ({config['federated_learning']['num_clients']}) "
              f"does not match the number of training domains ({len(train_domains)})")
        config['federated_learning']['num_clients'] = len(train_domains)
    
    print(f"Training domains: {train_domains}")
    print(f"Test domain: {test_domain}")
    
    # Create clients with CLIP alignment
    clients = []
    for i, domain in enumerate(train_domains):
        # Get dataloader for this domain
        dataloader = get_dataloader(config, domain, is_train=True)
        
        # Create client with CLIP alignment
        client = CLIPAlignmentClient(
            client_id=i,
            model=global_model,
            data_loader=dataloader,
            config=config,
            mapper=global_mapper,
            domain_name=domain  # Pass domain name to client
        )
        
        clients.append(client)
        print(f"Created CLIP alignment client {i} for domain {domain}")
    
    # Create test client with test domain
    test_dataloader = get_dataloader(config, test_domain, is_train=False)
    test_client = CLIPAlignmentClient(
        client_id=len(clients),
        model=global_model,
        data_loader=test_dataloader,
        config=config,
        mapper=global_mapper,
        domain_name=test_domain  # Pass domain name to test client
    )
    print(f"Created CLIP alignment test client for domain {test_domain}")
    
    # Create aggregators
    model_aggregator = FedAvgAggregator(global_model, config)
    mapper_aggregator = MapperAggregator(global_mapper, config)
    
    # Create server with CLIP alignment
    server = CLIPFederatedServer(
        model=global_model,
        mapper=global_mapper,
        clients=clients,
        model_aggregator=model_aggregator,
        mapper_aggregator=mapper_aggregator,
        test_client=test_client,
        config=config
    )
    
    # Run federated learning with CLIP alignment
    logs = server.train()
    
    # Print summary
    print_summary(logs)
    
    print("Federated learning with CLIP alignment completed successfully!")

if __name__ == "__main__":
    main() 