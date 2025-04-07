import torch
import os
import argparse
import yaml
from pathlib import Path

# Import project modules
from data.pacs_loader import get_dataloader, load_config
from data.download_pacs import download_and_extract_pacs
from model.cnn import get_model
from client.trainer import ClientTrainer
from server.aggregator import FedAvgAggregator
from server.coordinator import FederatedServer
from utils.metrics import plot_metrics, print_summary

def parse_args():
    parser = argparse.ArgumentParser(description='Federated Learning on PACS Dataset')
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
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def main():
    # Parse arguments and load config
    args = parse_args()
    config = load_config(args.config)
    
    # Override config with command line arguments if provided
    if args.test_domain:
        config['data']['test_domain'] = args.test_domain
    if args.device:
        config['federated_learning']['device'] = args.device
    
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
    
    # Initialize model
    global_model = get_model(config)
    
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
    
    # Create clients
    clients = []
    for i, domain in enumerate(train_domains):
        # Get dataloader for this domain
        dataloader = get_dataloader(config, domain, is_train=True)
        
        # Create client
        client = ClientTrainer(
            client_id=i,
            model=global_model,
            data_loader=dataloader,
            config=config,
            domain_name=domain  # Pass domain name to client
        )
        
        clients.append(client)
        print(f"Created client {i} for domain {domain}")
    
    # Create test client with test domain
    test_dataloader = get_dataloader(config, test_domain, is_train=False)
    test_client = ClientTrainer(
        client_id=len(clients),
        model=global_model,
        data_loader=test_dataloader,
        config=config,
        domain_name=test_domain  # Pass domain name to test client
    )
    print(f"Created test client for domain {test_domain}")
    
    # Create aggregator
    aggregator = FedAvgAggregator(global_model, config)
    
    # Create server
    server = FederatedServer(
        model=global_model,
        clients=clients,
        aggregator=aggregator,
        test_client=test_client,
        config=config
    )
    
    # Run federated learning
    logs = server.train()
    
    # Print summary and plot results
    print_summary(logs)
    
    if config['logging']['save_model']:
        plot_path = os.path.join(config['logging']['save_path'], 'metrics_plot.png')
        plot_metrics(logs, save_path=plot_path)
    else:
        plot_metrics(logs)
    
    print("Federated learning completed successfully!")

if __name__ == "__main__":
    main() 