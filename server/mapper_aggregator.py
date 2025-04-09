import torch
import copy
import collections
import os
import json
import time
import traceback
import matplotlib.pyplot as plt
import numpy as np

class MapperAggregator:
    """
    Specialized aggregator for the mapper network (φ) in federated learning.
    Only aggregates the mapper weights, keeping the CNN model weights separate.
    """
    
    def __init__(self, mapper, config):
        """
        Initialize the MapperAggregator
        
        Args:
            mapper (nn.Module): The global mapper network to be aggregated
            config (dict): Configuration dictionary
        """
        self.mapper = mapper
        self.config = config
        self.aggregation_history = []
        self.alignment_metrics = {
            'rounds': [],
            'avg_alignment_loss': []
        }
        
    def aggregate(self, client_results):
        """
        Aggregate client mapper models using weighted averaging
        
        Args:
            client_results (list): List of dicts containing client models and metadata
                Each dict should have:
                - 'client_id': ID of the client
                - 'mapper_state': State dict of the client mapper
                - 'train_size': Size of the client's training data
                - 'metrics': Dict with training metrics including 'align_loss'
        
        Returns:
            OrderedDict: Aggregated mapper state dict
        """
        if not client_results:
            raise ValueError("No client results to aggregate")
        
        # Extract client mapper models and weights
        client_mappers = [result['mapper_state'] for result in client_results]
        client_sizes = [result['train_size'] for result in client_results]
        client_ids = [result['client_id'] for result in client_results]
        client_domains = [result.get('domain_name', f"client_{cid}") for cid, result in zip(client_ids, client_results)]
        
        # Extract alignment losses for tracking
        client_align_losses = [result['metrics'].get('align_loss', 0) for result in client_results]
        avg_align_loss = sum(client_align_losses) / len(client_align_losses) if client_align_losses else 0
        
        # Update alignment metrics
        round_num = len(self.alignment_metrics['rounds']) + 1
        self.alignment_metrics['rounds'].append(round_num)
        self.alignment_metrics['avg_alignment_loss'].append(avg_align_loss)
        
        total_size = sum(client_sizes)
        
        # Calculate weight for each client based on data size
        client_weights = [size / total_size for size in client_sizes]
        
        # Print aggregation information
        print("\n----- Mapper Aggregation Information -----")
        print(f"Number of clients participating in aggregation: {len(client_results)}")
        for i, (cid, domain, weight, size, align_loss) in enumerate(zip(client_ids, client_domains, client_weights, client_sizes, client_align_losses)):
            print(f"Client {cid} ({domain}): Weight = {weight:.4f} (Data Size: {size}, Align Loss: {align_loss:.4f})")
        
        # Create a new state dict for the aggregated mapper
        global_mapper_state = collections.OrderedDict()
        
        # Get the state dict of the first client mapper to initialize
        first_mapper = client_mappers[0]
        
        # Initialize the aggregated mapper with zeros
        for key in first_mapper.keys():
            global_mapper_state[key] = torch.zeros_like(first_mapper[key], dtype=torch.float32)
            
        # Weighted average of mapper parameters
        for client_idx, client_mapper in enumerate(client_mappers):
            weight = client_weights[client_idx]
            for key in client_mapper.keys():
                # Convert tensor to float for aggregation if needed
                if client_mapper[key].dtype != torch.float32:
                    weighted_param = client_mapper[key].float() * weight
                else:
                    weighted_param = client_mapper[key] * weight
                
                global_mapper_state[key] += weighted_param
                
        # Convert back to original dtype
        for key in global_mapper_state.keys():
            if first_mapper[key].dtype != torch.float32:
                global_mapper_state[key] = global_mapper_state[key].to(first_mapper[key].dtype)
        
        # Store aggregation metadata
        aggregation_info = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_clients": len(client_results),
            "client_ids": client_ids,
            "client_domains": client_domains,
            "client_weights": client_weights,
            "client_data_sizes": client_sizes,
            "total_data_size": total_size,
            "avg_alignment_loss": avg_align_loss
        }
        
        self.aggregation_history.append(aggregation_info)
        
        # Save aggregation metadata if configured
        if self.config['logging']['save_model']:
            try:
                agg_dir = os.path.join(self.config['logging']['save_path'], "mapper_aggregation_info")
                os.makedirs(agg_dir, exist_ok=True)
                
                # Save current aggregation info
                round_num = len(self.aggregation_history)
                agg_path = os.path.join(agg_dir, f"aggregation_round_{round_num}.json")
                with open(agg_path, 'w') as f:
                    json.dump(aggregation_info, f, indent=4, default=str)
                
                # Save full history
                history_path = os.path.join(agg_dir, "aggregation_history.json")
                with open(history_path, 'w') as f:
                    json.dump(self.aggregation_history, f, indent=4, default=str)
                
                # Plot alignment metrics
                self.plot_alignment_metrics(save_dir=agg_dir)
                
            except Exception as e:
                print(f"Warning: Failed to save aggregation metadata: {e}")
                traceback.print_exc()
                # Continue execution even if saving fails
                pass
        
        print(f"Mapper aggregation completed. Global mapper updated.")
        print(f"Average alignment loss: {avg_align_loss:.4f}")
        print("------------------------------------------")
                
        return global_mapper_state
        
    def update_global_mapper(self, client_results):
        """
        Update the global mapper using the aggregated client mappers
        
        Args:
            client_results (list): List of dicts containing client models and metadata
        
        Returns:
            nn.Module: Updated global mapper
        """
        aggregated_state = self.aggregate(client_results)
        self.mapper.load_state_dict(aggregated_state)
        return self.mapper
    
    def plot_alignment_metrics(self, save_dir=None):
        """
        Plot alignment metrics over rounds
        
        Args:
            save_dir (str, optional): Directory to save the plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.alignment_metrics['rounds'], self.alignment_metrics['avg_alignment_loss'], 'o-', label='Avg Alignment Loss')
        plt.title('CLIP Alignment Loss Over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Alignment Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        if save_dir:
            try:
                plt.savefig(os.path.join(save_dir, 'alignment_loss.png'), dpi=300, bbox_inches='tight')
            except Exception as e:
                print(f"Warning: Failed to save alignment plot: {e}")
                traceback.print_exc()
        
        plt.close() 