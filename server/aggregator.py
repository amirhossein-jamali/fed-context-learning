import torch
import copy
import collections
import os
import json
import time
import traceback

class FedAvgAggregator:
    """
    FedAvg aggregation strategy for federated learning.
    Implements the Federated Averaging algorithm from the paper:
    "Communication-Efficient Learning of Deep Networks from Decentralized Data"
    (McMahan et al., 2017)
    """
    
    def __init__(self, model, config):
        """
        Initialize the FedAvg aggregator
        
        Args:
            model (nn.Module): The global model to be aggregated
            config (dict): Configuration dictionary
        """
        self.model = model
        self.config = config
        self.aggregation_history = []
        
    def aggregate(self, client_results):
        """
        Aggregate client models using FedAvg
        
        Args:
            client_results (list): List of dicts containing client models and metadata
                Each dict should have:
                - 'client_id': ID of the client
                - 'model_state': State dict of the client model
                - 'train_size': Size of the client's training data
        
        Returns:
            OrderedDict: Aggregated model state dict
        """
        if not client_results:
            raise ValueError("No client results to aggregate")
        
        # Extract client models and weights
        client_models = [result['model_state'] for result in client_results]
        client_sizes = [result['train_size'] for result in client_results]
        client_ids = [result['client_id'] for result in client_results]
        client_domains = [result.get('domain_name', f"client_{cid}") for cid, result in zip(client_ids, client_results)]
        
        total_size = sum(client_sizes)
        
        # Calculate weight for each client based on data size
        client_weights = [size / total_size for size in client_sizes]
        
        # Print aggregation information
        print("\n----- FedAvg Aggregation Information -----")
        print(f"Number of clients participating in aggregation: {len(client_results)}")
        for i, (cid, domain, weight, size) in enumerate(zip(client_ids, client_domains, client_weights, client_sizes)):
            print(f"Client {cid} ({domain}): Weight = {weight:.4f} (Data Size: {size})")
        
        # Create a new state dict for the aggregated model
        global_model_state = collections.OrderedDict()
        
        # Get the state dict of the first client model to initialize
        first_model = client_models[0]
        
        # Initialize the aggregated model with zeros
        for key in first_model.keys():
            global_model_state[key] = torch.zeros_like(first_model[key], dtype=torch.float32)
            
        # Weighted average of model parameters
        for client_idx, client_model in enumerate(client_models):
            weight = client_weights[client_idx]
            for key in client_model.keys():
                # Convert tensor to float for aggregation if needed
                if client_model[key].dtype != torch.float32:
                    weighted_param = client_model[key].float() * weight
                else:
                    weighted_param = client_model[key] * weight
                
                global_model_state[key] += weighted_param
                
        # Convert back to original dtype
        for key in global_model_state.keys():
            if first_model[key].dtype != torch.float32:
                global_model_state[key] = global_model_state[key].to(first_model[key].dtype)
        
        # Store aggregation metadata
        aggregation_info = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_clients": len(client_results),
            "client_ids": client_ids,
            "client_domains": client_domains,
            "client_weights": client_weights,
            "client_data_sizes": client_sizes,
            "total_data_size": total_size
        }
        
        self.aggregation_history.append(aggregation_info)
        
        # Save aggregation metadata if configured
        if self.config['logging']['save_model']:
            try:
                agg_dir = os.path.join(self.config['logging']['save_path'], "aggregation_info")
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
                
            except Exception as e:
                print(f"Warning: Failed to save aggregation metadata: {e}")
                traceback.print_exc()
                # Continue execution even if saving fails
                pass
        
        print(f"Aggregation completed. Global model updated.")
        print("------------------------------------------")
                
        return global_model_state
        
    def update_global_model(self, client_results):
        """
        Update the global model using the aggregated client models
        
        Args:
            client_results (list): List of dicts containing client models and metadata
        
        Returns:
            nn.Module: Updated global model
        """
        aggregated_state = self.aggregate(client_results)
        self.model.load_state_dict(aggregated_state)
        return self.model 