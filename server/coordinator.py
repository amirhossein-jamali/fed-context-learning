import torch
import os
import time
import json
import traceback
from pathlib import Path

class FederatedServer:
    """
    Server coordinator for the federated learning process.
    Manages the communication between server and clients and orchestrates training rounds.
    """
    
    def __init__(self, model, clients, aggregator, test_client, config):
        """
        Initialize the federated server
        
        Args:
            model (nn.Module): The global model
            clients (list): List of ClientTrainer instances
            aggregator: The aggregation strategy (e.g., FedAvgAggregator)
            test_client (ClientTrainer): Client for testing on unseen domain
            config (dict): Configuration dictionary
        """
        self.model = model
        self.clients = clients
        self.aggregator = aggregator
        self.test_client = test_client
        self.config = config
        self.device = torch.device(config['federated_learning']['device'])
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup logging
        self.logs = {
            'train_metrics': [],
            'test_metrics': [],
            'client_metrics': {}  # Store metrics for each client
        }
        
        # Initialize client metrics logs
        for client in self.clients:
            client_id = client.client_id
            domain_name = client.domain_name if hasattr(client, 'domain_name') else f"client_{client_id}"
            self.logs['client_metrics'][domain_name] = []
        
        # Create directories for saving models
        if config['logging']['save_model']:
            os.makedirs(config['logging']['save_path'], exist_ok=True)
            
            # Create directory for global models
            global_model_dir = os.path.join(config['logging']['save_path'], "global_model")
            os.makedirs(global_model_dir, exist_ok=True)
            
            # Create directories for each client's models
            for client in self.clients:
                if hasattr(client, 'domain_name') and client.domain_name:
                    client_dir = os.path.join(config['logging']['save_path'], f"client_{client.client_id}_{client.domain_name}")
                else:
                    client_dir = os.path.join(config['logging']['save_path'], f"client_{client.client_id}")
                os.makedirs(client_dir, exist_ok=True)
        
    def save_global_model(self, round_num=None):
        """
        Save the global model to disk
        
        Args:
            round_num (int, optional): Current round number
        """
        if self.config['logging']['save_model']:
            try:
                # Define filename based on round
                if round_num is not None:
                    filename = f"model_round_{round_num}.pt"
                else:
                    filename = "model_final.pt"
                
                # Save to global model directory
                model_path = os.path.join(self.config['logging']['save_path'], "global_model", filename)
                torch.save(self.model.state_dict(), model_path)
                print(f"Global model saved to {model_path}")
                
                # Save model architecture and hyperparameters
                model_info = {
                    "architecture": str(self.model),
                    "config": self.config,
                    "round": round_num,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                info_path = os.path.join(self.config['logging']['save_path'], "global_model", "model_info.json")
                with open(info_path, 'w') as f:
                    json.dump(model_info, f, indent=4, default=str)
            except Exception as e:
                print(f"Error saving global model: {e}")
                traceback.print_exc()
                # Continue execution even if saving fails
                pass
    
    def train_round(self, round_num):
        """
        Execute a single federated learning round
        
        Args:
            round_num (int): Current round number
            
        Returns:
            dict: Round metrics
        """
        print(f"\n--- Round {round_num+1}/{self.config['federated_learning']['num_rounds']} ---")
        
        try:
            # Collect client updates
            client_results = []
            round_start_time = time.time()
            
            # Train on each client
            for client in self.clients:
                try:
                    # Send global model to client
                    client.update_model(self.model.state_dict())
                    
                    # Train locally on client
                    result = client.train()
                    client_results.append(result)
                    
                    # Log client results
                    domain_name = result.get('domain_name', f"client_{result['client_id']}")
                    if domain_name in self.logs['client_metrics']:
                        self.logs['client_metrics'][domain_name].append({
                            'round': round_num + 1,
                            'accuracy': result['metrics']['accuracy'],
                            'loss': result['metrics']['loss']
                        })
                    
                    print(f"Client {client.client_id} completed training. "
                        f"Accuracy: {result['metrics']['accuracy']:.2f}%")
                    
                    # Save client model for each round
                    if self.config['logging']['save_model']:
                        try:
                            if hasattr(client, 'domain_name') and client.domain_name:
                                client_dir = os.path.join(self.config['logging']['save_path'], f"client_{client.client_id}_{client.domain_name}")
                            else:
                                client_dir = os.path.join(self.config['logging']['save_path'], f"client_{client.client_id}")
                            
                            # Create directory if it doesn't exist
                            os.makedirs(client_dir, exist_ok=True)
                            
                            # Save model state dictionary
                            model_path = os.path.join(client_dir, f"model_round_{round_num+1}.pt")
                            torch.save(result['model_state'], model_path)
                            print(f"Client {client.client_id} model saved to {model_path}")
                        except Exception as e:
                            print(f"Error saving client model: {e}")
                            traceback.print_exc()
                            # Continue execution even if saving fails
                            pass
                except Exception as e:
                    print(f"Error during client {client.client_id} training: {e}")
                    traceback.print_exc()
            
            # Aggregate client models using FedAvg
            if not client_results:
                print("No client results available for aggregation. Skipping round.")
                return None
                
            print(f"Aggregating models from {len(client_results)} clients...")
            try:
                self.model = self.aggregator.update_global_model(client_results)
                
                # Save global model after aggregation
                if self.config['logging']['save_model']:
                    self.save_global_model(round_num+1)
            except Exception as e:
                print(f"Error during model aggregation: {e}")
                traceback.print_exc()
                return None
            
            # Evaluate global model on all clients (for logging purposes)
            train_acc = 0.0
            train_loss = 0.0
            
            try:
                for client in self.clients:
                    try:
                        eval_result = client.evaluate(self.model)
                        train_acc += eval_result['metrics']['accuracy']
                        train_loss += eval_result['metrics']['loss']
                    except Exception as e:
                        print(f"Error evaluating client {client.client_id}: {e}")
                        traceback.print_exc()
                
                avg_train_acc = train_acc / len(self.clients)
                avg_train_loss = train_loss / len(self.clients)
                
                # Evaluate on test domain (unseen domain)
                test_result = self.test_client.evaluate(self.model)
                test_acc = test_result['metrics']['accuracy']
                test_loss = test_result['metrics']['loss']
            except Exception as e:
                print(f"Error during evaluation: {e}")
                traceback.print_exc()
                # Set default values if evaluation fails
                avg_train_acc = 0.0
                avg_train_loss = 0.0
                test_acc = 0.0
                test_loss = 0.0
            
            # Calculate round time
            round_time = time.time() - round_start_time
            
            # Log metrics
            round_metrics = {
                'round': round_num + 1,
                'train_accuracy': avg_train_acc,
                'train_loss': avg_train_loss,
                'test_accuracy': test_acc,
                'test_loss': test_loss,
                'round_time': round_time
            }
            
            # Add to logs
            self.logs['train_metrics'].append({
                'round': round_num + 1,
                'accuracy': avg_train_acc,
                'loss': avg_train_loss
            })
            
            self.logs['test_metrics'].append({
                'round': round_num + 1,
                'accuracy': test_acc,
                'loss': test_loss
            })
            
            # Print round summary
            print(f"Round {round_num+1} completed in {round_time:.2f}s")
            print(f"Train Accuracy: {avg_train_acc:.2f}%, Train Loss: {avg_train_loss:.4f}")
            print(f"Test Accuracy: {test_acc:.2f}%, Test Loss: {test_loss:.4f}")
            
            # Save metrics for this round
            if self.config['logging']['save_model']:
                try:
                    metrics_path = os.path.join(self.config['logging']['save_path'], f"metrics_round_{round_num+1}.json")
                    with open(metrics_path, 'w') as f:
                        json.dump(round_metrics, f, indent=4)
                    print(f"Metrics for round {round_num+1} saved to {metrics_path}")
                except Exception as e:
                    print(f"Error saving metrics: {e}")
                    traceback.print_exc()
                
            return round_metrics
            
        except Exception as e:
            print(f"Error in training round {round_num+1}: {e}")
            traceback.print_exc()
            return None
    
    def train(self):
        """
        Execute the full federated learning training process
        
        Returns:
            dict: Training logs
        """
        try:
            print(f"Starting Federated Learning with {len(self.clients)} clients")
            print(f"Test domain: {self.config['data']['test_domain']}")
            print(f"Number of rounds: {self.config['federated_learning']['num_rounds']}")
            print(f"Local epochs: {self.config['federated_learning']['local_epochs']}")
            
            # Save initial global model
            if self.config['logging']['save_model']:
                try:
                    initial_model_path = os.path.join(self.config['logging']['save_path'], "global_model", "model_initial.pt")
                    torch.save(self.model.state_dict(), initial_model_path)
                    print(f"Initial global model saved to {initial_model_path}")
                except Exception as e:
                    print(f"Error saving initial model: {e}")
                    traceback.print_exc()
            
            start_time = time.time()
            
            # Run training for specified number of rounds
            for round_num in range(self.config['federated_learning']['num_rounds']):
                round_metrics = self.train_round(round_num)
                if round_metrics is None:
                    print(f"Warning: Round {round_num+1} failed. Continuing to next round.")
                
            total_time = time.time() - start_time
            
            print(f"\n--- Training completed in {total_time:.2f}s ---")
            if self.logs['test_metrics']:
                print(f"Final Test Accuracy: {self.logs['test_metrics'][-1]['accuracy']:.2f}%")
            
            # Save final models
            if self.config['logging']['save_model']:
                try:
                    # Save global model
                    final_model_path = os.path.join(self.config['logging']['save_path'], "global_model", "model_final.pt")
                    torch.save(self.model.state_dict(), final_model_path)
                    print(f"Final global model saved to {final_model_path}")
                    
                    # Save logs
                    logs_path = os.path.join(self.config['logging']['save_path'], "training_logs.json")
                    with open(logs_path, 'w') as f:
                        json.dump(self.logs, f, indent=4)
                    print(f"Training logs saved to {logs_path}")
                    
                    # Save a summary file
                    summary = {
                        "total_rounds": self.config['federated_learning']['num_rounds'],
                        "total_clients": len(self.clients),
                        "domains": [client.domain_name for client in self.clients if hasattr(client, 'domain_name')],
                        "test_domain": self.test_client.domain_name if hasattr(self.test_client, 'domain_name') else None,
                        "final_train_accuracy": self.logs['train_metrics'][-1]['accuracy'] if self.logs['train_metrics'] else 0,
                        "final_test_accuracy": self.logs['test_metrics'][-1]['accuracy'] if self.logs['test_metrics'] else 0,
                        "training_time_seconds": total_time,
                        "config": self.config,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    summary_path = os.path.join(self.config['logging']['save_path'], "training_summary.json")
                    with open(summary_path, 'w') as f:
                        json.dump(summary, f, indent=4, default=str)
                    print(f"Training summary saved to {summary_path}")
                except Exception as e:
                    print(f"Error saving final data: {e}")
                    traceback.print_exc()
                    # Continue execution even if saving fails
                    pass
        
        except Exception as e:
            print(f"Error in training process: {e}")
            traceback.print_exc()
            
        return self.logs 