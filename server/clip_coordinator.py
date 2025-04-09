import torch
import os
import time
import json
import traceback
import matplotlib.pyplot as plt
from pathlib import Path
from model.mapper import EmbeddingMapper

class CLIPFederatedServer:
    """
    Server coordinator for the federated learning process with CLIP alignment.
    Manages the communication between server and clients and orchestrates training rounds.
    """
    
    def __init__(self, model, mapper, clients, model_aggregator, mapper_aggregator, test_client, config):
        """
        Initialize the federated server with CLIP alignment
        
        Args:
            model (nn.Module): The global CNN model
            mapper (EmbeddingMapper): The global mapper (φ) network
            clients (list): List of CLIPAlignmentClient instances
            model_aggregator: The aggregation strategy for CNN model (e.g., FedAvgAggregator)
            mapper_aggregator: The aggregation strategy for mapper (e.g., MapperAggregator)
            test_client: Client for testing on unseen domain
            config (dict): Configuration dictionary
        """
        self.model = model
        self.mapper = mapper
        self.clients = clients
        self.model_aggregator = model_aggregator
        self.mapper_aggregator = mapper_aggregator
        self.test_client = test_client
        self.config = config
        self.device = torch.device(config['federated_learning']['device'])
        
        # Move models to device
        self.model.to(self.device)
        self.mapper.to(self.device)
        
        # Setup logging
        self.logs = {
            'train_metrics': [],
            'test_metrics': [],
            'client_metrics': {},  # Store metrics for each client
            'alignment_metrics': {
                'rounds': [],
                'avg_align_loss': []
            }
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
            
            # Create directory for global mappers
            global_mapper_dir = os.path.join(config['logging']['save_path'], "global_mapper")
            os.makedirs(global_mapper_dir, exist_ok=True)
            
            # Create directories for each client's models
            for client in self.clients:
                if hasattr(client, 'domain_name') and client.domain_name:
                    client_dir = os.path.join(config['logging']['save_path'], f"client_{client.client_id}_{client.domain_name}")
                else:
                    client_dir = os.path.join(config['logging']['save_path'], f"client_{client.client_id}")
                os.makedirs(client_dir, exist_ok=True)
    
    def save_global_model(self, round_num=None):
        """
        Save the global model and mapper to disk
        
        Args:
            round_num (int, optional): Current round number
        """
        try:
            # Create global model directory
            global_model_dir = os.path.join(self.config['logging']['save_path'], "global_model")
            os.makedirs(global_model_dir, exist_ok=True)
            
            # Create global mapper directory
            global_mapper_dir = os.path.join(self.config['logging']['save_path'], "global_mapper")
            os.makedirs(global_mapper_dir, exist_ok=True)
            
            # Define filenames based on round
            if round_num is not None:
                model_filename = f"model_round_{round_num}.pt"
                mapper_filename = f"mapper_round_{round_num}.pt"
            else:
                model_filename = "model_final.pt"
                mapper_filename = "mapper_final.pt"
            
            # Save model and mapper
            model_path = os.path.join(global_model_dir, model_filename)
            mapper_path = os.path.join(global_mapper_dir, mapper_filename)
            
            torch.save(self.model.state_dict(), model_path)
            torch.save(self.mapper.state_dict(), mapper_path)
            
            print(f"Global model saved to {model_path}")
            print(f"Global mapper saved to {mapper_path}")
        except Exception as e:
            print(f"Error saving global model or mapper: {e}")
            traceback.print_exc()
            # Continue execution even if saving fails
            pass
    
    def plot_metrics(self, save_path=None):
        """
        Plot training and testing metrics
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        try:
            # Create a figure with subplots
            fig, axs = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot accuracy
            if self.logs['train_metrics'] and 'accuracy' in self.logs['train_metrics'][0]:
                rounds = [m['round'] for m in self.logs['train_metrics']]
                train_acc = [m['accuracy'] for m in self.logs['train_metrics']]
                axs[0, 0].plot(rounds, train_acc, 'o-', label='Train Accuracy')
                
                if self.logs['test_metrics']:
                    test_rounds = [m['round'] for m in self.logs['test_metrics']]
                    test_acc = [m['accuracy'] for m in self.logs['test_metrics']]
                    axs[0, 0].plot(test_rounds, test_acc, 's-', label='Test Accuracy')
                
                axs[0, 0].set_title('Accuracy Over Rounds')
                axs[0, 0].set_xlabel('Round')
                axs[0, 0].set_ylabel('Accuracy (%)')
                axs[0, 0].grid(True, linestyle='--', alpha=0.7)
                axs[0, 0].legend()
            
            # Plot loss
            if self.logs['train_metrics'] and 'loss' in self.logs['train_metrics'][0]:
                rounds = [m['round'] for m in self.logs['train_metrics']]
                train_loss = [m['loss'] for m in self.logs['train_metrics']]
                axs[0, 1].plot(rounds, train_loss, 'o-', label='Train Loss')
                
                if self.logs['test_metrics']:
                    test_rounds = [m['round'] for m in self.logs['test_metrics']]
                    test_loss = [m['loss'] for m in self.logs['test_metrics']]
                    axs[0, 1].plot(test_rounds, test_loss, 's-', label='Test Loss')
                
                axs[0, 1].set_title('Loss Over Rounds')
                axs[0, 1].set_xlabel('Round')
                axs[0, 1].set_ylabel('Loss')
                axs[0, 1].grid(True, linestyle='--', alpha=0.7)
                axs[0, 1].legend()
            
            # Plot alignment loss
            if self.logs['alignment_metrics']['rounds']:
                rounds = self.logs['alignment_metrics']['rounds']
                align_loss = self.logs['alignment_metrics']['avg_align_loss']
                axs[1, 0].plot(rounds, align_loss, 'o-', label='Alignment Loss')
                axs[1, 0].set_title('CLIP Alignment Loss Over Rounds')
                axs[1, 0].set_xlabel('Round')
                axs[1, 0].set_ylabel('Alignment Loss')
                axs[1, 0].grid(True, linestyle='--', alpha=0.7)
                axs[1, 0].legend()
            
            # Plot client metrics
            axs[1, 1].set_title('Client Accuracy Over Rounds')
            axs[1, 1].set_xlabel('Round')
            axs[1, 1].set_ylabel('Accuracy (%)')
            axs[1, 1].grid(True, linestyle='--', alpha=0.7)
            
            # Add each client's accuracy
            for domain, metrics in self.logs['client_metrics'].items():
                if metrics:
                    client_rounds = [m['round'] for m in metrics]
                    client_acc = [m['accuracy'] for m in metrics]
                    axs[1, 1].plot(client_rounds, client_acc, 'o-', label=f'{domain}')
            
            axs[1, 1].legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Metrics plot saved to {save_path}")
            
            plt.close()
        except Exception as e:
            print(f"Error plotting metrics: {e}")
            traceback.print_exc()
    
    def train_round(self, round_num):
        """
        Execute a single federated learning round with CLIP alignment
        
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
                    # Send global model and mapper to client
                    client.update_model(self.model.state_dict(), self.mapper.state_dict())
                    
                    # Train locally on client
                    result = client.train()
                    client_results.append(result)
                    
                    # Log client results
                    domain_name = result.get('domain_name', f"client_{result['client_id']}")
                    if domain_name in self.logs['client_metrics']:
                        self.logs['client_metrics'][domain_name].append({
                            'round': round_num + 1,
                            'accuracy': result['metrics']['accuracy'],
                            'loss': result['metrics']['loss'],
                            'ce_loss': result['metrics']['ce_loss'],
                            'align_loss': result['metrics']['align_loss']
                        })
                    
                    print(f"Client {client.client_id} completed training. "
                        f"Accuracy: {result['metrics']['accuracy']:.2f}%, "
                        f"CE Loss: {result['metrics']['ce_loss']:.4f}, "
                        f"Align Loss: {result['metrics']['align_loss']:.4f}")
                    
                    # Save client model and mapper for each round
                    if self.config['logging']['save_model']:
                        client.save_model(self.config['logging']['save_path'], round_num+1)
                except Exception as e:
                    print(f"Error during client {client.client_id} training: {e}")
                    traceback.print_exc()
            
            # Aggregate client models and mappers 
            if not client_results:
                print("No client results available for aggregation. Skipping round.")
                return None
                
            # Collect alignment losses
            align_losses = [result['metrics']['align_loss'] for result in client_results if 'align_loss' in result['metrics']]
            avg_align_loss = sum(align_losses) / len(align_losses) if align_losses else 0
            
            # Update alignment metrics
            self.logs['alignment_metrics']['rounds'].append(round_num + 1)
            self.logs['alignment_metrics']['avg_align_loss'].append(avg_align_loss)
            
            # First aggregate the CNN model
            print(f"Aggregating CNN models from {len(client_results)} clients...")
            try:
                self.model = self.model_aggregator.update_global_model(client_results)
            except Exception as e:
                print(f"Error during CNN model aggregation: {e}")
                traceback.print_exc()
            
            # Then aggregate the mapper networks
            print(f"Aggregating mapper networks from {len(client_results)} clients...")
            try:
                self.mapper = self.mapper_aggregator.update_global_mapper(client_results)
            except Exception as e:
                print(f"Error during mapper aggregation: {e}")
                traceback.print_exc()
                
            # Save global model and mapper after aggregation
            if self.config['logging']['save_model']:
                self.save_global_model(round_num+1)
            
            # Evaluate global model and mapper on all clients (for logging purposes)
            train_acc = 0.0
            train_loss = 0.0
            train_ce_loss = 0.0
            train_align_loss = 0.0
            
            try:
                for client in self.clients:
                    try:
                        eval_result = client.evaluate(self.model, self.mapper)
                        train_acc += eval_result['metrics']['accuracy']
                        train_loss += eval_result['metrics']['loss']
                        train_ce_loss += eval_result['metrics']['ce_loss']
                        train_align_loss += eval_result['metrics']['align_loss']
                    except Exception as e:
                        print(f"Error evaluating client {client.client_id}: {e}")
                        traceback.print_exc()
                
                num_clients = len(self.clients)
                avg_train_acc = train_acc / num_clients
                avg_train_loss = train_loss / num_clients
                avg_train_ce_loss = train_ce_loss / num_clients
                avg_train_align_loss = train_align_loss / num_clients
                
                # Record metrics
                train_metrics = {
                    'round': round_num + 1,
                    'accuracy': avg_train_acc,
                    'loss': avg_train_loss,
                    'ce_loss': avg_train_ce_loss,
                    'align_loss': avg_train_align_loss
                }
                self.logs['train_metrics'].append(train_metrics)
                
                # Evaluate on test domain (unseen domain)
                test_result = self.test_client.evaluate(self.model, self.mapper)
                test_acc = test_result['metrics']['accuracy']
                test_loss = test_result['metrics']['loss']
                test_ce_loss = test_result['metrics'].get('ce_loss', 0)
                test_align_loss = test_result['metrics'].get('align_loss', 0)
                
                # Record test metrics
                test_metrics = {
                    'round': round_num + 1,
                    'accuracy': test_acc,
                    'loss': test_loss,
                    'ce_loss': test_ce_loss,
                    'align_loss': test_align_loss
                }
                self.logs['test_metrics'].append(test_metrics)
                
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
            
            # Print round summary
            print(f"\nRound {round_num+1} Summary:")
            print(f"  Train Accuracy: {avg_train_acc:.2f}%")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Train CE Loss: {avg_train_ce_loss:.4f}")
            print(f"  Train Alignment Loss: {avg_train_align_loss:.4f}")
            print(f"  Test Accuracy: {test_acc:.2f}%")
            print(f"  Test Loss: {test_loss:.4f}")
            print(f"  Round Time: {round_time:.2f}s")
            
            return {
                'round': round_num + 1,
                'train_accuracy': avg_train_acc,
                'train_loss': avg_train_loss,
                'train_ce_loss': avg_train_ce_loss,
                'train_align_loss': avg_train_align_loss,
                'test_accuracy': test_acc,
                'test_loss': test_loss,
                'round_time': round_time
            }
        except Exception as e:
            print(f"Error in round {round_num+1}: {e}")
            traceback.print_exc()
            return None
    
    def train(self):
        """
        Execute the full federated learning training process with CLIP alignment
        
        Returns:
            dict: Training logs
        """
        try:
            print(f"Starting Federated Learning with CLIP Alignment using {len(self.clients)} clients")
            print(f"Test domain: {self.config['data']['test_domain']}")
            print(f"Number of rounds: {self.config['federated_learning']['num_rounds']}")
            print(f"Local epochs: {self.config['federated_learning']['local_epochs']}")
            print(f"Using differential privacy noise: {self.config.get('use_dp_noise', False)}")
            print(f"Alignment weight (λ): {self.config.get('lambda_align', 0.1)}")
            
            # Save initial global model and mapper
            if self.config['logging']['save_model']:
                try:
                    self.save_global_model(round_num=0)  # Round 0 = initial model
                except Exception as e:
                    print(f"Error saving initial model and mapper: {e}")
                    traceback.print_exc()
            
            start_time = time.time()
            
            # Run training for specified number of rounds
            for round_num in range(self.config['federated_learning']['num_rounds']):
                round_metrics = self.train_round(round_num)
                if round_metrics is None:
                    print(f"Warning: Round {round_num+1} failed. Continuing to next round.")
                
                # Plot metrics after each round if configured
                if self.config['logging']['save_model'] and self.config.get('plot_each_round', False):
                    try:
                        plot_path = os.path.join(self.config['logging']['save_path'], f'metrics_round_{round_num+1}.png')
                        self.plot_metrics(save_path=plot_path)
                    except Exception as e:
                        print(f"Error plotting metrics: {e}")
                        traceback.print_exc()
                
            total_time = time.time() - start_time
            
            print(f"\n--- Training completed in {total_time:.2f}s ---")
            if self.logs['test_metrics']:
                print(f"Final Test Accuracy: {self.logs['test_metrics'][-1]['accuracy']:.2f}%")
                print(f"Final Alignment Loss: {self.logs['alignment_metrics']['avg_align_loss'][-1]:.4f}")
            
            # Plot final metrics
            if self.config['logging']['save_model']:
                try:
                    plot_path = os.path.join(self.config['logging']['save_path'], 'metrics_final.png')
                    self.plot_metrics(save_path=plot_path)
                except Exception as e:
                    print(f"Error plotting final metrics: {e}")
                    traceback.print_exc()
            
            return self.logs
        except Exception as e:
            print(f"Error during training: {e}")
            traceback.print_exc()
            return self.logs 