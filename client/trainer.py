import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os
import traceback

class ClientTrainer:
    def __init__(self, client_id, model, data_loader, config, domain_name=None):
        """
        Client trainer for federated learning.
        
        Args:
            client_id (int): ID of the client
            model (nn.Module): The global model to be trained locally
            data_loader (DataLoader): DataLoader for this client's domain
            config (dict): Configuration dictionary
            domain_name (str, optional): Domain name of this client (e.g., 'photo', 'cartoon')
        """
        self.client_id = client_id
        self.model = copy.deepcopy(model)  # Create a local copy of the model
        self.data_loader = data_loader
        self.config = config
        self.device = torch.device(config['federated_learning']['device'])
        self.domain_name = domain_name
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(), 
            lr=config['federated_learning']['learning_rate'],
            momentum=0.9
        )
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
        
    def train(self, epochs=None):
        """
        Train the model for the specified number of epochs
        
        Args:
            epochs (int, optional): Number of epochs to train for. 
                                   If None, use config value.
                                   
        Returns:
            dict: Model state_dict, training metrics
        """
        try:
            if epochs is None:
                epochs = self.config['federated_learning']['local_epochs']
                
            # Set model to training mode
            self.model.train()
            
            total_loss = 0.0
            correct = 0
            total = 0
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                epoch_correct = 0
                epoch_total = 0
                
                for batch_idx, (data, target) in enumerate(self.data_loader):
                    try:
                        data, target = data.to(self.device), target.to(self.device)
                        
                        # Zero gradients
                        self.optimizer.zero_grad()
                        
                        # Forward pass
                        output = self.model(data)
                        loss = self.criterion(output, target)
                        
                        # Backward pass and optimize
                        loss.backward()
                        self.optimizer.step()
                        
                        # Update metrics
                        epoch_loss += loss.item()
                        pred = output.argmax(dim=1, keepdim=True)
                        epoch_correct += pred.eq(target.view_as(pred)).sum().item()
                        epoch_total += target.size(0)
                    except Exception as e:
                        print(f"Error processing batch {batch_idx} during training on client {self.client_id}: {e}")
                        traceback.print_exc()
                        continue
                
                # Skip epoch summary if no data was processed
                if epoch_total == 0:
                    print(f"Warning: No samples processed in epoch {epoch+1} for client {self.client_id}")
                    continue
                
                # Accumulate epoch metrics
                total_loss += epoch_loss / max(len(self.data_loader), 1)  # Avoid division by zero
                correct += epoch_correct
                total += epoch_total
                
                print(f"Client {self.client_id} - Epoch {epoch+1}/{epochs} - "
                      f"Loss: {epoch_loss/max(len(self.data_loader), 1):.4f}, "
                      f"Accuracy: {100. * epoch_correct / epoch_total:.2f}%")
            
            # Handle case where no data was processed at all
            if total == 0:
                print(f"Warning: No samples processed during training on client {self.client_id}")
                # Return empty model update
                return {
                    'client_id': self.client_id,
                    'domain_name': self.domain_name,
                    'model_state': self.model.state_dict(),
                    'train_size': 0,
                    'metrics': {
                        'loss': 0.0,
                        'accuracy': 0.0
                    }
                }
            
            # Return model weights and metrics
            return {
                'client_id': self.client_id,
                'domain_name': self.domain_name,
                'model_state': self.model.state_dict(),
                'train_size': total,
                'metrics': {
                    'loss': total_loss / max(epochs, 1),  # Avoid division by zero
                    'accuracy': 100. * correct / total
                }
            }
        except Exception as e:
            print(f"Error during training on client {self.client_id}: {e}")
            traceback.print_exc()
            
            # Return current model state despite error
            return {
                'client_id': self.client_id,
                'domain_name': self.domain_name,
                'model_state': self.model.state_dict(),
                'train_size': 0,
                'metrics': {
                    'loss': 0.0,
                    'accuracy': 0.0
                }
            }
    
    def evaluate(self, model=None):
        """
        Evaluate the model on the client's data
        
        Args:
            model (nn.Module, optional): Model to evaluate. If None, use client's model.
            
        Returns:
            dict: Evaluation metrics
        """
        try:
            if model is not None:
                eval_model = copy.deepcopy(model)
                eval_model.to(self.device)
            else:
                eval_model = self.model
                
            # Set model to evaluation mode
            eval_model.eval()
            
            test_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(self.data_loader):
                    try:
                        data, target = data.to(self.device), target.to(self.device)
                        
                        # Forward pass
                        output = eval_model(data)
                        loss = self.criterion(output, target)
                        
                        # Update metrics
                        test_loss += loss.item()
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
                        total += target.size(0)
                    except Exception as e:
                        print(f"Error processing batch {batch_idx} during evaluation on client {self.client_id}: {e}")
                        traceback.print_exc()
                        continue
            
            # Avoid division by zero
            if total == 0:
                print(f"Warning: No samples processed during evaluation on client {self.client_id}")
                return {
                    'client_id': self.client_id,
                    'domain_name': self.domain_name,
                    'test_size': 0,
                    'metrics': {
                        'loss': 0.0,
                        'accuracy': 0.0
                    }
                }
            
            # Calculate average metrics
            test_loss /= max(len(self.data_loader), 1)  # Avoid division by zero
            accuracy = 100. * correct / total
            
            return {
                'client_id': self.client_id,
                'domain_name': self.domain_name,
                'test_size': total,
                'metrics': {
                    'loss': test_loss,
                    'accuracy': accuracy
                }
            }
        except Exception as e:
            print(f"Error during evaluation on client {self.client_id}: {e}")
            traceback.print_exc()
            return {
                'client_id': self.client_id,
                'domain_name': self.domain_name,
                'test_size': 0,
                'metrics': {
                    'loss': 0.0,
                    'accuracy': 0.0
                }
            }
    
    def update_model(self, global_model_state):
        """
        Update the client's model with the global model state
        
        Args:
            global_model_state (OrderedDict): State dict of the global model
        """
        self.model.load_state_dict(global_model_state)
        self.model.to(self.device)
    
    def save_model(self, save_path, round_num=None):
        """
        Save the client's model to disk
        
        Args:
            save_path (str): Base directory to save the model
            round_num (int, optional): Current training round number
        """
        if self.domain_name:
            # Create domain-specific directory
            domain_dir = os.path.join(save_path, f"client_{self.client_id}_{self.domain_name}")
            os.makedirs(domain_dir, exist_ok=True)
            
            # Define filename based on round
            if round_num is not None:
                filename = f"model_round_{round_num}.pt"
            else:
                filename = "model_final.pt"
            
            # Save model
            model_path = os.path.join(domain_dir, filename)
            try:
                torch.save(self.model.state_dict(), model_path)
                print(f"Client {self.client_id} ({self.domain_name}) model saved to {model_path}")
            except Exception as e:
                print(f"Error saving client model: {e}")
                traceback.print_exc()
                # Continue execution even if saving fails
                pass 