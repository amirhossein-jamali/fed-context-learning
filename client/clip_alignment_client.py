import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os
import traceback
from .trainer import ClientTrainer
from model.clip_alignment import CLIPAligner, CombinedLoss
from model.mapper import EmbeddingMapper

class CLIPAlignmentClient(ClientTrainer):
    def __init__(self, client_id, model, data_loader, config, mapper=None, domain_name=None):
        """
        Client trainer for federated learning with CLIP alignment.
        
        Args:
            client_id (int): ID of the client
            model (nn.Module): The global model to be trained locally
            data_loader (DataLoader): DataLoader for this client's domain
            config (dict): Configuration dictionary
            mapper (EmbeddingMapper, optional): The mapper network φ
            domain_name (str, optional): Domain name of this client (e.g., 'photo', 'cartoon')
        """
        super().__init__(client_id, model, data_loader, config, domain_name)
        
        # Initialize CLIP aligner
        self.clip_aligner = CLIPAligner(device=self.device)
        
        # Create or copy the mapper network (φ)
        if mapper is None:
            small_dim = config['model'].get('small_dim', 64)
            hidden_dim = config['model'].get('mapper_hidden_dim', 256)
            clip_dim = config['model'].get('clip_dim', 512)
            self.mapper = EmbeddingMapper(small_dim, hidden_dim, clip_dim).to(self.device)
        else:
            self.mapper = copy.deepcopy(mapper).to(self.device)
        
        # Create optimizers - one for CNN model and one for mapper
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config['federated_learning']['learning_rate'],
            momentum=0.9
        )
        
        self.mapper_optimizer = optim.SGD(
            self.mapper.parameters(),
            lr=config['federated_learning'].get('mapper_learning_rate', 
                                              config['federated_learning']['learning_rate']),
            momentum=0.9
        )
        
        # Setup combined loss function
        lambda_align = config.get('lambda_align', 0.1)
        self.combined_loss = CombinedLoss(lambda_align=lambda_align)
        
        # Setup flag for differential privacy
        self.use_dp_noise = config.get('use_dp_noise', False)
        self.dp_noise_std = config.get('dp_noise_std', 0.01)
    
    def add_noise_to_embedding(self, z_small):
        """
        Add Gaussian noise to the small embedding for enhanced privacy.
        
        Args:
            z_small (torch.Tensor): Small embedding tensor
            
        Returns:
            torch.Tensor: Noisy embedding
        """
        if self.use_dp_noise:
            noise = torch.randn_like(z_small) * self.dp_noise_std
            return z_small + noise
        return z_small
    
    def train(self, epochs=None):
        """
        Train the model with CLIP alignment for the specified number of epochs
        
        Args:
            epochs (int, optional): Number of epochs to train for. 
                                   If None, use config value.
                                   
        Returns:
            dict: Model state_dict, mapper state_dict, training metrics
        """
        try:
            if epochs is None:
                epochs = self.config['federated_learning']['local_epochs']
                
            # Set models to training mode
            self.model.train()
            self.mapper.train()
            
            total_loss = 0.0
            total_ce_loss = 0.0
            total_align_loss = 0.0
            correct = 0
            total = 0
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                epoch_ce_loss = 0.0
                epoch_align_loss = 0.0
                epoch_correct = 0
                epoch_total = 0
                
                for batch_idx, (data, target) in enumerate(self.data_loader):
                    try:
                        data, target = data.to(self.device), target.to(self.device)
                        
                        # Zero gradients
                        self.optimizer.zero_grad()
                        self.mapper_optimizer.zero_grad()
                        
                        # Forward pass through CNN
                        outputs = self.model(data)
                        logits = outputs['logits']
                        z_small = outputs['z_small']
                        
                        # Apply DP noise if enabled
                        z_small_noisy = self.add_noise_to_embedding(z_small)
                        
                        # Map small embedding to CLIP space
                        mapped_clip = self.mapper(z_small_noisy)
                        outputs['mapped_clip'] = mapped_clip
                        
                        # Get CLIP embeddings for alignment
                        with torch.no_grad():
                            clip_embeddings = self.clip_aligner.get_clip_image_embedding(data)
                        
                        # Compute combined loss
                        loss, (ce_loss, align_loss) = self.combined_loss(outputs, clip_embeddings, target)
                        
                        # Backward pass and optimize
                        loss.backward()
                        self.optimizer.step()
                        self.mapper_optimizer.step()
                        
                        # Update metrics
                        epoch_loss += loss.item()
                        epoch_ce_loss += ce_loss.item()
                        epoch_align_loss += align_loss.item()
                        pred = logits.argmax(dim=1, keepdim=True)
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
                total_ce_loss += epoch_ce_loss / max(len(self.data_loader), 1)
                total_align_loss += epoch_align_loss / max(len(self.data_loader), 1)
                correct += epoch_correct
                total += epoch_total
                
                print(f"Client {self.client_id} - Epoch {epoch+1}/{epochs} - "
                      f"Loss: {epoch_loss/max(len(self.data_loader), 1):.4f}, "
                      f"CE Loss: {epoch_ce_loss/max(len(self.data_loader), 1):.4f}, "
                      f"Align Loss: {epoch_align_loss/max(len(self.data_loader), 1):.4f}, "
                      f"Accuracy: {100. * epoch_correct / epoch_total:.2f}%")
            
            # Handle case where no data was processed at all
            if total == 0:
                print(f"Warning: No samples processed during training on client {self.client_id}")
                # Return empty model update
                return {
                    'client_id': self.client_id,
                    'domain_name': self.domain_name,
                    'model_state': self.model.state_dict(),
                    'mapper_state': self.mapper.state_dict(),  # Only send mapper weights
                    'train_size': 0,
                    'metrics': {
                        'loss': 0.0,
                        'ce_loss': 0.0,
                        'align_loss': 0.0,
                        'accuracy': 0.0
                    }
                }
            
            # Return only mapper weights and metrics
            return {
                'client_id': self.client_id,
                'domain_name': self.domain_name,
                'model_state': self.model.state_dict(),
                'mapper_state': self.mapper.state_dict(),  # Only send mapper weights
                'train_size': total,
                'metrics': {
                    'loss': total_loss / max(epochs, 1),  # Avoid division by zero
                    'ce_loss': total_ce_loss / max(epochs, 1),
                    'align_loss': total_align_loss / max(epochs, 1),
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
                'mapper_state': self.mapper.state_dict(),  # Only send mapper weights
                'train_size': 0,
                'metrics': {
                    'loss': 0.0,
                    'ce_loss': 0.0,
                    'align_loss': 0.0,
                    'accuracy': 0.0
                }
            }
    
    def update_model(self, global_model_state, global_mapper_state=None):
        """
        Update the client's model and mapper with the global state
        
        Args:
            global_model_state (OrderedDict): State dict of the global model
            global_mapper_state (OrderedDict, optional): State dict of the global mapper
        """
        self.model.load_state_dict(global_model_state)
        self.model.to(self.device)
        
        if global_mapper_state is not None:
            self.mapper.load_state_dict(global_mapper_state)
            self.mapper.to(self.device)
    
    def evaluate(self, model=None, mapper=None):
        """
        Evaluate the model on the client's data
        
        Args:
            model (nn.Module, optional): Model to evaluate. If None, use client's model.
            mapper (nn.Module, optional): Mapper to evaluate. If None, use client's mapper.
            
        Returns:
            dict: Evaluation metrics
        """
        try:
            if model is not None:
                eval_model = copy.deepcopy(model)
                eval_model.to(self.device)
            else:
                eval_model = self.model
                
            if mapper is not None:
                eval_mapper = copy.deepcopy(mapper)
                eval_mapper.to(self.device)
            else:
                eval_mapper = self.mapper
                
            # Set models to evaluation mode
            eval_model.eval()
            eval_mapper.eval()
            
            test_loss = 0
            test_ce_loss = 0
            test_align_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(self.data_loader):
                    try:
                        data, target = data.to(self.device), target.to(self.device)
                        
                        # Forward pass
                        outputs = eval_model(data)
                        logits = outputs['logits']
                        z_small = outputs['z_small']
                        
                        # Map small embedding to CLIP space
                        mapped_clip = eval_mapper(z_small)
                        outputs['mapped_clip'] = mapped_clip
                        
                        # Get CLIP embeddings
                        clip_embeddings = self.clip_aligner.get_clip_image_embedding(data)
                        
                        # Compute loss
                        loss, (ce_loss, align_loss) = self.combined_loss(outputs, clip_embeddings, target)
                        
                        # Update metrics
                        test_loss += loss.item()
                        test_ce_loss += ce_loss.item()
                        test_align_loss += align_loss.item()
                        pred = logits.argmax(dim=1, keepdim=True)
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
                        'ce_loss': 0.0,
                        'align_loss': 0.0,
                        'accuracy': 0.0
                    }
                }
            
            # Calculate average metrics
            test_loss /= max(len(self.data_loader), 1)  # Avoid division by zero
            test_ce_loss /= max(len(self.data_loader), 1)
            test_align_loss /= max(len(self.data_loader), 1)
            accuracy = 100. * correct / total
            
            return {
                'client_id': self.client_id,
                'domain_name': self.domain_name,
                'test_size': total,
                'metrics': {
                    'loss': test_loss,
                    'ce_loss': test_ce_loss,
                    'align_loss': test_align_loss,
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
                    'ce_loss': 0.0,
                    'align_loss': 0.0,
                    'accuracy': 0.0
                }
            }
    
    def save_model(self, save_path, round_num=None):
        """
        Save the client's model and mapper to disk
        
        Args:
            save_path (str): Base directory to save the model
            round_num (int, optional): Current training round number
        """
        if self.domain_name:
            # Create domain-specific directory
            domain_dir = os.path.join(save_path, f"client_{self.client_id}_{self.domain_name}")
            os.makedirs(domain_dir, exist_ok=True)
            
            # Define filenames based on round
            if round_num is not None:
                model_filename = f"model_round_{round_num}.pt"
                mapper_filename = f"mapper_round_{round_num}.pt"
            else:
                model_filename = "model_final.pt"
                mapper_filename = "mapper_final.pt"
            
            # Save model
            model_path = os.path.join(domain_dir, model_filename)
            mapper_path = os.path.join(domain_dir, mapper_filename)
            try:
                torch.save(self.model.state_dict(), model_path)
                torch.save(self.mapper.state_dict(), mapper_path)
                print(f"Client {self.client_id} ({self.domain_name}) model saved to {model_path}")
                print(f"Client {self.client_id} ({self.domain_name}) mapper saved to {mapper_path}")
            except Exception as e:
                print(f"Error saving client model or mapper: {e}")
                traceback.print_exc()
                # Continue execution even if saving fails
                pass 