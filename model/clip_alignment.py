import torch
import torch.nn as nn
import clip
from PIL import Image
import torch.nn.functional as F

class CLIPAligner:
    """Handles loading and using CLIP for computing image embeddings."""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize CLIP model for computing image embeddings.
        
        Args:
            device (str): Device to run CLIP on
        """
        self.device = device
        # Load CLIP model
        self.clip_model, self.preprocess = clip.load('ViT-B/32', device=device)
        # Freeze CLIP
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()
        
    def get_clip_image_embedding(self, images):
        """
        Get CLIP image embeddings for a batch of images.
        
        Args:
            images: Either a batch of tensors [B, C, H, W] or a single PIL Image
            
        Returns:
            torch.Tensor: CLIP image embeddings [B, 512]
        """
        with torch.no_grad():
            # Handle different input types
            if isinstance(images, Image.Image):
                # Single PIL image
                processed_image = self.preprocess(images).unsqueeze(0).to(self.device)
                return self._encode_images(processed_image)
            elif isinstance(images, torch.Tensor):
                if images.dim() == 3:
                    # Single tensor image [C, H, W]
                    images = images.unsqueeze(0)  # Add batch dimension
                
                # CLIP expects 224x224 images, so we need to resize
                # Convert to PIL Images and use CLIP's preprocessing
                batch_size = images.shape[0]
                processed_images = []
                
                for i in range(batch_size):
                    img = images[i].detach().cpu()
                    if img.shape[1] != 224 or img.shape[2] != 224:
                        # Convert to PIL image and preprocess
                        if img.min() < 0 or img.max() > 1:
                            # Normalize the image if needed
                            img = (img - img.min()) / (img.max() - img.min())
                        
                        # Convert to PIL image
                        img_np = (img.permute(1, 2, 0).numpy() * 255).astype('uint8')
                        pil_img = Image.fromarray(img_np)
                        
                        # Apply CLIP preprocessing
                        processed_img = self.preprocess(pil_img)
                    else:
                        # If already 224x224, just apply normalization if needed
                        processed_img = img
                    
                    processed_images.append(processed_img)
                
                # Stack back into a batch and move to device
                processed_batch = torch.stack(processed_images).to(self.device)
                return self._encode_images(processed_batch)
            else:
                raise TypeError(f"Unsupported image type: {type(images)}")
    
    def _encode_images(self, images):
        """
        Internal method to encode preprocessed images with CLIP.
        
        Args:
            images (torch.Tensor): Batch of preprocessed images [B, C, H, W]
            
        Returns:
            torch.Tensor: Normalized CLIP embeddings [B, 512]
        """
        # Ensure images are on the correct device
        images = images.to(self.device)
        
        # Get image features from CLIP
        image_features = self.clip_model.encode_image(images)
        
        # Normalize the features (important for similarity computations)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features


class CombinedLoss:
    """Loss function that combines classification and CLIP alignment losses."""
    
    def __init__(self, lambda_align=0.1):
        """
        Initialize combined loss function.
        
        Args:
            lambda_align (float): Weight for alignment loss term
        """
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.lambda_align = lambda_align
        
    def __call__(self, outputs, clip_embeddings, targets):
        """
        Compute combined loss.
        
        Args:
            outputs (dict): Contains 'logits' and 'mapped_clip'
            clip_embeddings (torch.Tensor): Target CLIP embeddings
            targets (torch.Tensor): Classification targets
            
        Returns:
            tuple: (total_loss, (ce_loss, align_loss))
        """
        # Classification loss
        ce_loss = self.ce_loss(outputs['logits'], targets)
        
        # Alignment loss between mapped embeddings and CLIP embeddings
        align_loss = self.mse_loss(outputs['mapped_clip'], clip_embeddings)
        
        # Combined loss with weighting
        total_loss = ce_loss + self.lambda_align * align_loss
        
        return total_loss, (ce_loss, align_loss) 