import torch
import torch.nn as nn
import clip
from PIL import Image

class CLIPAligner:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        # Load CLIP model
        self.clip_model, self.preprocess = clip.load('ViT-B/32', device=device)
        # Freeze CLIP
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()
        
    def get_clip_image_embedding(self, image):
        """Get CLIP image embedding."""
        with torch.no_grad():
            if isinstance(image, Image.Image):
                # If PIL Image, preprocess it
                image = self.preprocess(image).unsqueeze(0).to(self.device)
            elif isinstance(image, torch.Tensor) and image.dim() == 3:
                # If single tensor image, add batch dimension
                image = image.unsqueeze(0)
            
            image_features = self.clip_model.encode_image(image)
            # Normalize the features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

class CombinedLoss:
    def __init__(self, lambda_align=0.1):
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.lambda_align = lambda_align
        
    def __call__(self, model_outputs, clip_embeddings, targets):
        """
        Compute combined loss.
        
        Args:
            model_outputs (dict): Contains 'logits' and 'z_small'
            clip_embeddings (torch.Tensor): Target CLIP embeddings
            targets (torch.Tensor): Classification targets
            
        Returns:
            total_loss, (ce_loss, align_loss)
        """
        # Classification loss
        ce_loss = self.ce_loss(model_outputs['logits'], targets)
        
        # Alignment loss
        align_loss = self.mse_loss(model_outputs['mapped_clip'], clip_embeddings)
        
        # Combined loss
        total_loss = ce_loss + self.lambda_align * align_loss
        
        return total_loss, (ce_loss, align_loss) 