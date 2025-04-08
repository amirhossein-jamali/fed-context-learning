import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=7, small_dim=64):
        """
        A simple CNN model for the PACS dataset with additional small embedding output.
        
        Args:
            input_channels (int): Number of input channels (3 for RGB images)
            num_classes (int): Number of output classes (7 for PACS)
            small_dim (int): Dimension of the small embedding space
        """
        super(SimpleCNN, self).__init__()
        
        # Feature extraction
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Base embedding layer
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        
        # Small embedding projection
        self.proj_small = nn.Linear(512, small_dim)
        
        # Classifier
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Input: [B, 3, 64, 64]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  # [B, 32, 32, 32]
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)  # [B, 64, 16, 16]
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)  # [B, 128, 8, 8]
        
        x = x.view(-1, 128 * 8 * 8)
        base_features = F.relu(self.fc1(x))  # [B, 512]
        
        # Get small embedding
        z_small = self.proj_small(base_features)  # [B, small_dim]
        
        # Get classification logits
        x = self.dropout(base_features)
        logits = self.fc2(x)
        
        return {
            'logits': logits,
            'z_small': z_small,
            'base_features': base_features
        }

def get_model(config):
    """
    Create a model based on config settings
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized model
    """
    if config['model']['name'] == 'SimpleCNN':
        model = SimpleCNN(
            input_channels=config['model']['input_channels'],
            num_classes=config['model']['num_classes'],
            small_dim=config['model'].get('small_dim', 64)
        )
    else:
        raise ValueError(f"Unknown model: {config['model']['name']}")
    
    return model 