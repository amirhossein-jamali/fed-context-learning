import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=7):
        """
        A simple CNN model for the PACS dataset.
        
        Args:
            input_channels (int): Number of input channels (3 for RGB images)
            num_classes (int): Number of output classes (7 for PACS)
        """
        super(SimpleCNN, self).__init__()
        
        # Feature extraction
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Classifier
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
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
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

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
            num_classes=config['model']['num_classes']
        )
    else:
        raise ValueError(f"Unknown model: {config['model']['name']}")
    
    return model 